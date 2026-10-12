from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo
import asyncio

from ncatbot.core import registrar
from ncatbot.plugin import NcatBotPlugin
from ncatbot.event.qq import PrivateMessageEvent
from caldav import aio
from icalendar import Calendar
from icalendar.cal import Event as ICalEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.base import JobLookupError
from caldav import SynchronizableCalendarObjectCollection

import recurring_ical_events


#: 通知作业 id 前缀，形如 caldav_notify:{uid}:{occurrence-id-iso}
NOTIFY_JOB_PREFIX = "caldav_notify:"

#: CalDAV 轮询周期（分钟）
POLL_INTERVAL_MINUTES = 30

#: 强制全量同步的周期（小时）。滚动窗口靠它推进，必须远小于 notify_window_days，
#: 否则窗口边缘的实例来不及被排上。
FULL_SYNC_INTERVAL_HOURS = 12

#: 全天事件的默认提醒时刻：前一天的这个时间点
DEFAULT_ALL_DAY_NOTIFY_HOUR = 23
DEFAULT_ALL_DAY_NOTIFY_MINUTE = 30

#: 缺少时区信息的事件（尤其全天事件）按这个时区解释
DEFAULT_TIMEZONE = "Asia/Shanghai"

#: 手动触发同步的命令名（含前缀，首 token 精确匹配）
SYNC_COMMAND = "/sync_cal"

#: 手动同步的超时秒数：CalDAV 服务器卡住时不至于把聊天命令挂死
SYNC_TIMEOUT_SECONDS = 60


class ChangeKind(Enum):
    ADDED = "added"
    UPDATED = "updated"
    DELETED = "deleted"


@dataclass
class Change:
    uid: str
    href: str
    kind: ChangeKind
    # 完整 VCALENDAR：展开重复事件必须带上主事件 + 覆盖实例，不能拆开。
    calendar: Calendar | None = None


class NotifyPlugin(NcatBotPlugin):
    caldav_url: str
    caldav_user: str
    caldav_password: str
    davclient: "aio.AsyncDAVClient"
    scheduler: AsyncIOScheduler
    sync_tokens: dict[str, str]   # {calendar_url: sync_token}
    snapshot: dict[str, str]      # {对象绝对URL: uid}
    target_id: int
    notify_before_minutes: int
    notify_window_days: int
    all_day_notify_hour: int
    all_day_notify_minute: int
    tz: ZoneInfo
    _jobs_by_uid: dict[str, set[str]]
    _last_full_sync: datetime
    _sync_lock: asyncio.Lock

    async def on_load(self) -> None:
        self.caldav_url = self.config.get("CALDAV_URL", "https://caldav.icloud.com/")
        self.caldav_user = self.config.get("CALDAV_USER", "your_username")
        self.caldav_password = self.config.get("CALDAV_PASSWORD", "your_password")
        self.target_id = int(self.config.get("target_id", 0))
        self.notify_before_minutes = max(0, int(self.config.get("notify_before_minutes", 10)))
        self.notify_window_days = max(1, int(self.config.get("notify_window_days", 30)))
        self.all_day_notify_hour = int(
            self.config.get("all_day_notify_hour", DEFAULT_ALL_DAY_NOTIFY_HOUR)
        )
        self.all_day_notify_minute = int(
            self.config.get("all_day_notify_minute", DEFAULT_ALL_DAY_NOTIFY_MINUTE)
        )
        self.tz = self._resolve_timezone(str(self.config.get("timezone", DEFAULT_TIMEZONE)))
        self._jobs_by_uid = {}
        # 手动 /sync_cal 与定时轮询共用这把锁，避免两者交错重排作业
        self._sync_lock = asyncio.Lock()
        # 启动后首轮就是全量（sync_tokens 为空），从加载时刻起算全量周期
        self._last_full_sync = datetime.now(self.tz)

        if self.target_id == 0:
            self.logger.error("未配置 target_id，日程提醒将无法发送")

        self.davclient = await aio.get_async_davclient(
            url=self.caldav_url,
            username=self.caldav_user,
            password=self.caldav_password,
        )
        self.sync_tokens = {}
        self.snapshot = {}

        self.scheduler = AsyncIOScheduler(
            timezone=self.tz,
            jobstores=self._build_jobstores(),
            job_defaults={
                'coalesce': True,          # 合并错过的运行
                'max_instances': 1,        # 同一任务最大并发实例数
                'misfire_grace_time': 300,  # 允许延迟执行的秒数
            })
        self.scheduler.add_job(
            self._scheduled_poll,
            trigger=IntervalTrigger(minutes=POLL_INTERVAL_MINUTES),
            id="caldav_poll",
            replace_existing=True,
            # 立即跑第一轮：否则启动后 30 分钟内一条提醒都排不上，
            # 而错过的提醒点又不补发，重启等于白等半个钟头。
            next_run_time=datetime.now(self.tz),
        )
        self.scheduler.start()
        self.logger.info(
            f"{self.name} 已加载，提前 {self.notify_before_minutes} 分钟提醒"
        )

    def _resolve_timezone(self, name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except (KeyError, ValueError):
            self.logger.warning(f"时区 {name} 不可用，回退到 {DEFAULT_TIMEZONE}")
            return ZoneInfo(DEFAULT_TIMEZONE)

    def _build_jobstores(self) -> dict:
        """只用内存作业存储。

        不用持久化 jobstore 的原因：轮询本身就是恢复机制——插件重启后
        sync_tokens 为空，下一轮是全量同步，所有事件重报 ADDED 并重排作业，
        下一轮轮询内自动恢复。而持久化会让 APScheduler 序列化整个作业：
        绑定方法会把 self 一起写进 args，而 NcatBot 注入的 self.api 等
        持有 websocket、事件循环，不可序列化。
        """
        return {"default": MemoryJobStore()}

    async def on_close(self) -> None:
        if getattr(self, "scheduler", None) is not None:
            self.scheduler.shutdown(wait=False)
        if getattr(self, "davclient", None) is not None:
            await self.davclient.close()
        self.logger.info(f"{self.name} 已卸载")

    # ---------- CalDAV 同步 ----------

    async def get_calendar_events(self) -> list[Change]:
        principal = await self.davclient.get_principal()
        calendars = await self.davclient.get_calendars(principal)

        changes: list[Change] = []
        for calendar in calendars:
            cal_url = str(calendar.url)
            old_token = self.sync_tokens.get(cal_url)
            # 私有 API：公开的 get_objects_by_sync_token() 返回 list | Coroutine
            # 联合类型，Pyright 无法判定要不要 await。先探测属性存在再调用，
            # 免得 except AttributeError 把库内部的同名异常一起吞掉。
            if not hasattr(calendar, "_async_get_objects_by_sync_token"):
                self.logger.error("不存在_async_get_objects_by_sync_token(),检查库实现")
                continue
            collection: SynchronizableCalendarObjectCollection = (
                await calendar._async_get_objects_by_sync_token(
                    sync_token=old_token,
                    load_objects=True,       # 需要完整 iCal 数据
                )
            )
            # 保存新 token（首次调用 old_token=None，会全量返回并给初始 token）
            self.sync_tokens[cal_url] = collection.sync_token
            seen: set[str] = set()

            for obj in collection:
                href = str(obj.url.canonical())
                seen.add(href)

                # 快照存 {href: uid}，删除时用于反查需要取消的作业
                if obj.data is None:
                    changes.append(
                        Change(href=href, kind=ChangeKind.DELETED,
                            uid=self.snapshot.pop(href, ""))
                    )
                    continue

                # get_icalendar_instance() 直接给 icalendar.Calendar。
                # 返回类型声明可空，显式收窄，避免下游访问报 Optional。
                cal = obj.get_icalendar_instance()
                if not isinstance(cal, Calendar):
                    continue
                kind = ChangeKind.UPDATED if href in self.snapshot else ChangeKind.ADDED
                uid = str(obj.icalendar_component.get("UID") or "")
                self.snapshot[href] = uid
                changes.append(
                    Change(href=href, kind=kind, calendar=cal, uid=uid)
                )

            # 全量同步不返回删除条目：拿"本次拉到的集合"和快照对账，
            # 否则上一轮之后被删掉的事件会带着已排的作业一直留着。
            # 对象 URL 是日历 URL 的子路径，按前缀归属。
            if old_token is None and cal_url:
                for href in [h for h in self.snapshot if h.startswith(cal_url)]:
                    if href in seen:
                        continue
                    changes.append(
                        Change(href=href, kind=ChangeKind.DELETED,
                            uid=self.snapshot.pop(href, ""))
                    )
        return changes

    # ---------- 通知调度 ----------

    def _cancel_notifications(self, uid: str) -> None:
        """撤掉某个 UID 下所有待发的通知作业。"""
        for jid in self._jobs_by_uid.pop(uid, ()):
            try:
                self.scheduler.remove_job(jid)
            except JobLookupError:
                pass

    def _forget_job(self, uid: str, job_id: str) -> None:
        """作业是一次性的：触发后 APScheduler 会删掉它，索引也得同步摘掉。"""
        self._jobs_by_uid.get(uid, set()).discard(job_id)

    def _schedule_occurrence(self, occ: ICalEvent, now: datetime) -> None:
        # 被取消的实例直接跳过
        status = occ.get("STATUS")
        if status is not None and str(status).upper() == "CANCELLED":
            return

        dtstart = occ.get("DTSTART")
        if dtstart is None:
            return
        raw = dtstart.dt

        # all_day 决定消息里怎么描述时间
        if isinstance(raw, datetime):
            # 定时事件：提前 notify_before_minutes 分钟提醒
            start = raw
            if start.tzinfo is None:
                start = start.replace(tzinfo=self.tz)
            if start <= now:
                return                      # 已经开始了，不补发
            notify_at = start - timedelta(minutes=self.notify_before_minutes)
            all_day = False
        elif isinstance(raw, date):
            # 全天事件：前一天 all_day_notify_hour:minute 提醒
            if raw < now.date():
                return                      # 已经过去了，不补发
            start = datetime.combine(raw, time.min, tzinfo=self.tz)
            notify_at = (start - timedelta(days=1)).replace(
                hour=self.all_day_notify_hour, minute=self.all_day_notify_minute
            )
            all_day = True
        else:
            return

        if notify_at <= now:
            # 提醒点已过：插件刚启动 / 重启 / 通知点落在两轮轮询之间。
            # 不补发——本插件唯一的恢复机制就是轮询本身；补发需要"已发送"状态，
            # 而状态又需要持久化，会把 MemoryJobStore 的简化全部吐回去。
            # 与 misfire_grace_time=300 的语义保持一致：错过就错过。
            return

        uid = str(occ.get("UID") or "")
        rid = occ.get("RECURRENCE-ID")
        anchor = rid.dt if rid is not None else raw
        job_id = f"{NOTIFY_JOB_PREFIX}{uid}:{anchor.isoformat()}"

        # 时间在这里就格式化好：作业参数不用再扛 ISO 串给运行时反解
        when = f"{start:%Y-%m-%d} 全天" if all_day else f"{start:%Y-%m-%d %H:%M}"
        self.scheduler.add_job(
            self.send_notification,
            trigger=DateTrigger(run_date=notify_at),
            args=[str(occ.get("SUMMARY") or "(无标题)"), when, uid, job_id],
            id=job_id,
            replace_existing=True,
        )
        self._jobs_by_uid.setdefault(uid, set()).add(job_id)

    async def notify_events(self, changes: list[Change]) -> None:
        now = datetime.now(self.tz)
        horizon = now + timedelta(days=self.notify_window_days)

        for change in changes:
            if change.uid:
                self._cancel_notifications(change.uid)
            if change.kind is ChangeKind.DELETED:
                continue
            if change.calendar is None:
                continue

            try:
                # 用完整 Calendar 展开，自动处理 RRULE / EXDATE / 覆盖实例
                occurrences = recurring_ical_events.of(change.calendar).between(now, horizon)
            except Exception as exc:
                self.logger.warning(f"展开事件失败 {change.href}: {exc}")
                continue

            for occ in occurrences:
                # between() 静态返回 list[Component]，运行时是 VEVENT，这里收窄。
                if not isinstance(occ, ICalEvent):
                    continue
                self._schedule_occurrence(occ, now)

    async def send_notification(
        self, summary: str, when: str, uid: str, job_id: str
    ) -> None:
        try:
            if self.target_id == 0:
                self.logger.error("未配置 target_id，无法发送日程提醒")
                return
            text = f"日程提醒：{summary}\n开始时间：{when}"
            await self.api.qq.post_private_msg(self.target_id, text=text)
            self.logger.info(f"已发送日程提醒：{summary} @ {when}")
        finally:
            self._forget_job(uid, job_id)

    async def _scheduled_poll(self) -> None:
        """定时任务入口：拿锁跑一次普通轮询。"""
        async with self._sync_lock:
            await self.poll_once()

    async def poll_once(self, force_full: bool = False) -> tuple[int, int]:
        """定时入口：拉取变更并重排通知作业。

        Args:
            force_full: 丢弃 sync token 强制全量。带 token 的 REPORT 只返回
                变更过的对象，所以"手动同步"必须走全量，否则在没有新变更时
                拉回来是空的，看起来像没生效。

        Returns:
            (变更条数, 当前待发作业数)；拉取失败返回 (-1, 作业数)
        """
        now = datetime.now(self.tz)
        if force_full or now - self._last_full_sync >= timedelta(
            hours=FULL_SYNC_INTERVAL_HOURS
        ):
            # 带 token 的 REPORT 只返回变更过的对象，纯重复事件不会被重新展开，
            # 于是"原本在窗口外、后来滚进窗口"的实例永远排不上提醒——提醒会在
            # 事件最后一次被编辑后 notify_window_days 天左右静默冻结。
            # 丢掉 token 强制全量，把窗口重新填满，顺带自愈任何状态漂移。
            self.logger.info("丢弃 sync token 重新全量拉取")
            self.sync_tokens.clear()
            self._last_full_sync = now
        try:
            changes = await self.get_calendar_events()
        except Exception as exc:
            self.logger.error(f"拉取 CalDAV 失败: {exc}")
            return -1, self._pending_job_count()
        if changes:
            await self.notify_events(changes)
        return len(changes), self._pending_job_count()

    def _pending_job_count(self) -> int:
        return sum(len(jids) for jids in self._jobs_by_uid.values())

    # ---------- 聊天命令 ----------

    async def _handle_sync_command(self, event) -> None:
        """/sync_cal：立刻全量同步一次，并回报结果。"""
        # 与定时 poll_once 互斥：两者都会清 sync token 并重排作业，
        # 交错执行会让作业表被互相覆盖成半截状态。
        async with self._sync_lock:
            try:
                async with asyncio.timeout(SYNC_TIMEOUT_SECONDS):
                    count, jobs = await self.poll_once(force_full=True)
            except TimeoutError:
                # CalDAV 服务器卡住：给个回执，别让命令石沉大海。
                # 锁已随 async with 释放，后续指令仍可用。
                self.logger.error(f"{SYNC_COMMAND} 同步超时（>{SYNC_TIMEOUT_SECONDS}s）")
                await event.reply(text=f"同步超时（>{SYNC_TIMEOUT_SECONDS}s），详见日志。")
                return
            except Exception as exc:
                self.logger.error(f"{SYNC_COMMAND} 同步异常: {exc}")
                await event.reply(text="同步出错，详见日志。")
                return

        if count < 0:
            await event.reply(text="同步失败，详见日志。")
            return
        await event.reply(
            text=f"同步完成：{count} 条变更，当前待发提醒 {jobs} 条。"
        )

    # 只挂私聊。用 registrar.qq.* 而不是顶层 registrar.on_private_command：
    # 后者 platform 默认 None，不会挂 PlatformFilter，命令会在其它平台的
    # 私聊里误触发。
    @registrar.qq.on_private_command(SYNC_COMMAND, ignore_case=True)
    async def cmd_sync_cal_private(self, event: PrivateMessageEvent) -> None:
        await self._handle_sync_command(event)
