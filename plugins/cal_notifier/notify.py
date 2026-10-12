from dataclasses import dataclass
from enum import Enum

from ncatbot.plugin import NcatBotPlugin
from caldav import aio
from icalendar import Calendar
from icalendar.cal import Event as ICalEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.base import JobLookupError
from datetime import date, datetime, time, timedelta
from caldav import SynchronizableCalendarObjectCollection

import recurring_ical_events


#: 通知作业 id 前缀，形如 caldav_notify:{uid}:{occurrence-id-iso}
NOTIFY_JOB_PREFIX = "caldav_notify:"

#: CalDAV 轮询周期（分钟）
POLL_INTERVAL_MINUTES = 30

#: 全天事件的提醒时刻：前一天的这个时间点
ALL_DAY_NOTIFY_HOUR = 23
ALL_DAY_NOTIFY_MINUTE = 30

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
    sync_tokens: dict[str, str]   # {calendar_url: sync_token}
    snapshot: dict[str, str]      # {对象绝对URL: uid}
    target_id: int
    notify_before_minutes: int
    notify_window_days: int

    async def on_load(self) -> None:
        self.caldav_url = self.config.get("CALDAV_URL", "https://caldav.icloud.com/")
        self.caldav_user = self.config.get("CALDAV_USER", "your_username")
        self.caldav_password = self.config.get("CALDAV_PASSWORD", "your_password")
        self.target_id = int(self.config.get("target_id", 0))
        self.notify_before_minutes = int(self.config.get("notify_before_minutes", 10))
        self.notify_window_days = int(self.config.get("notify_window_days", 30))
        self._jobs_by_uid: dict[str, set[str]] = {}

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
            timezone="Asia/Shanghai",
            jobstores=self._build_jobstores(),
            job_defaults={
                'coalesce': True,          # 合并错过的运行
                'max_instances': 1,        # 同一任务最大并发实例数
                'misfire_grace_time': 300,  # 允许延迟执行的秒数
            })
        self.scheduler.add_job(
            self.poll_once,
            trigger=IntervalTrigger(minutes=POLL_INTERVAL_MINUTES),
            id="caldav_poll",
            replace_existing=True,
        )
        self.scheduler.start()
        self.logger.info(
            f"{self.name} 已加载，提前 {self.notify_before_minutes} 分钟提醒"
        )

    def _build_jobstores(self) -> dict:
        """只用内存作业存储。

        不用持久化 jobstore 的原因：轮询本身就是恢复机制——插件重启后
        sync_tokens 为空，下一轮是全量同步，所有事件重报 ADDED 并重排作业，
        30 分钟内自动恢复。而持久化会让 APScheduler pickle 整个作业
        （apscheduler/job.py:277 把 self 一起塞进 args），而 ncatbot 注入的
        self.api / self.services 等持有 websocket、事件循环，不可 pickle。
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
            collection : SynchronizableCalendarObjectCollection | None = None
            # 在 calendar 上调，返回的是集合，不是元组。
            # 公开的 get_objects_by_sync_token() 返回 list | Coroutine 联合类型，
            # Pyright 无法判定要不要 await；这里直接用内部那个真正 async 的实现。
            try:
                collection = await calendar._async_get_objects_by_sync_token(
                    sync_token=old_token,
                    load_objects=True,       # 需要完整 iCal 数据
                )
            except AttributeError:
                self.logger.error("不存在_async_get_objects_by_sync_token(),检查库实现")
            if collection is None:
                continue
            # 保存新 token（首次调用 old_token=None，会全量返回并给初始 token）
            self.sync_tokens[cal_url] = collection.sync_token

            for obj in collection:
                href = str(obj.url.canonical())

                #  快照存 {href: uid}，删除时用于反查需要取消的作业
                if obj.data is None:
                    changes.append(
                        Change(href=href, kind=ChangeKind.DELETED,
                            uid=self.snapshot.pop(href, ""))
                    )
                    continue

                # get_icalendar_instance() 直接给 icalendar.Calendar，
                # 可以 walk("VEVENT") 拿到组件，不必再 to_ical() 序列化一遍。
                # 返回类型声明可空，显式收窄，避免下游 to_ical()/_vevents() 报 Optional。
                cal = obj.get_icalendar_instance()
                if not isinstance(cal, Calendar):
                    continue
                kind = ChangeKind.UPDATED if href in self.snapshot else ChangeKind.ADDED
                uid = str(obj.icalendar_component.get("UID") or "")
                self.snapshot[href] = uid
                changes.append(
                    Change(href=href, kind=kind, calendar=cal,uid=uid)
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
                start = start.replace(tzinfo=now.tzinfo)
            if start <= now:
                return                      # 已经开始了，不补发
            notify_at = start - timedelta(minutes=self.notify_before_minutes)
            all_day = False
        elif isinstance(raw, date):
            # 全天事件：前一天 23:30 提醒
            if raw < now.date():
                return                      # 已经过去了，不补发
            start = datetime.combine(raw, time.min, tzinfo=now.tzinfo)
            notify_at = (start - timedelta(days=1)).replace(
                hour=ALL_DAY_NOTIFY_HOUR, minute=ALL_DAY_NOTIFY_MINUTE
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

        self.scheduler.add_job(
            self.send_notification,
            trigger=DateTrigger(run_date=notify_at),
            args=[str(occ.get("SUMMARY") or "(无标题)"), start.isoformat(), all_day],
            id=job_id,
            replace_existing=True,
        )
        self._jobs_by_uid.setdefault(uid, set()).add(job_id)

    async def notify_events(self, changes: list[Change]) -> None:
        now = datetime.now().astimezone()
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
        self, summary: str, start_iso: str, all_day: bool = False
    ) -> None:
        if self.target_id == 0:
            self.logger.error("未配置 target_id，无法发送日程提醒")
            return
        try:
            start = datetime.fromisoformat(start_iso)
            when = f"{start:%Y-%m-%d} 全天" if all_day else f"{start:%Y-%m-%d %H:%M}"
        except ValueError:
            when = start_iso
        text = f"日程提醒：{summary}\n开始时间：{when}"
        await self.api.qq.post_private_msg(self.target_id, text=text)
        self.logger.info(f"已发送日程提醒：{summary} @ {when}")

    async def poll_once(self) -> None:
        """定时入口：拉取变更并重排通知作业。"""
        try:
            changes = await self.get_calendar_events()
        except Exception as exc:
            self.logger.error(f"拉取 CalDAV 失败: {exc}")
            return
        if changes:
            await self.notify_events(changes)
