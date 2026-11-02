import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

import recurring_ical_events
from apscheduler.jobstores.base import JobLookupError
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from caldav import SynchronizableCalendarObjectCollection, aio
from icalendar import Calendar
from icalendar.cal import Component
from icalendar.cal import Event as ICalEvent
from icalendar.cal import Todo as ICalTodo
from ncatbot.core import registrar
from ncatbot.event.qq import PrivateMessageEvent
from ncatbot.plugin import NcatBotPlugin

#: 通知作业 id 前缀，形如 caldav_notify:{component}:{uid}:{occurrence-id-iso}
NOTIFY_JOB_PREFIX = "caldav_notify:"

#: 日程组件名
COMPONENT_EVENT = "VEVENT"

#: 待办组件名
COMPONENT_TODO = "VTODO"

#: 会触发提醒的组件类型。其余（VJOURNAL / VFREEBUSY 等）直接忽略。
NOTIFY_COMPONENTS = (COMPONENT_EVENT, COMPONENT_TODO)

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

#: 待办固定时刻提醒的默认值：不提醒，由配置决定
DEFAULT_TODO_NOTIFY_TIMES: tuple[tuple[int, int], ...] = ()

#: 手动触发同步的命令名（含前缀，首 token 精确匹配）
SYNC_COMMAND = "/sync_cal"

#: 手动同步的超时秒数：CalDAV 服务器卡住时不至于把聊天命令挂死
SYNC_TIMEOUT_SECONDS = 60

#: 固定时刻汇总提醒的作业 id 前缀，形如 caldav_todo_daily:09:00
DIGEST_JOB_PREFIX = "caldav_todo_daily:"


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
    # 组件类型：VEVENT 走日程语义，VTODO 走待办语义（按 DUE 提醒）。
    component: str = COMPONENT_EVENT


@dataclass
class TodoEntry:
    """汇总提醒里的一条待办。"""

    summary: str
    #: 最近的截止/开始时刻，None 表示没有日期
    due: datetime | None = None
    all_day: bool = False
    #: 时间的措辞：截止时间 / 开始时间
    label: str = ""

    def is_expired(self, now: datetime) -> bool:
        """过了截止时间就不再进汇总；全天待办按当天结束算。"""
        if self.due is None:
            return False
        end = self.due + timedelta(days=1) if self.all_day else self.due
        return end <= now

    def describe(self) -> str:
        if self.due is None:
            return f"{self.summary}（未设置日期）"
        when = f"{self.due:%Y-%m-%d}" if self.all_day else f"{self.due:%Y-%m-%d %H:%M}"
        return f"{self.summary}（{self.label} {when}）"


def _owner_key(component: str, uid: str) -> str:
    """作业归属键：组件类型 + UID。

    带上前缀是因为 VEVENT 与 VTODO 可能撞 UID，而两者提醒语义不同，
    不隔开就会互相取消对方的作业，把提醒吞掉。
    """
    return f"{component}:{uid}"


def _primary_component(cal: Calendar) -> Component | None:
    """取资源里的主组件（跳过 VTIMEZONE）。"""
    for sub in cal.subcomponents:
        name = str(getattr(sub, "name", ""))
        if name and name != "VTIMEZONE":
            return sub
    return None


def _todo_has_time(cal: Calendar) -> bool:
    """待办必须自带 DTSTART 或 DUE 才排提醒。

    纯清单型待办（两者都没有）很常见，而 recurring_ical_events 会给它们
    合成 DTSTART=1970-01-01 / DUE=2038-01-01，展开后再判断已经晚了。
    """
    primary = _primary_component(cal)
    if primary is None:
        return False
    return primary.get("DTSTART") is not None or primary.get("DUE") is not None


class NotifyPlugin(NcatBotPlugin):
    caldav_url: str
    caldav_user: str
    caldav_password: str
    davclient: aio.AsyncDAVClient
    scheduler: AsyncIOScheduler
    sync_tokens: dict[str, str]  # {calendar_url: sync_token}
    snapshot: dict[str, tuple[str, str]]  # {对象绝对URL: (uid, 组件类型)}
    target_id: int
    notify_before_minutes: int
    notify_window_days: int
    all_day_notify_hour: int
    all_day_notify_minute: int
    todo_notify_times: tuple[tuple[int, int], ...]
    tz: ZoneInfo
    _jobs_by_owner: dict[str, set[str]]
    _todo_snapshot: dict[str, TodoEntry]
    _last_full_sync: datetime
    _sync_lock: asyncio.Lock

    async def on_load(self) -> None:
        self.caldav_url = self.config.get("CALDAV_URL", "https://caldav.icloud.com/")
        self.caldav_user = self.config.get("CALDAV_USER", "your_username")
        self.caldav_password = self.config.get("CALDAV_PASSWORD", "your_password")
        self.target_id = int(self.config.get("target_id", 0))
        self.notify_before_minutes = max(
            0, int(self.config.get("notify_before_minutes", 10))
        )
        self.notify_window_days = max(1, int(self.config.get("notify_window_days", 30)))
        self.all_day_notify_hour = int(
            self.config.get("all_day_notify_hour", DEFAULT_ALL_DAY_NOTIFY_HOUR)
        )
        self.all_day_notify_minute = int(
            self.config.get("all_day_notify_minute", DEFAULT_ALL_DAY_NOTIFY_MINUTE)
        )
        self.todo_notify_times = self._parse_notify_times(
            self.config.get("todo_notify_times")
        )
        self.tz = self._resolve_timezone(
            str(self.config.get("timezone", DEFAULT_TIMEZONE))
        )
        self._jobs_by_owner = {}
        self._todo_snapshot = {}
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
                "coalesce": True,  # 合并错过的运行
                "max_instances": 1,  # 同一任务最大并发实例数
                "misfire_grace_time": 300,  # 允许延迟执行的秒数
            },
        )
        self.scheduler.add_job(
            self._scheduled_poll,
            trigger=IntervalTrigger(minutes=POLL_INTERVAL_MINUTES),
            id="caldav_poll",
            replace_existing=True,
            # 立即跑第一轮：否则启动后 30 分钟内一条提醒都排不上，
            # 而错过的提醒点又不补发，重启等于白等半个钟头。
            next_run_time=datetime.now(self.tz),
        )
        # 每个时刻只挂一个汇总作业，待办多时才不会退化成 待办数 × 时刻数 个作业
        for hour, minute in self.todo_notify_times:
            hhmm = f"{hour:02d}:{minute:02d}"
            self.scheduler.add_job(
                self.send_todo_digest,
                trigger=CronTrigger(hour=hour, minute=minute, timezone=self.tz),
                args=[hhmm],
                id=f"{DIGEST_JOB_PREFIX}{hhmm}",
                replace_existing=True,
            )
        self.scheduler.start()
        self.logger.info(
            f"{self.name} 已加载，提前 {self.notify_before_minutes} 分钟提醒"
        )

    def _parse_notify_times(self, value: object) -> tuple[tuple[int, int], ...]:
        """解析待办固定时刻提醒。

        接受 YAML 列表（``["09:00", "20:00"]``）或逗号/空格分隔的字符串；
        非法项记一条警告后跳过，不因为写错一个时刻就整段失效。
        """
        if value is None or value == "":
            return DEFAULT_TODO_NOTIFY_TIMES
        if isinstance(value, str):
            raw = [item for item in value.replace(",", " ").split() if item]
        elif isinstance(value, (list, tuple)):
            raw = [str(item) for item in value]
        else:
            self.logger.warning(f"todo_notify_times 格式不对：{value!r}")
            return DEFAULT_TODO_NOTIFY_TIMES

        times: list[tuple[int, int]] = []
        for item in raw:
            try:
                hour_text, minute_text = item.strip().split(":")
                hour, minute = int(hour_text), int(minute_text)
            except ValueError:
                self.logger.warning(f"忽略无效的提醒时刻：{item!r}")
                continue
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                self.logger.warning(f"忽略越界的提醒时刻：{item!r}")
                continue
            if (hour, minute) not in times:
                times.append((hour, minute))
        return tuple(times)

    def _resolve_timezone(self, name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except KeyError, ValueError:
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
                    load_objects=True,  # 需要完整 iCal 数据
                )
            )
            # 保存新 token（首次调用 old_token=None，会全量返回并给初始 token）
            self.sync_tokens[cal_url] = collection.sync_token
            seen: set[str] = set()

            for obj in collection:
                href = str(obj.url.canonical())
                seen.add(href)

                # 快照存 {href: (uid, 组件类型)}，删除时用于反查需要取消的作业
                if obj.data is None:
                    uid, component = self.snapshot.pop(href, ("", COMPONENT_EVENT))
                    changes.append(
                        Change(
                            href=href,
                            kind=ChangeKind.DELETED,
                            uid=uid,
                            component=component,
                        )
                    )
                    continue

                # get_icalendar_instance() 直接给 icalendar.Calendar。
                # 返回类型声明可空，显式收窄，避免下游访问报 Optional。
                cal = obj.get_icalendar_instance()
                if not isinstance(cal, Calendar):
                    continue
                primary = _primary_component(cal)
                if primary is None:
                    continue
                # VEVENT 和 VTODO 的提醒语义不同，必须按组件类型分流
                component = str(getattr(primary, "name", COMPONENT_EVENT))
                if component not in NOTIFY_COMPONENTS:
                    continue
                kind = ChangeKind.UPDATED if href in self.snapshot else ChangeKind.ADDED
                uid = str(primary.get("UID") or "")
                self.snapshot[href] = (uid, component)
                changes.append(
                    Change(
                        href=href,
                        kind=kind,
                        calendar=cal,
                        uid=uid,
                        component=component,
                    )
                )

            # 全量同步不返回删除条目：拿"本次拉到的集合"和快照对账，
            # 否则上一轮之后被删掉的事件会带着已排的作业一直留着。
            # 对象 URL 是日历 URL 的子路径，按前缀归属。
            if old_token is None and cal_url:
                for href in [h for h in self.snapshot if h.startswith(cal_url)]:
                    if href in seen:
                        continue
                    uid, component = self.snapshot.pop(href, ("", COMPONENT_EVENT))
                    changes.append(
                        Change(
                            href=href,
                            kind=ChangeKind.DELETED,
                            uid=uid,
                            component=component,
                        )
                    )
        return changes

    # ---------- 通知调度 ----------

    def _cancel_notifications(self, owner: str) -> None:
        """撤掉某个归属键（组件类型 + UID）下所有待发的通知作业。"""
        for jid in self._jobs_by_owner.pop(owner, ()):
            try:
                self.scheduler.remove_job(jid)
            except JobLookupError:
                pass

    def _forget_job(self, owner: str, job_id: str) -> None:
        """作业是一次性的：触发后 APScheduler 会删掉它，索引也得同步摘掉。"""
        self._jobs_by_owner.get(owner, set()).discard(job_id)

    def _schedule_occurrence(
        self, occ: Component, now: datetime, component: str
    ) -> None:
        # 被取消的实例直接跳过
        if self._is_cancelled(occ):
            return

        if component == COMPONENT_TODO:
            if self._todo_is_done(occ):
                return
            anchor = self._todo_anchor(occ, now)
            if anchor is None:
                return
            start, all_day, time_label, raw = anchor
        else:
            raw = self._raw_start(occ)
            if raw is None:
                return
            time_label = "开始时间"
            start = self._localize(raw, now)
            if start is None:
                return
            all_day = not isinstance(raw, datetime)

        # 定时：提前 notify_before_minutes 分钟提醒；全天：前一天固定时刻提醒
        if all_day:
            notify_at = (start - timedelta(days=1)).replace(
                hour=self.all_day_notify_hour, minute=self.all_day_notify_minute
            )
        else:
            notify_at = start - timedelta(minutes=self.notify_before_minutes)

        uid = str(occ.get("UID") or "")
        owner = _owner_key(component, uid)
        rid = occ.get("RECURRENCE-ID")
        anchor_dt = rid.dt if rid is not None else raw

        # 时间在这里就格式化好：作业参数不用再扛 ISO 串给运行时反解
        if all_day:
            when = (
                f"{start:%Y-%m-%d}"
                if component == COMPONENT_TODO
                else f"{start:%Y-%m-%d} 全天"
            )
        else:
            when = f"{start:%Y-%m-%d %H:%M}"

        # 提醒点已过就不补发：插件唯一的恢复机制是轮询，补发需要"已发送"状态，
        # 而状态又需要持久化，会把 MemoryJobStore 的简化全部吐回去。
        # 与 misfire_grace_time=300 的语义一致：错过就错过。
        if notify_at <= now:
            return

        job_id = f"{NOTIFY_JOB_PREFIX}{owner}:{anchor_dt.isoformat()}"
        self.scheduler.add_job(
            self.send_notification,
            trigger=DateTrigger(run_date=notify_at),
            args=[
                str(occ.get("SUMMARY") or "(无标题)"),
                f"{time_label}：{when}",
                owner,
                job_id,
            ],
            id=job_id,
            replace_existing=True,
        )
        self._jobs_by_owner.setdefault(owner, set()).add(job_id)

    @staticmethod
    def _raw_start(occ: Component) -> date | datetime | None:
        dtstart = occ.get("DTSTART")
        return dtstart.dt if dtstart is not None else None

    def _localize(self, raw: object, now: datetime) -> datetime | None:
        """把 ical 的时间值换算成带本地时区的 datetime；过去的值返回 None。"""
        if isinstance(raw, datetime):
            # UTC / 别的 TZID 一律换算到本地时区，消息里才不会出现 UTC 时间
            start = (
                raw.replace(tzinfo=self.tz)
                if raw.tzinfo is None
                else raw.astimezone(self.tz)
            )
            return None if start <= now else start
        if isinstance(raw, date):
            if raw < now.date():
                return None
            return datetime.combine(raw, time.min, tzinfo=self.tz)
        return None

    def _todo_anchor(
        self, occ: Component, now: datetime
    ) -> tuple[datetime, bool, str, date | datetime] | None:
        """待办的锚点：(本地时刻, 是否全天, 措辞, 原始值)。

        DUE 是截止时间，优先；没有 DUE 才退回 DTSTART。已经过去返回 None。
        """
        due = occ.get("DUE")
        if due is not None:
            raw, label = due.dt, "截止时间"
        else:
            raw = self._raw_start(occ)
            if raw is None:
                return None
            label = "开始时间"
        start = self._localize(raw, now)
        if start is None:
            return None
        return start, not isinstance(raw, datetime), label, raw

    @staticmethod
    def _is_cancelled(occ: Component) -> bool:
        status = occ.get("STATUS")
        return status is not None and str(status).upper() == "CANCELLED"

    @staticmethod
    def _todo_is_done(occ: Component) -> bool:
        """已完成的待办（STATUS:COMPLETED 或 PERCENT-COMPLETE:100）不再提醒。"""
        status = occ.get("STATUS")
        if status is not None and str(status).upper() == "COMPLETED":
            return True
        percent = occ.get("PERCENT-COMPLETE")
        if percent is not None:
            try:
                return int(percent) >= 100
            except TypeError, ValueError:
                return False
        return False

    async def notify_events(self, changes: list[Change]) -> None:
        now = datetime.now(self.tz)
        horizon = now + timedelta(days=self.notify_window_days)

        for change in changes:
            if change.uid:
                self._cancel_notifications(_owner_key(change.component, change.uid))
            if change.kind is ChangeKind.DELETED:
                if change.component == COMPONENT_TODO:
                    self._todo_snapshot.pop(change.uid, None)
                continue
            if change.calendar is None:
                continue
            # 无 DTSTART/DUE 的待办没有一次性提醒可排，但固定时刻汇总里要有它
            if change.component == COMPONENT_TODO and not _todo_has_time(
                change.calendar
            ):
                self._record_undated_todo(change)
                continue

            try:
                # 用完整 Calendar 展开，自动处理 RRULE / EXDATE / 覆盖实例。
                # components 必须显式给：默认只展开 VEVENT，待办会一条都出不来。
                occurrences = recurring_ical_events.of(
                    change.calendar, components=(change.component,)
                ).between(now, horizon)
            except (KeyError, TypeError, ValueError) as exc:
                self.logger.warning(f"展开事件失败 {change.href}: {exc}")
                continue

            occurrences = [
                occ for occ in occurrences if isinstance(occ, (ICalEvent, ICalTodo))
            ]
            if change.component == COMPONENT_TODO:
                self._record_todo(change.uid, occurrences, now)
            for occ in occurrences:
                self._schedule_occurrence(occ, now, change.component)

    def _record_undated_todo(self, change: Change) -> None:
        """把无日期的待办记进汇总清单；已完成/已取消的踢掉。"""
        primary = _primary_component(change.calendar) if change.calendar else None
        if (
            primary is None
            or self._todo_is_done(primary)
            or self._is_cancelled(primary)
        ):
            self._todo_snapshot.pop(change.uid, None)
            return
        self._todo_snapshot[change.uid] = TodoEntry(
            str(primary.get("SUMMARY") or "(无标题)")
        )

    def _record_todo(
        self, uid: str, occurrences: Sequence[Component], now: datetime
    ) -> None:
        """重复待办只保留最近一次实例的时间，供固定时刻汇总使用。"""
        for occ in occurrences:
            if self._todo_is_done(occ) or self._is_cancelled(occ):
                continue
            anchor = self._todo_anchor(occ, now)
            if anchor is None:
                continue
            start, all_day, label, _ = anchor
            self._todo_snapshot[uid] = TodoEntry(
                str(occ.get("SUMMARY") or "(无标题)"),
                due=start,
                all_day=all_day,
                label=label,
            )
            return
        # 没有未完成且未过期的实例：不再进汇总
        self._todo_snapshot.pop(uid, None)

    async def send_todo_digest(self, hhmm: str) -> None:
        """固定时刻提醒：把当前该提醒的待办汇总成一条消息发出去。"""
        if self.target_id == 0:
            self.logger.error("未配置 target_id，无法发送待办提醒")
            return
        now = datetime.now(self.tz)
        for uid in [
            uid for uid, entry in self._todo_snapshot.items() if entry.is_expired(now)
        ]:
            del self._todo_snapshot[uid]

        entries = sorted(
            self._todo_snapshot.values(),
            key=lambda entry: (entry.due is None, entry.due or now),
        )
        if not entries:
            return
        lines = "\n".join(f"· {entry.describe()}" for entry in entries)
        await self.api.qq.post_private_msg(
            self.target_id, text=f"待办提醒（{hhmm}）\n{lines}"
        )
        self.logger.info(f"已发送待办提醒（{hhmm}）：{len(entries)} 条")

    async def send_notification(
        self, summary: str, body: str, owner: str, job_id: str
    ) -> None:
        try:
            if self.target_id == 0:
                self.logger.error("未配置 target_id，无法发送日程提醒")
                return
            label = (
                "待办提醒" if owner.split(":", 1)[0] == COMPONENT_TODO else "日程提醒"
            )
            await self.api.qq.post_private_msg(
                self.target_id, text=f"{label}：{summary}\n{body}"
            )
            self.logger.info(f"已发送{label}：{summary} @ {body}")
        finally:
            # 作业是一次性的，触发后 APScheduler 会删掉它，索引也同步摘掉
            self._forget_job(owner, job_id)

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
        except (OSError, RuntimeError, ValueError) as exc:
            self.logger.error(f"拉取 CalDAV 失败: {exc}")
            return -1, self._pending_job_count()
        if changes:
            await self.notify_events(changes)
        return len(changes), self._pending_job_count()

    def _pending_job_count(self) -> int:
        return sum(len(jids) for jids in self._jobs_by_owner.values())

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
                self.logger.error(
                    f"{SYNC_COMMAND} 同步超时（>{SYNC_TIMEOUT_SECONDS}s）"
                )
                await event.reply(
                    text=f"同步超时（>{SYNC_TIMEOUT_SECONDS}s），详见日志。"
                )
                return
            except (OSError, RuntimeError, ValueError) as exc:
                self.logger.error(f"{SYNC_COMMAND} 同步异常: {exc}")
                await event.reply(text="同步出错，详见日志。")
                return

        if count < 0:
            await event.reply(text="同步失败，详见日志。")
            return
        await event.reply(text=f"同步完成：{count} 条变更，当前待发提醒 {jobs} 条。")

    # 只挂私聊。用 registrar.qq.* 而不是顶层 registrar.on_private_command：
    # 后者 platform 默认 None，不会挂 PlatformFilter，命令会在其它平台的
    # 私聊里误触发。
    @registrar.qq.on_private_command(SYNC_COMMAND, ignore_case=True)
    async def cmd_sync_cal_private(self, event: PrivateMessageEvent) -> None:
        await self._handle_sync_command(event)
