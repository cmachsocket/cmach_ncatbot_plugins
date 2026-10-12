from dataclasses import dataclass, field
from enum import Enum

from ncatbot.plugin import NcatBotPlugin
from caldav import aio
from icalendar import Calendar
from icalendar.cal import Component
from icalendar.cal import Event as ICalEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.base import JobLookupError
from datetime import datetime, timedelta

import recurring_ical_events


#: 通知作业 id 前缀，形如 caldav_notify:{uid}:{recurrence-id-iso}
NOTIFY_JOB_PREFIX = "caldav_notify:"

#: CalDAV 轮询周期（分钟）
POLL_INTERVAL_MINUTES = 30

#: 通知作业 id 前缀，形如 caldav_notify:{uid}:{recurrence-id-iso}
NOTIFY_JOB_PREFIX = "caldav_notify:"

#: CalDAV 轮询周期（分钟）
POLL_INTERVAL_MINUTES = 30


class ChangeKind(Enum):
    ADDED = "added"
    UPDATED = "updated"
    DELETED = "deleted"


@dataclass
class Change:
    href: str
    kind: ChangeKind
    # 完整 VCALENDAR：展开重复事件必须带上主事件 + 覆盖实例，不能拆开。
    calendar: Calendar | None = None
    # 该资源里的 VEVENT（主事件 + 覆盖实例），用于取 UID 清理作业。
    # 会重复触发的事件(如每周重复的会议)，可能同时存在主事件和覆盖实例，
    # 一般而言，一个 href 对应一个逻辑事件。
    events: list[ICalEvent] = field(default_factory=list)


class NotifyPlugin(NcatBotPlugin):
    caldav_url: str
    caldav_user: str
    caldav_password: str
    davclient: "aio.AsyncDAVClient"
    sync_tokens: dict[str, str]   # {calendar_url: sync_token}
    snapshot: dict[str, str]      # {对象绝对URL: 完整 iCal 文本}
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

    @staticmethod
    def _vevents(cal: Component) -> list[ICalEvent]:
        # icalendar 的 walk() 静态标注是 list[Component]，这里收窄到 Event。
        # 形参用 Component：Calendar 是它的子类，且 walk() 就定义在它上面。
        return [c for c in cal.walk("VEVENT") if isinstance(c, ICalEvent)]

    @classmethod
    def _parse_ics(cls, ics: str) -> tuple[Calendar | None, list[ICalEvent]]:
        parsed = Calendar.from_ical(ics)
        if not isinstance(parsed, Calendar):
            return None, []
        return parsed, cls._vevents(parsed)

    async def get_calendar_events(self) -> list[Change]:
        principal = await self.davclient.get_principal()
        calendars = await self.davclient.get_calendars(principal)

        changes: list[Change] = []
        for calendar in calendars:
            cal_url = str(calendar.url)
            old_token = self.sync_tokens.get(cal_url)

            # 在 calendar 上调，返回的是集合，不是元组。
            # 公开的 get_objects_by_sync_token() 返回 list | Coroutine 联合类型，
            # Pyright 无法判定要不要 await；这里直接用内部那个真正 async 的实现。
            collection = await calendar._async_get_objects_by_sync_token(
                sync_token=old_token,
                load_objects=True,       # 需要完整 iCal 数据
            )
            # 保存新 token（首次调用 old_token=None，会全量返回并给初始 token）
            self.sync_tokens[cal_url] = collection.sync_token

            for obj in collection:
                href = str(obj.url.canonical())

                # 被删除的对象：data 为 None，从快照反查内容
                if obj.data is None:
                    cached_ics = self.snapshot.pop(href, None)
                    cal: Calendar | None = None
                    events: list[ICalEvent] = []
                    if cached_ics:
                        cal, events = self._parse_ics(cached_ics)
                    changes.append(
                        Change(
                            href=href,
                            kind=ChangeKind.DELETED,
                            calendar=cal,
                            events=events,
                        )
                    )
                    continue

                # get_icalendar_instance() 直接给 icalendar.Calendar，
                # 可以 walk("VEVENT") 拿到组件，不必再 to_ical() 序列化一遍。
                # 返回类型声明可空，显式收窄，避免下游 to_ical()/_vevents() 报 Optional。
                cal = obj.get_icalendar_instance()
                if not isinstance(cal, Calendar):
                    continue
                kind = ChangeKind.UPDATED if href in self.snapshot else ChangeKind.ADDED
                # 快照存 iCal 文本，删除时反查用
                self.snapshot[href] = cal.to_ical().decode()
                changes.append(
                    Change(href=href, kind=kind, calendar=cal, events=self._vevents(cal))
                )
        return changes

    # ---------- 通知调度 ----------

    def _cancel_notifications(self, uid: str) -> None:
        """撤掉某个 UID 下所有待发的通知作业。"""
        prefix = f"{NOTIFY_JOB_PREFIX}{uid}:"
        for job in self.scheduler.get_jobs():
            if job.id.startswith(prefix):
                try:
                    self.scheduler.remove_job(job.id)
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
        start = dtstart.dt
        if not isinstance(start, datetime):
            return  # 全天事件暂不提醒
        if start.tzinfo is None:
            start = start.replace(tzinfo=now.tzinfo)
        if start <= now:
            return

        notify_at = start - timedelta(minutes=self.notify_before_minutes)
        if notify_at <= now:
            # 通知点已过（如插件刚启动），跳过，避免补发一堆历史提醒
            return

        uid = str(occ.get("UID") or "")
        rid = occ.get("RECURRENCE-ID")
        anchor = rid.dt if rid is not None else start
        job_id = f"{NOTIFY_JOB_PREFIX}{uid}:{anchor.isoformat()}"

        self.scheduler.add_job(
            self.send_notification,
            trigger=DateTrigger(run_date=notify_at),
            args=[str(occ.get("SUMMARY") or "(无标题)"), start.isoformat()],
            id=job_id,
            replace_existing=True,
        )

    async def notify_events(self, changes: list[Change]) -> None:
        now = datetime.now().astimezone()
        horizon = now + timedelta(days=self.notify_window_days)

        for change in changes:
            uids = {str(e.get("UID")) for e in change.events if e.get("UID")}

            # 删除：撤掉该事件所有待发通知
            if change.kind is ChangeKind.DELETED:
                for uid in uids:
                    self._cancel_notifications(uid)
                continue

            # 新增/更新：先清旧作业，再按最新内容重排
            for uid in uids:
                self._cancel_notifications(uid)

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

    async def send_notification(self, summary: str, start_iso: str) -> None:
        if self.target_id == 0:
            self.logger.error("未配置 target_id，无法发送日程提醒")
            return
        try:
            start = datetime.fromisoformat(start_iso)
            when = f"{start:%Y-%m-%d %H:%M}"
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
