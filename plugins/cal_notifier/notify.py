from dataclasses import dataclass, field
from enum import Enum

from ncatbot.core import registrar
from ncatbot.plugin import NcatBotPlugin
from caldav import aio
from icalendar import Calendar
from icalendar.cal import Component
from icalendar.cal import Event as ICalEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from datetime import datetime
import recurring_ical_events

class ChangeKind(Enum):
    ADDED = "added"
    UPDATED = "updated"
    DELETED = "deleted"


@dataclass
class Change:
    href: str
    kind: ChangeKind
    # 会重复触发的事件(如 每周重复的会议)，可能会有多个 Event 对象，一个是主事件（如开会），一个是重复事件实例（如1月1日的重要会议），
    # 一般而言，一个herf 对应一个事件。
    events: list[ICalEvent] = field(default_factory=list)


class NotifyPlugin(NcatBotPlugin):
    caldav_url: str
    caldav_user: str
    caldav_password: str
    davclient: "aio.AsyncDAVClient"
    sync_tokens: dict[str, str]   # {calendar_url: sync_token}
    snapshot: dict[str, str]      # {对象绝对URL: 完整 iCal 文本}

    async def on_load(self) -> None:
        self.caldav_url = self.config.get("CALDAV_URL", "https://caldav.icloud.com/")
        self.caldav_user = self.config.get("CALDAV_USER", "your_username")
        self.caldav_password = self.config.get("CALDAV_PASSWORD", "your_password")
        self.mariadb_url = self.config.get("MARIADB_URL", "127.0.0.1:3306")
        self.mariadb_user = self.config.get("MARIADB_USER", "user")
        self.mariadb_password = self.config.get("MARIADB_PASSWORD", "password")
        jobstores = {
            'default': SQLAlchemyJobStore(
                url=f"mariadb+mariadbconnector://{self.mariadb_user}:{self.mariadb_password}@{self.mariadb_url}/apscheduler_db?charset=utf8mb4",
            )
        }
        self.davclient = await aio.get_async_davclient(
            url=self.caldav_url,
            username=self.caldav_user,
            password=self.caldav_password,
        )
        self.sync_tokens = {}
        self.snapshot = {}

        self.scheduler = AsyncIOScheduler(
            timezone="Asia/Shanghai",
            jobstores=jobstores,
            job_defaults={
                'coalesce': True,       # 合并错过的运行
                'max_instances': 1,     # 同一任务最大并发实例数
                'misfire_grace_time': 60 # 允许延迟执行的秒数
            })
        if not self.scheduler.get_job("caldav_poll"):
            self.scheduler.add_job(
                self.get_calendar_events,
                trigger=IntervalTrigger(minutes=30),
                id="caldav_poll",
                replace_existing=True
            )
        self.scheduler.start()          # ← 必须启动
        self.logger.info(f"{self.name} 已加载")

    async def on_close(self) -> None:
        if getattr(self, "scheduler", None) is not None:
            self.scheduler.shutdown(wait=False)
        if getattr(self, "davclient", None) is not None:
            await self.davclient.close()
        self.logger.info(f"{self.name} 已卸载")

    @staticmethod
    def _vevents(cal: Component) -> list[ICalEvent]:
        # icalendar 的 walk() 静态标注是 list[Component]，这里收窄到 Event。
        # 形参用 Component：Calendar 是它的子类，且 walk() 就定义在它上面。
        return [c for c in cal.walk("VEVENT") if isinstance(c, ICalEvent)]

    @classmethod
    def _parse_ics(cls, ics: str) -> list[ICalEvent]:
        return cls._vevents(Calendar.from_ical(ics))

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
                    changes.append(
                        Change(
                            href=href,
                            kind=ChangeKind.DELETED,
                            events=self._parse_ics(cached_ics) if cached_ics else [],
                        )
                    )
                    continue

                # get_icalendar_instance() 直接给 icalendar.Calendar，
                # 可以 walk("VEVENT") 拿到组件，不必再 to_ical() 序列化一遍。
                cal = obj.get_icalendar_instance()
                kind = ChangeKind.UPDATED if href in self.snapshot else ChangeKind.ADDED
                # 快照存 iCal 文本，删除时反查用
                self.snapshot[href] = cal.to_ical().decode()
                changes.append(
                    Change(href=href, kind=kind, events=self._vevents(cal))
                )
        return changes

    async def notify_events(self, changes: list[Change]) -> None:
        now = datetime.now().astimezone()
        for change in changes:
            for event in change.events:
                dtstart = event.decoded("DTSTART")
                if dtstart is None:
                    continue
                # 这里可以根据事件的时间、标题等信息，发送通知给用户
                self.logger.info(f"即将发生的事件: {event.get('UID')}，开始时间: {dtstart}")
                if dtstart >= now:
                    pass