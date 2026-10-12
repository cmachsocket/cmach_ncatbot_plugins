from ncatbot.core import registrar
from ncatbot.plugin import NcatBotPlugin
from caldav import aio
from ical.calendar_stream import CalendarStream
from ical.event import Event
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore



class NotifyPlugin(NcatBotPlugin):
    caldav_url: str
    caldav_user: str
    caldav_password: str
    davclient: "aio.AsyncDAVClient"
    sync_tokens: dict[str, str]   # {calendar_url: sync_token}

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

    async def get_calendar_events(self) -> list[Event]:
        principal = await self.davclient.get_principal()
        calendars = await self.davclient.get_calendars(principal)

        events: list[Event] = []
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
                # 被删除的对象：data 为 None，直接跳过
                if obj.data is None:
                    continue

                ics_string = obj.get_icalendar_instance().to_ical().decode()
                stream = CalendarStream.from_ics(ics_string)
                for ical_calendar in stream.calendars:
                    events.extend(ical_calendar.events)
        return events
    async def notify_events(self, events: list[Event]) -> None:
        for event in events:
            # 这里可以根据事件的时间、标题等信息，发送通知给用户
            self.logger.info(f"即将发生的事件: {event.uid}，开始时间: {event.start}")
            # 例如，可以调用 bot 的消息发送接口，发送到指定的频道或用户
            # await self.bot.send_message(channel_id, f"即将发生的事件: {event.summary}，开始时间: {event.start_time}")
            