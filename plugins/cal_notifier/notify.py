from ncatbot.core import registrar
from ncatbot.plugin import NcatBotPlugin
from caldav import aio
from ical.calendar_stream import CalendarStream
from ical.event import Event
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

class NotifyPlugin(NcatBotPlugin):
    caldav_url: str
    caldav_user: str
    caldav_password: str
    davclient: "aio.AsyncDAVClient"
    async def on_load(self) -> None:
        self.caldav_url = self.config.get("CALDAV_URL", "https://caldav.icloud.com/")
        self.caldav_user = self.config.get("CALDAV_USER", "your_username")
        self.caldav_password = self.config.get("CALDAV_PASSWORD", "your_password")
        self.davclient = await aio.get_async_davclient(
            url=self.caldav_url,
            username=self.caldav_user,
            password=self.caldav_password,
        )
        self.logger.info(f"{self.name} 已加载")
        self.scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
        self.scheduler.add_job(self.get_calendar_events, trigger=IntervalTrigger(minutes=30),)

    async def on_close(self) -> None:
        if getattr(self, "davclient", None) is not None:
            await self.davclient.close()
        self.logger.info(f"{self.name} 已卸载")
    async def get_calendar_events(self) -> list[Event]:
        # 用 AsyncDAVClient 上的高层 async API，避开 dual-mode 的联合返回类型
        principal = await self.davclient.get_principal()
        calendars = await self.davclient.get_calendars(principal)
        events: list[Event] = []
        for calendar in calendars:
            # 获取日历中的事件
            calendar_events = await self.davclient.search_calendar(calendar, event=True)
            for event in calendar_events:
                # caldav 底层是 icalendar；vobject 已不再是其依赖（未安装），
                # 所以用 get_icalendar_instance()，序列化方法为 to_ical()
                ics_string = event.get_icalendar_instance().to_ical().decode()
                # to_ical() 输出的已经是完整的 VCALENDAR，不能再手动包裹一层
                # 工厂方法是 from_ics()，不是 parse()
                stream = CalendarStream.from_ics(ics_string)
                for ical_calendar in stream.calendars:
                    events.extend(ical_calendar.events)
        return events