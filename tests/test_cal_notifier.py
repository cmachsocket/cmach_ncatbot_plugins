"""CalDAV 日程提醒插件测试（不访问网络）。"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from apscheduler.jobstores.base import JobLookupError
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger
from icalendar import Calendar
from icalendar.cal import Event as ICalEvent
from icalendar.cal import Todo as ICalTodo

from plugins.cal_notifier import notify as notify_module

TZ = ZoneInfo("Asia/Shanghai")


class _FakeScheduler:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.jobs: dict[str, dict[str, Any]] = {}
        self.started = False
        self.shutdown_wait: bool | None = None

    def add_job(
        self,
        func: Any,
        trigger: Any = None,
        args: list[Any] | None = None,
        id: str | None = None,
        replace_existing: bool = False,
        **kwargs: Any,
    ) -> None:
        assert id is not None
        self.jobs[id] = {
            "func": func,
            "trigger": trigger,
            "args": args,
            "replace_existing": replace_existing,
            **kwargs,
        }

    def remove_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise JobLookupError(job_id)
        del self.jobs[job_id]

    def start(self) -> None:
        self.started = True

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_wait = wait


class _QQ:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []
        self.error: Exception | None = None

    async def post_private_msg(self, target_id: int, *, text: str) -> None:
        if self.error is not None:
            raise self.error
        self.messages.append((target_id, text))


class _Api:
    def __init__(self) -> None:
        self.qq = _QQ()


class _DavClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Event:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply(self, *, text: str) -> None:
        self.replies.append(text)


def _plugin(**overrides: Any) -> notify_module.NotifyPlugin:
    plugin = object.__new__(notify_module.NotifyPlugin)
    plugin.name = "cal-test"
    plugin.tz = TZ
    plugin.notify_before_minutes = 10
    plugin.notify_window_days = 30
    plugin.all_day_notify_hour = 23
    plugin.all_day_notify_minute = 30
    plugin.todo_notify_times = ()
    plugin.target_id = 100
    plugin.scheduler = _FakeScheduler()  # type: ignore[assignment]
    plugin._jobs_by_owner = {}
    plugin._todo_snapshot = {}
    plugin._last_full_sync = datetime.now(TZ)
    plugin._sync_lock = asyncio.Lock()
    plugin.snapshot = {}
    plugin.sync_tokens = {}
    plugin.davclient = _DavClient()  # type: ignore[assignment]
    plugin.api = _Api()  # type: ignore[assignment]
    for key, value in overrides.items():
        setattr(plugin, key, value)
    return plugin


def _override(obj: Any, name: str, value: Any) -> None:
    vars(obj)[name] = value


def _ics(body: str) -> Calendar:
    return Calendar.from_ical(
        "BEGIN:VCALENDAR\nVERSION:2.0\n" + body + "\nEND:VCALENDAR"
    )


def _event(body: str) -> ICalEvent:
    event = _ics("BEGIN:VEVENT\n" + body + "\nEND:VEVENT").walk("VEVENT")[0]
    assert isinstance(event, ICalEvent)
    return event


def _todo(body: str) -> ICalTodo:
    todo = _ics("BEGIN:VTODO\n" + body + "\nEND:VTODO").walk("VTODO")[0]
    assert isinstance(todo, ICalTodo)
    return todo


# ---------- 配置 ----------


def test_resolve_timezone_returns_requested_zone() -> None:
    plugin = _plugin()

    assert plugin._resolve_timezone("UTC").key == "UTC"


def test_resolve_timezone_falls_back_for_invalid_name(caplog) -> None:
    plugin = _plugin()

    with caplog.at_level("WARNING"):
        zone = plugin._resolve_timezone("Mars/Olympus")

    assert zone.key == notify_module.DEFAULT_TIMEZONE
    assert "不可用" in caplog.text


def test_build_jobstores_only_uses_memory_store() -> None:
    stores = _plugin()._build_jobstores()

    assert list(stores) == ["default"]
    assert isinstance(stores["default"], MemoryJobStore)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ()),
        ("", ()),
        ([], ()),
        (["09:00", "20:30"], ((9, 0), (20, 30))),
        ("09:00, 20:30", ((9, 0), (20, 30))),
        ("09:00 20:30", ((9, 0), (20, 30))),
        ("09:00,09:00", ((9, 0),)),
        (["09:00", "bad", "24:00", "12:60"], ((9, 0),)),
        (123, ()),
    ],
)
def test_parse_notify_times(value: Any, expected: tuple[tuple[int, int], ...]) -> None:
    assert _plugin()._parse_notify_times(value) == expected


def test_parse_notify_times_warns_on_bad_entries(caplog) -> None:
    with caplog.at_level("WARNING"):
        times = _plugin()._parse_notify_times(["nope", "25:00"])

    assert times == ()
    assert "忽略无效的提醒时刻" in caplog.text
    assert "忽略越界的提醒时刻" in caplog.text


def test_on_load_reads_config_and_starts_scheduler(monkeypatch, caplog) -> None:
    scheduler = _FakeScheduler()
    dav = _DavClient()
    created: dict[str, Any] = {}

    def fake_scheduler(**kwargs: Any) -> _FakeScheduler:
        created.update(kwargs)
        scheduler.kwargs = kwargs
        return scheduler

    async def fake_davclient(**kwargs: Any) -> _DavClient:
        created["dav"] = kwargs
        return dav

    monkeypatch.setattr(notify_module, "AsyncIOScheduler", fake_scheduler)
    monkeypatch.setattr(notify_module.aio, "get_async_davclient", fake_davclient)

    plugin = object.__new__(notify_module.NotifyPlugin)
    plugin.name = "cal-test"
    plugin.config = {
        "CALDAV_URL": "https://dav.example/",
        "CALDAV_USER": "alice",
        "CALDAV_PASSWORD": "secret",
        "target_id": "42",
        "notify_before_minutes": "-5",
        "notify_window_days": "0",
        "all_day_notify_hour": "22",
        "all_day_notify_minute": "15",
        "todo_notify_times": ["09:00", "20:00"],
        "timezone": "UTC",
    }

    with caplog.at_level("INFO"):
        asyncio.run(plugin.on_load())

    assert plugin.target_id == 42
    assert plugin.notify_before_minutes == 0
    assert plugin.notify_window_days == 1
    assert plugin.all_day_notify_hour == 22
    assert plugin.all_day_notify_minute == 15
    assert plugin.todo_notify_times == ((9, 0), (20, 0))
    assert plugin.tz.key == "UTC"
    assert created["dav"] == {
        "url": "https://dav.example/",
        "username": "alice",
        "password": "secret",
    }
    assert scheduler.started is True
    assert "caldav_poll" in scheduler.jobs
    assert "caldav_todo_daily:09:00" in scheduler.jobs
    assert "caldav_todo_daily:20:00" in scheduler.jobs
    # 每个时刻只有一个汇总作业，和待办数量无关
    digest = scheduler.jobs["caldav_todo_daily:09:00"]
    assert isinstance(digest["trigger"], CronTrigger)
    assert digest["args"] == ["09:00"]
    assert "已加载" in caplog.text

    asyncio.run(plugin.on_close())

    assert scheduler.shutdown_wait is False
    assert dav.closed is True


def test_on_load_warns_when_target_id_is_missing(monkeypatch, caplog) -> None:
    async def fake_davclient(**kwargs: Any) -> _DavClient:
        return _DavClient()

    monkeypatch.setattr(notify_module, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(notify_module.aio, "get_async_davclient", fake_davclient)

    plugin = object.__new__(notify_module.NotifyPlugin)
    plugin.name = "cal-test"
    plugin.config = {}

    with caplog.at_level("ERROR"):
        asyncio.run(plugin.on_load())

    assert "未配置 target_id" in caplog.text


def test_on_close_is_safe_without_resources(caplog) -> None:
    plugin = object.__new__(notify_module.NotifyPlugin)
    plugin.name = "cal-test"

    with caplog.at_level("INFO"):
        asyncio.run(plugin.on_close())

    assert "已卸载" in caplog.text


# ---------- 作业调度 ----------


def test_schedule_timed_occurrence_sets_notify_time() -> None:
    plugin = _plugin()
    event = _event("UID:uid-1\nSUMMARY:Meeting\nDTSTART:20261101T100000")
    now = datetime(2026, 10, 26, 12, 0, tzinfo=TZ)

    plugin._schedule_occurrence(event, now, notify_module.COMPONENT_EVENT)

    job_id = "caldav_notify:VEVENT:uid-1:2026-11-01T10:00:00"
    job = plugin.scheduler.jobs[job_id]  # type: ignore[index]
    assert job["args"] == [
        "Meeting",
        "开始时间：2026-11-01 10:00",
        "VEVENT:uid-1",
        job_id,
    ]
    assert job["trigger"].run_date == datetime(2026, 11, 1, 9, 50, tzinfo=TZ)
    assert plugin._jobs_by_owner["VEVENT:uid-1"] == {job_id}


def test_schedule_all_day_occurrence_notifies_previous_evening() -> None:
    plugin = _plugin()
    event = _event("UID:all-day\nSUMMARY:Holiday\nDTSTART;VALUE=DATE:20261102")
    now = datetime(2026, 10, 26, 12, 0, tzinfo=TZ)

    plugin._schedule_occurrence(event, now, notify_module.COMPONENT_EVENT)

    job_id = "caldav_notify:VEVENT:all-day:2026-11-02"
    job = plugin.scheduler.jobs[job_id]  # type: ignore[index]
    assert job["args"] == [
        "Holiday",
        "开始时间：2026-11-02 全天",
        "VEVENT:all-day",
        job_id,
    ]
    assert job["trigger"].run_date == datetime(2026, 11, 1, 23, 30, tzinfo=TZ)


def test_schedule_uses_recurrence_id_as_anchor() -> None:
    plugin = _plugin()
    event = _event(
        "UID:rec\nSUMMARY:Weekly\nDTSTART:20261101T100000\n"
        "RECURRENCE-ID:20261108T100000"
    )
    now = datetime(2026, 10, 26, 12, 0, tzinfo=TZ)

    plugin._schedule_occurrence(event, now, notify_module.COMPONENT_EVENT)

    assert plugin._jobs_by_owner["VEVENT:rec"] == {
        "caldav_notify:VEVENT:rec:2026-11-08T10:00:00"
    }


def test_schedule_skips_cancelled_occurrence() -> None:
    plugin = _plugin()
    event = _event(
        "UID:uid-1\nSUMMARY:Meeting\nSTATUS:CANCELLED\nDTSTART:20261101T100000"
    )

    plugin._schedule_occurrence(
        event, datetime(2026, 10, 26, tzinfo=TZ), notify_module.COMPONENT_EVENT
    )

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "body",
    [
        "UID:no-start\nSUMMARY:No start",
        "UID:past\nSUMMARY:Past\nDTSTART:20261001T100000",
        "UID:too-late\nSUMMARY:Too late\nDTSTART:20261026T120500",
    ],
)
def test_schedule_skips_invalid_or_past_occurrences(body: str) -> None:
    plugin = _plugin()
    now = datetime(2026, 10, 26, 12, 0, tzinfo=TZ)

    plugin._schedule_occurrence(_event(body), now, notify_module.COMPONENT_EVENT)

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def test_schedule_skips_past_all_day_occurrence() -> None:
    plugin = _plugin()
    event = _event("UID:old\nSUMMARY:Old\nDTSTART;VALUE=DATE:20261025")

    plugin._schedule_occurrence(
        event, datetime(2026, 10, 26, 12, 0, tzinfo=TZ), notify_module.COMPONENT_EVENT
    )

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def test_schedule_ignores_unsupported_start_value() -> None:
    plugin = _plugin()
    event = _event("UID:clock\nSUMMARY:Clock\nDTSTART:20261101T100000")
    event["DTSTART"] = SimpleNamespace(dt="not a date")

    plugin._schedule_occurrence(
        event, datetime(2026, 10, 26, 12, 0, tzinfo=TZ), notify_module.COMPONENT_EVENT
    )

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


# ---------- 待办调度 ----------


def test_schedule_todo_uses_due_as_anchor() -> None:
    plugin = _plugin()
    todo = _todo(
        "UID:todo-1\nSUMMARY:Submit report\n"
        "DTSTART:20261101T090000\nDUE:20261101T180000"
    )
    now = datetime(2026, 10, 26, 12, 0, tzinfo=TZ)

    plugin._schedule_occurrence(todo, now, notify_module.COMPONENT_TODO)

    job_id = "caldav_notify:VTODO:todo-1:2026-11-01T18:00:00"
    job = plugin.scheduler.jobs[job_id]  # type: ignore[index]
    assert job["args"] == [
        "Submit report",
        "截止时间：2026-11-01 18:00",
        "VTODO:todo-1",
        job_id,
    ]
    assert job["trigger"].run_date == datetime(2026, 11, 1, 17, 50, tzinfo=TZ)


def test_schedule_todo_without_due_falls_back_to_dtstart() -> None:
    plugin = _plugin()
    todo = _todo("UID:todo-2\nSUMMARY:Start only\nDTSTART:20261101T090000")

    plugin._schedule_occurrence(
        todo, datetime(2026, 10, 26, 12, 0, tzinfo=TZ), notify_module.COMPONENT_TODO
    )

    job_id = "caldav_notify:VTODO:todo-2:2026-11-01T09:00:00"
    job = plugin.scheduler.jobs[job_id]  # type: ignore[index]
    assert job["trigger"].run_date == datetime(2026, 11, 1, 8, 50, tzinfo=TZ)


def test_schedule_all_day_todo_notifies_previous_evening() -> None:
    plugin = _plugin()
    todo = _todo("UID:todo-3\nSUMMARY:Pay bill\nDUE;VALUE=DATE:20261102")

    plugin._schedule_occurrence(
        todo, datetime(2026, 10, 26, 12, 0, tzinfo=TZ), notify_module.COMPONENT_TODO
    )

    job_id = "caldav_notify:VTODO:todo-3:2026-11-02"
    job = plugin.scheduler.jobs[job_id]  # type: ignore[index]
    assert job["args"] == ["Pay bill", "截止时间：2026-11-02", "VTODO:todo-3", job_id]
    assert job["trigger"].run_date == datetime(2026, 11, 1, 23, 30, tzinfo=TZ)


@pytest.mark.parametrize(
    "body",
    [
        "UID:done\nSUMMARY:Done\nDUE:20261101T180000\nSTATUS:COMPLETED",
        "UID:done\nSUMMARY:Done\nDUE:20261101T180000\nPERCENT-COMPLETE:100",
    ],
)
def test_schedule_skips_completed_todo(body: str) -> None:
    plugin = _plugin()

    plugin._schedule_occurrence(
        _todo(body),
        datetime(2026, 10, 26, 12, 0, tzinfo=TZ),
        notify_module.COMPONENT_TODO,
    )

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def test_schedule_todo_without_any_time_is_skipped() -> None:
    plugin = _plugin()

    plugin._schedule_occurrence(
        _todo("UID:todo-4\nSUMMARY:Nothing"),
        datetime(2026, 10, 26, 12, 0, tzinfo=TZ),
        notify_module.COMPONENT_TODO,
    )

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "body",
    [
        "UID:utc-event\nSUMMARY:UTC\nDTSTART:20261104T174023Z",
        "UID:utc-todo\nSUMMARY:UTC\nDUE:20261104T174023Z",
    ],
)
def test_schedule_converts_utc_time_to_local(body: str) -> None:
    plugin = _plugin()
    component = (
        notify_module.COMPONENT_TODO
        if "utc-todo" in body
        else notify_module.COMPONENT_EVENT
    )
    occ = _todo(body) if component == notify_module.COMPONENT_TODO else _event(body)

    plugin._schedule_occurrence(
        occ, datetime(2026, 10, 26, 12, 0, tzinfo=TZ), component
    )

    (job,) = plugin.scheduler.jobs.values()  # type: ignore[union-attr]
    assert job["args"][1].endswith("2026-11-05 01:40")
    assert job["trigger"].run_date == datetime(2026, 11, 5, 1, 30, 23, tzinfo=TZ)


def test_notify_events_skips_undated_todo_before_expanding() -> None:
    plugin = _plugin()
    calendar = Calendar()
    todo = ICalTodo()
    todo.add("UID", "todo-nodate")
    todo.add("SUMMARY", "清单项")
    calendar.add_component(todo)

    asyncio.run(
        plugin.notify_events(
            [
                notify_module.Change(
                    uid="todo-nodate",
                    href="todo-nodate.ics",
                    kind=notify_module.ChangeKind.ADDED,
                    calendar=calendar,
                    component=notify_module.COMPONENT_TODO,
                )
            ]
        )
    )

    # 没有 DTSTART/DUE 就没有可锚定的时间，不能靠合成出来的 2038-01-01 排作业
    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def _undated_todo_calendar(uid: str = "todo-nodate", **props: Any) -> Calendar:
    calendar = Calendar()
    todo = ICalTodo()
    todo.add("UID", uid)
    todo.add("SUMMARY", "清单项")
    for key, value in props.items():
        todo.add(key, value)
    calendar.add_component(todo)
    return calendar


def _ics_for(component: Any) -> Calendar:
    calendar = Calendar()
    calendar.add_component(component)
    return calendar


def _undated_change(calendar: Calendar, uid: str = "todo-nodate") -> Any:
    return notify_module.Change(
        uid=uid,
        href=f"{uid}.ics",
        kind=notify_module.ChangeKind.ADDED,
        calendar=calendar,
        component=notify_module.COMPONENT_TODO,
    )


def test_notify_events_records_undated_todo_for_digest() -> None:
    plugin = _plugin()

    asyncio.run(plugin.notify_events([_undated_change(_undated_todo_calendar())]))

    assert plugin._todo_snapshot == {"todo-nodate": notify_module.TodoEntry("清单项")}
    # 无日期待办不排一次性作业，汇总由每个时刻一个的 cron 统一负责
    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def test_notify_events_drops_completed_undated_todo_from_digest() -> None:
    plugin = _plugin()

    asyncio.run(
        plugin.notify_events(
            [_undated_change(_undated_todo_calendar(STATUS="COMPLETED"))]
        )
    )

    assert plugin._todo_snapshot == {}


def test_notify_events_drops_cancelled_undated_todo_from_digest() -> None:
    plugin = _plugin()

    asyncio.run(
        plugin.notify_events(
            [_undated_change(_undated_todo_calendar(STATUS="CANCELLED"))]
        )
    )

    assert plugin._todo_snapshot == {}


def test_notify_events_drops_deleted_todo_from_digest() -> None:
    plugin = _plugin()
    plugin._todo_snapshot = {"todo-nodate": notify_module.TodoEntry("清单项")}

    asyncio.run(
        plugin.notify_events(
            [
                notify_module.Change(
                    uid="todo-nodate",
                    href="todo-nodate.ics",
                    kind=notify_module.ChangeKind.DELETED,
                    component=notify_module.COMPONENT_TODO,
                )
            ]
        )
    )

    assert plugin._todo_snapshot == {}


def test_notify_events_records_dated_todo_for_digest() -> None:
    plugin = _plugin()
    due = (datetime.now(TZ) + timedelta(days=3)).replace(microsecond=0, tzinfo=None)
    todo = _todo(f"UID:todo-due\nSUMMARY:Submit report\nDUE:{due:%Y%m%dT%H%M%S}")

    asyncio.run(plugin.notify_events([_undated_change(_ics_for(todo), uid="todo-due")]))

    entry = plugin._todo_snapshot["todo-due"]
    assert entry.summary == "Submit report"
    assert entry.all_day is False
    assert entry.label == "截止时间"
    assert entry.due == due.replace(tzinfo=TZ)


def test_recurring_todo_digest_uses_nearest_occurrence() -> None:
    plugin = _plugin(notify_window_days=60)
    start = (datetime.now(TZ) + timedelta(days=2)).replace(microsecond=0)
    calendar = Calendar()
    todo = ICalTodo()
    todo.add("UID", "todo-rec")
    todo.add("SUMMARY", "周报")
    todo.add("DUE", start)
    todo.add("RRULE", {"FREQ": "WEEKLY", "COUNT": "3"})
    calendar.add_component(todo)

    asyncio.run(
        plugin.notify_events(
            [
                notify_module.Change(
                    uid="todo-rec",
                    href="todo-rec.ics",
                    kind=notify_module.ChangeKind.ADDED,
                    calendar=calendar,
                    component=notify_module.COMPONENT_TODO,
                )
            ]
        )
    )

    # 三次实例各有一个一次性提醒，汇总里只记最近一次
    assert len(plugin.scheduler.jobs) == 3  # type: ignore[attr-defined]
    entry = plugin._todo_snapshot["todo-rec"]
    assert entry.due == start
    assert entry.label == "截止时间"


def test_send_todo_digest_posts_one_message() -> None:
    plugin = _plugin()
    now = datetime.now(TZ).replace(microsecond=0)
    plugin._todo_snapshot = {
        "b": notify_module.TodoEntry(
            "交周报", due=now + timedelta(days=2), all_day=False, label="截止时间"
        ),
        "a": notify_module.TodoEntry("买菜"),
        "c": notify_module.TodoEntry(
            "体检", due=now + timedelta(days=4), all_day=True, label="截止时间"
        ),
    }

    asyncio.run(plugin.send_todo_digest("09:00"))

    ((target, text),) = plugin.api.qq.messages  # type: ignore[union-attr]
    assert target == 100
    lines = text.splitlines()
    assert lines[0] == "待办提醒（09:00）"
    # 有日期的按时间排，无日期的排最后
    assert lines[1] == f"· 交周报（截止时间 {now + timedelta(days=2):%Y-%m-%d %H:%M}）"
    assert lines[2] == f"· 体检（截止时间 {now + timedelta(days=4):%Y-%m-%d}）"
    assert lines[3] == "· 买菜（未设置日期）"


def test_send_todo_digest_prunes_expired_and_skips_empty() -> None:
    plugin = _plugin()
    now = datetime.now(TZ).replace(microsecond=0)
    plugin._todo_snapshot = {
        "old": notify_module.TodoEntry(
            "过期", due=now - timedelta(hours=1), label="截止时间"
        )
    }

    asyncio.run(plugin.send_todo_digest("09:00"))

    assert plugin._todo_snapshot == {}
    assert plugin.api.qq.messages == []  # type: ignore[union-attr]


def test_send_todo_digest_without_target_logs_and_skips(caplog) -> None:
    plugin = _plugin(target_id=0)
    plugin._todo_snapshot = {"a": notify_module.TodoEntry("买菜")}

    with caplog.at_level("ERROR"):
        asyncio.run(plugin.send_todo_digest("09:00"))

    assert plugin.api.qq.messages == []  # type: ignore[union-attr]
    assert "未配置 target_id" in caplog.text


def test_cancel_notifications_removes_all_jobs_and_swallows_missing_onces() -> None:
    plugin = _plugin()
    plugin.scheduler.jobs = {"a": {}, "b": {}}  # type: ignore[union-attr]
    plugin._jobs_by_owner = {"VEVENT:uid": {"a", "missing", "b"}}

    plugin._cancel_notifications("VEVENT:uid")

    assert plugin.scheduler.jobs == {}  # type: ignore[union-attr]
    assert "VEVENT:uid" not in plugin._jobs_by_owner


def test_cancel_notifications_keeps_other_component_with_same_uid() -> None:
    plugin = _plugin()
    plugin.scheduler.jobs = {"a": {}, "b": {}}  # type: ignore[union-attr]
    plugin._jobs_by_owner = {"VEVENT:uid": {"a"}, "VTODO:uid": {"b"}}

    plugin._cancel_notifications("VTODO:uid")

    assert plugin.scheduler.jobs == {"a": {}}  # type: ignore[union-attr]
    assert plugin._jobs_by_owner == {"VEVENT:uid": {"a"}}


def test_forget_job_discards_from_index() -> None:
    plugin = _plugin()
    plugin._jobs_by_owner = {"VEVENT:uid": {"a", "b"}}

    plugin._forget_job("VEVENT:uid", "a")
    plugin._forget_job("unknown", "x")

    assert plugin._jobs_by_owner == {"VEVENT:uid": {"b"}}


# ---------- 变更处理 ----------


def test_notify_events_expands_recurring_calendar_and_schedules(monkeypatch) -> None:
    plugin = _plugin(notify_window_days=60)
    start = datetime.now(TZ).replace(microsecond=0) + timedelta(days=1)
    calendar = Calendar()
    event = ICalEvent()
    event.add("UID", "rec-1")
    event.add("SUMMARY", "Weekly")
    event.add("DTSTART", start)
    event.add("RRULE", {"FREQ": "WEEKLY", "COUNT": "3"})
    calendar.add_component(event)

    scheduled: list[ICalEvent] = []
    _override(
        plugin,
        "_schedule_occurrence",
        lambda occ, now, component, **kwargs: scheduled.append(occ),
    )

    asyncio.run(
        plugin.notify_events(
            [
                notify_module.Change(
                    uid="rec-1",
                    href="rec-1.ics",
                    kind=notify_module.ChangeKind.UPDATED,
                    calendar=calendar,
                )
            ]
        )
    )

    assert len(scheduled) == 3
    assert all(isinstance(occ, ICalEvent) for occ in scheduled)


def test_notify_events_handles_deleted_and_empty_changes() -> None:
    plugin = _plugin()
    cancelled: list[str] = []
    _override(
        plugin,
        "_schedule_occurrence",
        lambda occ, now, component: pytest.fail("不应调度"),
    )

    def fake_cancel(owner: str) -> None:
        cancelled.append(owner)

    _override(plugin, "_cancel_notifications", fake_cancel)

    asyncio.run(
        plugin.notify_events(
            [
                notify_module.Change(
                    uid="gone", href="gone.ics", kind=notify_module.ChangeKind.DELETED
                ),
                notify_module.Change(
                    uid="no-calendar",
                    href="x.ics",
                    kind=notify_module.ChangeKind.UPDATED,
                    calendar=None,
                ),
            ]
        )
    )

    assert cancelled == ["VEVENT:gone", "VEVENT:no-calendar"]


def test_notify_events_expands_todo_series(monkeypatch) -> None:
    plugin = _plugin(notify_window_days=60)
    due = datetime.now(TZ).replace(microsecond=0) + timedelta(days=1)
    calendar = Calendar()
    todo = ICalTodo()
    todo.add("UID", "todo-rec")
    todo.add("SUMMARY", "Water plants")
    todo.add("DUE", due)
    todo.add("RRULE", {"FREQ": "WEEKLY", "COUNT": "2"})
    calendar.add_component(todo)

    scheduled: list[tuple[Any, str]] = []
    _override(
        plugin,
        "_schedule_occurrence",
        lambda occ, now, component, **kwargs: scheduled.append((occ, component)),
    )

    asyncio.run(
        plugin.notify_events(
            [
                notify_module.Change(
                    uid="todo-rec",
                    href="todo-rec.ics",
                    kind=notify_module.ChangeKind.ADDED,
                    calendar=calendar,
                    component=notify_module.COMPONENT_TODO,
                )
            ]
        )
    )

    assert len(scheduled) == 2
    assert all(isinstance(occ, ICalTodo) for occ, _ in scheduled)
    assert all(component == notify_module.COMPONENT_TODO for _, component in scheduled)


def test_notify_events_does_not_cancel_event_jobs_for_same_uid_todo() -> None:
    plugin = _plugin()
    event_job = "caldav_notify:VEVENT:shared:2026-11-01T10:00:00"
    plugin.scheduler.jobs = {event_job: {}}  # type: ignore[union-attr]
    plugin._jobs_by_owner = {"VEVENT:shared": {event_job}}

    asyncio.run(
        plugin.notify_events(
            [
                notify_module.Change(
                    uid="shared",
                    href="todo.ics",
                    kind=notify_module.ChangeKind.ADDED,
                    calendar=Calendar(),
                    component=notify_module.COMPONENT_TODO,
                )
            ]
        )
    )

    # 待办与日程撞 UID 时不能把日程的作业一起撤掉
    assert plugin.scheduler.jobs == {event_job: {}}  # type: ignore[union-attr]
    assert plugin._jobs_by_owner == {"VEVENT:shared": {event_job}}


def test_notify_events_logs_expansion_failure(monkeypatch, caplog) -> None:
    plugin = _plugin()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad rrule")

    monkeypatch.setattr(notify_module.recurring_ical_events, "of", broken)

    with caplog.at_level("WARNING"):
        asyncio.run(
            plugin.notify_events(
                [
                    notify_module.Change(
                        uid="u",
                        href="u.ics",
                        kind=notify_module.ChangeKind.UPDATED,
                        calendar=Calendar(),
                    )
                ]
            )
        )

    assert "展开事件失败" in caplog.text


# ---------- CalDAV 同步 ----------


class _Url:
    def __init__(self, value: str) -> None:
        self.value = value

    def canonical(self) -> str:
        return self.value


class _RemoteObject:
    def __init__(
        self,
        href: str,
        *,
        data: bytes | None = b"data",
        calendar: Any = None,
    ) -> None:
        self.url = _Url(href)
        self.data = data
        self._calendar = calendar

    def get_icalendar_instance(self) -> Any:
        return self._calendar


class _Collection(list):
    def __init__(self, objects: list[_RemoteObject], token: str) -> None:
        super().__init__(objects)
        self.sync_token = token


class _Calendar:
    def __init__(self, url: str, collection: _Collection | None) -> None:
        self.url = url
        self._collection = collection
        self.received: tuple[str | None, bool] | None = None

    async def _async_get_objects_by_sync_token(
        self, *, sync_token: str | None, load_objects: bool
    ) -> _Collection:
        self.received = (sync_token, load_objects)
        assert self._collection is not None
        return self._collection


class _Dav:
    def __init__(self, calendars: list[Any]) -> None:
        self._calendars = calendars

    async def get_principal(self) -> str:
        return "principal"

    async def get_calendars(self, principal: str) -> list[Any]:
        assert principal == "principal"
        return self._calendars


def test_get_calendar_events_full_sync_reports_added_and_deleted() -> None:
    calendar_data = _ics(
        "BEGIN:VEVENT\nUID:uid-1\nSUMMARY:One\nDTSTART:20261101T100000\nEND:VEVENT"
    )
    added = _RemoteObject("https://dav/cal/1/a.ics", calendar=calendar_data)
    deleted = _RemoteObject("https://dav/cal/1/b.ics", data=None)
    collection = _Collection([added, deleted], "token-1")
    calendar = _Calendar("https://dav/cal/1/", collection)
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]
    plugin.snapshot = {
        "https://dav/cal/1/b.ics": ("uid-b", notify_module.COMPONENT_EVENT),
        "https://dav/cal/1/stale.ics": ("uid-stale", notify_module.COMPONENT_EVENT),
    }

    changes = asyncio.run(plugin.get_calendar_events())

    assert calendar.received == (None, True)
    assert plugin.sync_tokens == {"https://dav/cal/1/": "token-1"}
    assert plugin.snapshot == {
        "https://dav/cal/1/a.ics": ("uid-1", notify_module.COMPONENT_EVENT)
    }
    kinds = {
        (change.href, change.kind, change.uid, change.component) for change in changes
    }
    assert kinds == {
        (
            "https://dav/cal/1/a.ics",
            notify_module.ChangeKind.ADDED,
            "uid-1",
            notify_module.COMPONENT_EVENT,
        ),
        (
            "https://dav/cal/1/b.ics",
            notify_module.ChangeKind.DELETED,
            "uid-b",
            notify_module.COMPONENT_EVENT,
        ),
        (
            "https://dav/cal/1/stale.ics",
            notify_module.ChangeKind.DELETED,
            "uid-stale",
            notify_module.COMPONENT_EVENT,
        ),
    }


def test_get_calendar_events_marks_todo_component() -> None:
    calendar_data = _ics(
        "BEGIN:VTODO\nUID:todo-1\nSUMMARY:Buy milk\nDUE:20261105T180000\nEND:VTODO"
    )
    obj = _RemoteObject("https://dav/cal/1/t.ics", calendar=calendar_data)
    calendar = _Calendar("https://dav/cal/1/", _Collection([obj], "token-1"))
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]

    changes = asyncio.run(plugin.get_calendar_events())

    assert len(changes) == 1
    assert changes[0].component == notify_module.COMPONENT_TODO
    assert changes[0].uid == "todo-1"
    assert plugin.snapshot == {
        "https://dav/cal/1/t.ics": ("todo-1", notify_module.COMPONENT_TODO)
    }


def test_get_calendar_events_skips_unsupported_component() -> None:
    calendar_data = _ics(
        "BEGIN:VJOURNAL\nUID:j-1\nSUMMARY:Diary\nDTSTART:20261105T180000\nEND:VJOURNAL"
    )
    obj = _RemoteObject("https://dav/cal/1/j.ics", calendar=calendar_data)
    calendar = _Calendar("https://dav/cal/1/", _Collection([obj], "token-1"))
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]

    changes = asyncio.run(plugin.get_calendar_events())

    assert changes == []
    assert plugin.snapshot == {}


def test_get_calendar_events_incremental_marks_existing_as_updated() -> None:
    calendar_data = _ics(
        "BEGIN:VEVENT\nUID:uid-1\nSUMMARY:One\nDTSTART:20261101T100000\nEND:VEVENT"
    )
    obj = _RemoteObject("https://dav/cal/1/a.ics", calendar=calendar_data)
    calendar = _Calendar("https://dav/cal/1/", _Collection([obj], "token-2"))
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]
    plugin.sync_tokens = {"https://dav/cal/1/": "token-1"}
    plugin.snapshot = {
        "https://dav/cal/1/a.ics": ("uid-1", notify_module.COMPONENT_EVENT)
    }

    changes = asyncio.run(plugin.get_calendar_events())

    assert calendar.received == ("token-1", True)
    assert len(changes) == 1
    assert changes[0].kind is notify_module.ChangeKind.UPDATED


def test_get_calendar_events_skips_calendar_without_private_api(caplog) -> None:
    calendar = SimpleNamespace(url="https://dav/cal/1/")
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]

    with caplog.at_level("ERROR"):
        changes = asyncio.run(plugin.get_calendar_events())

    assert changes == []
    assert "不存在_async_get_objects_by_sync_token" in caplog.text


def test_get_calendar_events_skips_non_icalendar_payload() -> None:
    obj = _RemoteObject("https://dav/cal/1/a.ics", calendar="not a calendar")
    calendar = _Calendar("https://dav/cal/1/", _Collection([obj], "token-1"))
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]

    changes = asyncio.run(plugin.get_calendar_events())

    assert changes == []
    assert plugin.snapshot == {}


# ---------- 轮询 ----------


def test_poll_once_force_full_clears_tokens(monkeypatch) -> None:
    plugin = _plugin()
    plugin.sync_tokens = {"cal": "token"}
    plugin._last_full_sync = datetime.now(TZ) - timedelta(hours=1)
    calls: list[bool] = []

    async def fake_changes() -> list[notify_module.Change]:
        calls.append(True)
        return []

    _override(plugin, "get_calendar_events", fake_changes)

    count, jobs = asyncio.run(plugin.poll_once(force_full=True))

    assert calls == [True]
    assert plugin.sync_tokens == {}
    assert count == 0
    assert jobs == 0


def test_poll_once_auto_full_sync_after_interval(monkeypatch) -> None:
    plugin = _plugin()
    plugin.sync_tokens = {"cal": "token"}
    old = datetime.now(TZ) - timedelta(hours=notify_module.FULL_SYNC_INTERVAL_HOURS + 1)
    plugin._last_full_sync = old

    async def fake_changes() -> list[notify_module.Change]:
        return []

    _override(plugin, "get_calendar_events", fake_changes)

    asyncio.run(plugin.poll_once())

    assert plugin.sync_tokens == {}
    assert plugin._last_full_sync > old


def test_poll_once_keeps_tokens_within_interval() -> None:
    plugin = _plugin()
    plugin.sync_tokens = {"cal": "token"}
    plugin._last_full_sync = datetime.now(TZ)

    async def fake_changes() -> list[notify_module.Change]:
        return []

    _override(plugin, "get_calendar_events", fake_changes)

    asyncio.run(plugin.poll_once())

    assert plugin.sync_tokens == {"cal": "token"}


def test_poll_once_returns_failure_and_keeps_jobs(monkeypatch, caplog) -> None:
    plugin = _plugin()
    plugin._jobs_by_owner = {"VEVENT:u": {"a", "b"}}
    plugin._last_full_sync = datetime.now(TZ)

    async def fake_changes() -> list[notify_module.Change]:
        raise OSError("network down")

    _override(plugin, "get_calendar_events", fake_changes)

    with caplog.at_level("ERROR"):
        count, jobs = asyncio.run(plugin.poll_once())

    assert count == -1
    assert jobs == 2
    assert "拉取 CalDAV 失败" in caplog.text


def test_poll_once_notifies_changes() -> None:
    plugin = _plugin()
    plugin._last_full_sync = datetime.now(TZ)
    change = notify_module.Change(
        uid="u", href="h.ics", kind=notify_module.ChangeKind.ADDED
    )
    notified: list[list[notify_module.Change]] = []

    async def fake_changes() -> list[notify_module.Change]:
        return [change]

    async def fake_notify(changes: list[notify_module.Change]) -> None:
        notified.append(changes)

    _override(plugin, "get_calendar_events", fake_changes)
    _override(plugin, "notify_events", fake_notify)

    count, jobs = asyncio.run(plugin.poll_once())

    assert count == 1
    assert jobs == 0
    assert notified == [[change]]


# ---------- 聊天命令 ----------


def test_sync_command_reports_counts(monkeypatch) -> None:
    plugin = _plugin()
    event = _Event()

    async def fake_poll(*, force_full: bool = False) -> tuple[int, int]:
        assert force_full is True
        return 3, 5

    _override(plugin, "poll_once", fake_poll)

    asyncio.run(plugin._handle_sync_command(event))

    assert event.replies == ["同步完成：3 条变更，当前待发提醒 5 条。"]


def test_sync_command_reports_failure(monkeypatch) -> None:
    plugin = _plugin()
    event = _Event()

    async def fake_poll(*, force_full: bool = False) -> tuple[int, int]:
        return -1, 0

    _override(plugin, "poll_once", fake_poll)

    asyncio.run(plugin._handle_sync_command(event))

    assert event.replies == ["同步失败，详见日志。"]


def test_sync_command_reports_exception(monkeypatch, caplog) -> None:
    plugin = _plugin()
    event = _Event()

    async def fake_poll(*, force_full: bool = False) -> tuple[int, int]:
        raise RuntimeError("boom")

    _override(plugin, "poll_once", fake_poll)

    with caplog.at_level("ERROR"):
        asyncio.run(plugin._handle_sync_command(event))

    assert event.replies == ["同步出错，详见日志。"]
    assert "同步异常" in caplog.text


def test_sync_command_times_out_and_releases_lock(monkeypatch, caplog) -> None:
    plugin = _plugin()
    event = _Event()
    monkeypatch.setattr(notify_module, "SYNC_TIMEOUT_SECONDS", 0.01)

    async def slow_poll(*, force_full: bool = False) -> tuple[int, int]:
        await asyncio.sleep(1)
        return 0, 0

    _override(plugin, "poll_once", slow_poll)

    with caplog.at_level("ERROR"):
        asyncio.run(plugin._handle_sync_command(event))

    assert event.replies == ["同步超时（>0.01s），详见日志。"]
    assert plugin._sync_lock.locked() is False


# ---------- 发送提醒 ----------


def test_send_notification_posts_message_and_forgets_job() -> None:
    plugin = _plugin()
    plugin._jobs_by_owner = {"VEVENT:uid": {"job"}}

    asyncio.run(
        plugin.send_notification(
            "Meeting", "开始时间：2026-11-01 10:00", "VEVENT:uid", "job"
        )
    )

    assert plugin.api.qq.messages == [  # type: ignore[union-attr]
        (100, "日程提醒：Meeting\n开始时间：2026-11-01 10:00")
    ]
    assert plugin._jobs_by_owner == {"VEVENT:uid": set()}


def test_send_notification_words_todo_message() -> None:
    plugin = _plugin()
    plugin._jobs_by_owner = {"VTODO:uid": {"job"}}

    asyncio.run(
        plugin.send_notification(
            "Buy milk", "截止时间：2026-11-05 18:00", "VTODO:uid", "job"
        )
    )

    assert plugin.api.qq.messages == [  # type: ignore[union-attr]
        (100, "待办提醒：Buy milk\n截止时间：2026-11-05 18:00")
    ]


def test_send_notification_forgets_one_shot_job() -> None:
    plugin = _plugin()
    job_id = "caldav_notify:VTODO:uid:2026-11-05T18:00:00"
    plugin._jobs_by_owner = {"VTODO:uid": {job_id}}

    asyncio.run(
        plugin.send_notification(
            "清单项", "截止时间：2026-11-05 18:00", "VTODO:uid", job_id
        )
    )

    assert plugin.api.qq.messages == [  # type: ignore[union-attr]
        (100, "待办提醒：清单项\n截止时间：2026-11-05 18:00")
    ]
    assert plugin._jobs_by_owner == {"VTODO:uid": set()}


def test_send_notification_without_target_logs_and_skips(caplog) -> None:
    plugin = _plugin(target_id=0)
    plugin._jobs_by_owner = {"VEVENT:uid": {"job"}}

    with caplog.at_level("ERROR"):
        asyncio.run(plugin.send_notification("Meeting", "when", "VEVENT:uid", "job"))

    assert plugin.api.qq.messages == []  # type: ignore[union-attr]
    assert "未配置 target_id" in caplog.text
    assert plugin._jobs_by_owner == {"VEVENT:uid": set()}


def test_send_notification_forgets_job_even_when_send_fails() -> None:
    plugin = _plugin()
    plugin.api.qq.error = RuntimeError("send failed")  # type: ignore[union-attr]
    plugin._jobs_by_owner = {"VEVENT:uid": {"job"}}

    with pytest.raises(RuntimeError, match="send failed"):
        asyncio.run(plugin.send_notification("S", "W", "VEVENT:uid", "job"))

    assert plugin._jobs_by_owner == {"VEVENT:uid": set()}


def test_scheduled_poll_runs_under_lock() -> None:
    plugin = _plugin()
    called: list[bool] = []

    async def fake_poll_once(*, force_full: bool = False) -> tuple[int, int]:
        called.append(plugin._sync_lock.locked())
        return 0, 0

    _override(plugin, "poll_once", fake_poll_once)

    asyncio.run(plugin._scheduled_poll())

    assert called == [True]
    assert plugin._sync_lock.locked() is False
