"""CalDAV 日程提醒插件测试（不访问网络）。"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from apscheduler.jobstores.base import JobLookupError
from apscheduler.jobstores.memory import MemoryJobStore
from icalendar import Calendar
from icalendar.cal import Event as ICalEvent

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
    plugin.target_id = 100
    plugin.scheduler = _FakeScheduler()  # type: ignore[assignment]
    plugin._jobs_by_uid = {}
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
        "timezone": "UTC",
    }

    with caplog.at_level("INFO"):
        asyncio.run(plugin.on_load())

    assert plugin.target_id == 42
    assert plugin.notify_before_minutes == 0
    assert plugin.notify_window_days == 1
    assert plugin.all_day_notify_hour == 22
    assert plugin.all_day_notify_minute == 15
    assert plugin.tz.key == "UTC"
    assert created["dav"] == {
        "url": "https://dav.example/",
        "username": "alice",
        "password": "secret",
    }
    assert scheduler.started is True
    assert "caldav_poll" in scheduler.jobs
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

    plugin._schedule_occurrence(event, now)

    job_id = "caldav_notify:uid-1:2026-11-01T10:00:00"
    job = plugin.scheduler.jobs[job_id]  # type: ignore[index]
    assert job["args"] == ["Meeting", "2026-11-01 10:00", "uid-1", job_id]
    assert job["trigger"].run_date == datetime(2026, 11, 1, 9, 50, tzinfo=TZ)
    assert plugin._jobs_by_uid["uid-1"] == {job_id}


def test_schedule_all_day_occurrence_notifies_previous_evening() -> None:
    plugin = _plugin()
    event = _event("UID:all-day\nSUMMARY:Holiday\nDTSTART;VALUE=DATE:20261102")
    now = datetime(2026, 10, 26, 12, 0, tzinfo=TZ)

    plugin._schedule_occurrence(event, now)

    job_id = "caldav_notify:all-day:2026-11-02"
    job = plugin.scheduler.jobs[job_id]  # type: ignore[index]
    assert job["args"] == ["Holiday", "2026-11-02 全天", "all-day", job_id]
    assert job["trigger"].run_date == datetime(2026, 11, 1, 23, 30, tzinfo=TZ)


def test_schedule_uses_recurrence_id_as_anchor() -> None:
    plugin = _plugin()
    event = _event(
        "UID:rec\nSUMMARY:Weekly\nDTSTART:20261101T100000\n"
        "RECURRENCE-ID:20261108T100000"
    )
    now = datetime(2026, 10, 26, 12, 0, tzinfo=TZ)

    plugin._schedule_occurrence(event, now)

    assert plugin._jobs_by_uid["rec"] == {"caldav_notify:rec:2026-11-08T10:00:00"}


def test_schedule_skips_cancelled_occurrence() -> None:
    plugin = _plugin()
    event = _event(
        "UID:uid-1\nSUMMARY:Meeting\nSTATUS:CANCELLED\nDTSTART:20261101T100000"
    )

    plugin._schedule_occurrence(event, datetime(2026, 10, 26, tzinfo=TZ))

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

    plugin._schedule_occurrence(_event(body), now)

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def test_schedule_skips_past_all_day_occurrence() -> None:
    plugin = _plugin()
    event = _event("UID:old\nSUMMARY:Old\nDTSTART;VALUE=DATE:20261025")

    plugin._schedule_occurrence(event, datetime(2026, 10, 26, 12, 0, tzinfo=TZ))

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def test_schedule_ignores_unsupported_start_value() -> None:
    plugin = _plugin()
    event = _event("UID:clock\nSUMMARY:Clock\nDTSTART:20261101T100000")
    event["DTSTART"] = SimpleNamespace(dt="not a date")

    plugin._schedule_occurrence(event, datetime(2026, 10, 26, 12, 0, tzinfo=TZ))

    assert plugin.scheduler.jobs == {}  # type: ignore[attr-defined]


def test_cancel_notifications_removes_all_jobs_and_swallows_missing_onces() -> None:
    plugin = _plugin()
    plugin.scheduler.jobs = {"a": {}, "b": {}}  # type: ignore[union-attr]
    plugin._jobs_by_uid = {"uid": {"a", "missing", "b"}}

    plugin._cancel_notifications("uid")

    assert plugin.scheduler.jobs == {}  # type: ignore[union-attr]
    assert "uid" not in plugin._jobs_by_uid


def test_forget_job_discards_from_index() -> None:
    plugin = _plugin()
    plugin._jobs_by_uid = {"uid": {"a", "b"}}

    plugin._forget_job("uid", "a")
    plugin._forget_job("unknown", "x")

    assert plugin._jobs_by_uid == {"uid": {"b"}}


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
    _override(plugin, "_schedule_occurrence", lambda occ, now: scheduled.append(occ))

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
    _override(plugin, "_schedule_occurrence", lambda occ, now: pytest.fail("不应调度"))

    def fake_cancel(uid: str) -> None:
        cancelled.append(uid)

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

    assert cancelled == ["gone", "no-calendar"]


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
        uid: str = "",
    ) -> None:
        self.url = _Url(href)
        self.data = data
        self._calendar = calendar
        self.icalendar_component = {"UID": uid}

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
    added = _RemoteObject(
        "https://dav/cal/1/a.ics", calendar=calendar_data, uid="uid-1"
    )
    deleted = _RemoteObject("https://dav/cal/1/b.ics", data=None, uid="ignored")
    collection = _Collection([added, deleted], "token-1")
    calendar = _Calendar("https://dav/cal/1/", collection)
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]
    plugin.snapshot = {
        "https://dav/cal/1/b.ics": "uid-b",
        "https://dav/cal/1/stale.ics": "uid-stale",
    }

    changes = asyncio.run(plugin.get_calendar_events())

    assert calendar.received == (None, True)
    assert plugin.sync_tokens == {"https://dav/cal/1/": "token-1"}
    assert plugin.snapshot == {"https://dav/cal/1/a.ics": "uid-1"}
    kinds = {(change.href, change.kind, change.uid) for change in changes}
    assert kinds == {
        ("https://dav/cal/1/a.ics", notify_module.ChangeKind.ADDED, "uid-1"),
        ("https://dav/cal/1/b.ics", notify_module.ChangeKind.DELETED, "uid-b"),
        (
            "https://dav/cal/1/stale.ics",
            notify_module.ChangeKind.DELETED,
            "uid-stale",
        ),
    }


def test_get_calendar_events_incremental_marks_existing_as_updated() -> None:
    calendar_data = _ics(
        "BEGIN:VEVENT\nUID:uid-1\nSUMMARY:One\nDTSTART:20261101T100000\nEND:VEVENT"
    )
    obj = _RemoteObject("https://dav/cal/1/a.ics", calendar=calendar_data, uid="uid-1")
    calendar = _Calendar("https://dav/cal/1/", _Collection([obj], "token-2"))
    plugin = _plugin()
    plugin.davclient = _Dav([calendar])  # type: ignore[assignment]
    plugin.sync_tokens = {"https://dav/cal/1/": "token-1"}
    plugin.snapshot = {"https://dav/cal/1/a.ics": "uid-1"}

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
    obj = _RemoteObject(
        "https://dav/cal/1/a.ics", calendar="not a calendar", uid="uid-1"
    )
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
    plugin._jobs_by_uid = {"u": {"a", "b"}}
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
    plugin._jobs_by_uid = {"uid": {"job"}}

    asyncio.run(plugin.send_notification("Meeting", "2026-11-01 10:00", "uid", "job"))

    assert plugin.api.qq.messages == [  # type: ignore[union-attr]
        (100, "日程提醒：Meeting\n开始时间：2026-11-01 10:00")
    ]
    assert plugin._jobs_by_uid == {"uid": set()}


def test_send_notification_without_target_logs_and_skips(caplog) -> None:
    plugin = _plugin(target_id=0)
    plugin._jobs_by_uid = {"uid": {"job"}}

    with caplog.at_level("ERROR"):
        asyncio.run(plugin.send_notification("Meeting", "when", "uid", "job"))

    assert plugin.api.qq.messages == []  # type: ignore[union-attr]
    assert "未配置 target_id" in caplog.text
    assert plugin._jobs_by_uid == {"uid": set()}


def test_send_notification_forgets_job_even_when_send_fails() -> None:
    plugin = _plugin()
    plugin.api.qq.error = RuntimeError("send failed")  # type: ignore[union-attr]
    plugin._jobs_by_uid = {"uid": {"job"}}

    with pytest.raises(RuntimeError, match="send failed"):
        asyncio.run(plugin.send_notification("S", "W", "uid", "job"))

    assert plugin._jobs_by_uid == {"uid": set()}


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
