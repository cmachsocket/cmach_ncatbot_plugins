"""RSS 摘要插件测试。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from plugins.send_rss_summary import rss as rss_module


class _FakeScheduler:
    instances: ClassVar[list[_FakeScheduler]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.jobs: list[tuple[Any, Any, dict[str, Any]]] = []
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def add_job(self, func: Any, trigger: Any, **kwargs: Any) -> None:
        self.jobs.append((func, trigger, kwargs))

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.stopped = True


class _QQ:
    def __init__(self) -> None:
        self.files: list[tuple[int, str]] = []

    async def send_private_file(self, target_id: int, path: str) -> None:
        self.files.append((target_id, path))


class _Api:
    def __init__(self) -> None:
        self.qq = _QQ()


def _plugin(config: dict[str, Any]) -> rss_module.RssSender:
    plugin = object.__new__(rss_module.RssSender)
    plugin.name = "rss-test"
    plugin.config = config
    plugin.target_id = config.get("target_id", 0)
    plugin.api = _Api()  # type: ignore[assignment]
    return plugin


def _patch_scheduler(monkeypatch) -> None:
    _FakeScheduler.instances.clear()
    monkeypatch.setattr(rss_module, "AsyncIOScheduler", _FakeScheduler)


def test_config_path_points_into_plugin_directory() -> None:
    assert rss_module.CONFIG_PATH == (
        Path(rss_module.__file__).resolve().parent / "config.yaml"
    )


def test_on_load_without_target_id_skips_scheduler(monkeypatch, caplog) -> None:
    _patch_scheduler(monkeypatch)
    plugin = _plugin({})

    with caplog.at_level("ERROR"):
        asyncio.run(plugin.on_load())

    assert "未配置 target_id" in caplog.text
    assert _FakeScheduler.instances == []
    assert not hasattr(plugin, "scheduler")


def test_on_load_schedules_daily_summary(monkeypatch, caplog) -> None:
    _patch_scheduler(monkeypatch)
    plugin = _plugin({"target_id": 123})

    with caplog.at_level("INFO"):
        asyncio.run(plugin.on_load())

    assert plugin.target_id == 123
    scheduler = _FakeScheduler.instances[0]
    assert scheduler.kwargs == {"timezone": "Asia/Shanghai"}
    assert scheduler.started is True
    func, trigger, kwargs = scheduler.jobs[0]
    assert func == plugin.send_rss_summary
    assert trigger == "cron"
    assert kwargs == {"hour": 6, "minute": 0}
    assert "已加载" in caplog.text


def test_on_close_shuts_down_scheduler(monkeypatch, caplog) -> None:
    _patch_scheduler(monkeypatch)
    plugin = _plugin({"target_id": 123})
    asyncio.run(plugin.on_load())
    scheduler = _FakeScheduler.instances[0]

    with caplog.at_level("INFO"):
        asyncio.run(plugin.on_close())

    assert scheduler.stopped is True
    assert "已卸载" in caplog.text


def test_send_rss_summary_sends_generated_file(monkeypatch) -> None:
    plugin = _plugin({"target_id": 321})
    summary_path = Path("/tmp/summary.md")
    config = object()
    calls: list[object] = []

    monkeypatch.setattr(rss_module.rssfetch, "load_config", lambda path: config)

    async def fake_run(received: object) -> SimpleNamespace:
        calls.append(received)
        return SimpleNamespace(summary_path=summary_path)

    monkeypatch.setattr(rss_module.rssfetch, "run", fake_run)

    asyncio.run(plugin.send_rss_summary())

    assert calls == [config]
    assert plugin.api.qq.files == [(321, str(summary_path))]  # type: ignore[union-attr]


def test_send_rss_summary_logs_when_generation_fails(monkeypatch, caplog) -> None:
    plugin = _plugin({"target_id": 321})

    monkeypatch.setattr(rss_module.rssfetch, "load_config", lambda path: object())

    async def fake_run(received: object) -> SimpleNamespace:
        return SimpleNamespace(summary_path=None)

    monkeypatch.setattr(rss_module.rssfetch, "run", fake_run)

    with caplog.at_level("ERROR"):
        asyncio.run(plugin.send_rss_summary())

    assert "RSS 摘要生成失败" in caplog.text
    assert plugin.api.qq.files == []  # type: ignore[union-attr]
