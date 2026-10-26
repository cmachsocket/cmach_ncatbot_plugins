"""WOL 插件命令测试。"""

import asyncio

from plugins.wol import wol as wol_module


class _Event:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply(self, *, text: str) -> None:
        self.replies.append(text)


def _plugin() -> wol_module.WolPlugin:
    plugin = object.__new__(wol_module.WolPlugin)
    plugin.name = "wol-test"
    return plugin


def test_private_hello_replies_hi() -> None:
    event = _Event()

    asyncio.run(_plugin().on_private_hello(event))  # type: ignore[arg-type]

    assert event.replies == ["hi"]


def test_private_wol_wakes_both_interfaces(monkeypatch) -> None:
    event = _Event()
    woken: list[str] = []
    monkeypatch.setattr(wol_module, "wake", woken.append)

    asyncio.run(_plugin().on_private_wol(event))  # type: ignore[arg-type]

    assert event.replies == ["正在发送唤醒包...", "唤醒包已发送"]
    assert woken == [wol_module.mac_enp4s0, wol_module.mac_wlp2s0]


def test_lifecycle_hooks_log_without_error(caplog) -> None:
    plugin = _plugin()

    with caplog.at_level("INFO"):
        asyncio.run(plugin.on_load())
        asyncio.run(plugin.on_close())

    assert "已加载" in caplog.text
    assert "已卸载" in caplog.text
