"""cmach_socket_plugin 的 hello 命令测试。"""

import asyncio

from plugins.cmach_socket_plugin import plugin as plugin_module


class _Event:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply(self, *, text: str) -> None:
        self.replies.append(text)


def _plugin() -> plugin_module.CmachSocketPlugin:
    plugin = object.__new__(plugin_module.CmachSocketPlugin)
    plugin.name = "template-test"
    return plugin


def test_group_hello_replies_hi() -> None:
    event = _Event()

    asyncio.run(_plugin().on_group_hello(event))  # type: ignore[arg-type]

    assert event.replies == ["hi"]


def test_private_hello_replies_hi() -> None:
    event = _Event()

    asyncio.run(_plugin().on_private_hello(event))  # type: ignore[arg-type]

    assert event.replies == ["hi"]


def test_lifecycle_hooks_log_without_error(caplog) -> None:
    plugin = _plugin()

    with caplog.at_level("INFO"):
        asyncio.run(plugin.on_load())
        asyncio.run(plugin.on_close())

    assert "已加载" in caplog.text
    assert "已卸载" in caplog.text
