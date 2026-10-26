"""合并转发解析测试。"""

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncatbot.types import MessageArray, PlainText, parse_segment
from ncatbot.types.qq import Forward, ForwardNode

from plugins.chat_bot.chat_bot import (
    _MAX_FORWARD_CHARS,
    _MAX_FORWARD_IMAGES,
    AIPlugin,
    _render_forward,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _no_fetch(forward_id: str) -> Sequence[ForwardNode]:
    raise AssertionError(f"不应拉取转发 {forward_id}")


def _node(
    name: str,
    *segments: dict[str, Any],
    uid: str = "1",
) -> ForwardNode:
    return ForwardNode(
        user_id=uid,
        nickname=name,
        content=[parse_segment(seg) for seg in segments],
    )


def test_render_inline_forward_content() -> None:
    forward = Forward(
        content=[
            _node(
                "Alice",
                {"type": "text", "data": {"text": "你好"}},
                {"type": "image", "data": {"file": "a.jpg", "url": "https://x"}},
                {"type": "at", "data": {"qq": "2"}},
            )
        ]
    )

    result = _run(_render_forward(forward, _no_fetch))

    assert result.text == "<forward>\nAlice：你好[图片]@2\n</forward>"
    assert result.images == ["https://x"]


def test_render_forward_by_id_fetches_nodes() -> None:
    fetched: list[str] = []

    async def fetch(forward_id: str) -> Sequence[ForwardNode]:
        fetched.append(forward_id)
        return [_node("Bob", {"type": "text", "data": {"text": "拉取成功"}})]

    result = _run(_render_forward(Forward(id="fid"), fetch))

    assert fetched == ["fid"]
    assert result.text == "<forward>\nBob：拉取成功\n</forward>"


def test_render_forward_fetch_failure_returns_placeholder() -> None:
    async def fetch(forward_id: str) -> Sequence[ForwardNode]:
        raise RuntimeError("network down")

    result = _run(_render_forward(Forward(id="bad"), fetch))

    assert result.text == "[合并转发（获取失败）]"


def test_render_forward_without_content_or_id() -> None:
    assert _run(_render_forward(Forward(), _no_fetch)).text == "[合并转发]"


def test_render_nested_forward_inline() -> None:
    nested = Forward(
        content=[_node("Inner", {"type": "text", "data": {"text": "内层"}})]
    )
    outer_node = _node("Outer")
    outer_node.content.append(nested)
    outer = Forward(content=[outer_node])

    result = _run(_render_forward(outer, _no_fetch))

    assert "Outer：" in result.text
    assert "<forward>\nInner：内层\n</forward>" in result.text


def test_render_forward_truncates_long_content() -> None:
    forward = Forward(
        content=[_node("A", {"type": "text", "data": {"text": "x" * 5000}})]
    )

    result = _run(_render_forward(forward, _no_fetch))

    assert len(result.text) > _MAX_FORWARD_CHARS
    assert result.text.startswith("<forward>\nA：xxx")
    assert result.text.endswith("...（合并转发过长，已截断）")


def test_render_forward_collects_image_urls() -> None:
    forward = Forward(
        content=[
            _node(
                "A",
                {"type": "image", "data": {"file": "a.jpg", "url": "https://x"}},
                {"type": "image", "data": {"file": "b.jpg", "url": "base64://QUJD"}},
                {"type": "image", "data": {"file": "/tmp/c.jpg"}},
            )
        ]
    )

    result = _run(_render_forward(forward, _no_fetch))

    assert result.text == "<forward>\nA：[图片][图片][图片]\n</forward>"
    assert result.images == ["https://x", "data:image/png;base64,QUJD"]


def test_render_forward_limits_images() -> None:
    segments = [
        {"type": "image", "data": {"file": f"{i}.jpg", "url": f"https://x/{i}"}}
        for i in range(_MAX_FORWARD_IMAGES + 2)
    ]
    forward = Forward(content=[_node("A", *segments)])

    result = _run(_render_forward(forward, _no_fetch))

    assert len(result.images) == _MAX_FORWARD_IMAGES
    assert result.images[0] == "https://x/0"
    assert result.text.count("[图片]") == _MAX_FORWARD_IMAGES + 2


class _FakePlugin:
    async def _fetch_forward_nodes(self, forward_id: str) -> Sequence[ForwardNode]:
        return [_node("Bob", {"type": "text", "data": {"text": f"fid={forward_id}"}})]


def test_resolve_message_multimodal_includes_forward() -> None:
    message = MessageArray.from_any(
        [
            PlainText(text="看这个"),
            Forward(id="f1"),
        ]
    )

    text, parts = _run(
        AIPlugin.resolve_message_multimodal(cast(Any, _FakePlugin()), message)
    )

    assert "看这个" in text
    assert "\n<forward>\nBob：fid=f1\n</forward>" in text
    assert parts == []


class _FakePluginWithImage:
    async def _fetch_forward_nodes(self, forward_id: str) -> Sequence[ForwardNode]:
        return [
            _node(
                "Bob",
                {"type": "image", "data": {"file": "a.jpg", "url": "https://x"}},
            )
        ]


def test_resolve_message_multimodal_includes_forward_images() -> None:
    message = MessageArray.from_any([Forward(id="f1")])

    text, parts = _run(
        AIPlugin.resolve_message_multimodal(cast(Any, _FakePluginWithImage()), message)
    )

    assert text == "<forward>\nBob：[图片]\n</forward>\n"
    assert {"type": "image_url", "image_url": {"url": "https://x"}} in parts


class _Sender:
    user_id = "42"
    nickname = "Nick"
    card = None


class _ForwardMsg:
    sender = _Sender()
    message: ClassVar[list[dict[str, Any]]] = [
        {"type": "text", "data": {"text": "hi"}},
        {"type": "mface", "data": {"summary": "unsupported"}},
    ]


class _ForwardData:
    messages: ClassVar[list[_ForwardMsg]] = [_ForwardMsg()]


class _FakeApi:
    class _QQ:
        class _Query:
            async def get_forward_msg(self, forward_id: str) -> _ForwardData:
                return _ForwardData()

        query = _Query()

    qq = _QQ()


def test_fetch_forward_nodes_converts_messages_and_skips_unknown_segments() -> None:
    plugin = object.__new__(AIPlugin)
    plugin.api = _FakeApi()  # type: ignore[assignment]

    nodes = _run(AIPlugin._fetch_forward_nodes(plugin, "f1"))

    assert len(nodes) == 1
    assert nodes[0].user_id == "42"
    assert nodes[0].nickname == "Nick"
    assert [seg.text for seg in nodes[0].content] == ["hi"]
