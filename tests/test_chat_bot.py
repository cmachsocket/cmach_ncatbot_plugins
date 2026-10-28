"""chat_bot 插件的上下文、多模态解析和工具测试。"""

import asyncio
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from ncatbot.types import MessageArray
from pydantic import Field

from plugins.chat_bot import chat_bot as chat_module


class _Query:
    def __init__(self) -> None:
        self.member_info: Any = SimpleNamespace(card="群名片", nickname="昵称")
        self.member_error: Exception | None = None
        self.messages: dict[str, Any] = {}
        self.ptt_text: str = ""
        self.ptt_error: Exception | None = None
        self.ptt_calls: list[str] = []

    async def get_group_member_info(self, group_id: int, user_id: int | str) -> Any:
        if self.member_error is not None:
            raise self.member_error
        return self.member_info

    async def get_msg(self, message_id: str) -> Any:
        return self.messages.get(message_id)

    async def fetch_ptt_text(self, message_id: str) -> Any:
        self.ptt_calls.append(message_id)
        if self.ptt_error is not None:
            raise self.ptt_error
        return SimpleNamespace(text=self.ptt_text)


class _QQ:
    def __init__(self) -> None:
        self.query = _Query()
        self.texts: list[tuple[int, str]] = []

    async def send_group_text(self, group_id: int, text: str) -> None:
        self.texts.append((group_id, text))


class _Api:
    def __init__(self) -> None:
        self.qq = _QQ()


def _plugin() -> chat_module.AIPlugin:
    plugin = object.__new__(chat_module.AIPlugin)
    plugin.name = "chat-test"
    plugin.api = _Api()  # type: ignore[assignment]
    plugin.target_group_id = 123
    plugin.bot_id = "999"
    plugin.assistent_messages = []
    plugin.max_k = 60
    plugin.model = "test-model"
    plugin.chat_llm = None
    plugin.agent = None
    return plugin


# ---------- 上下文 ----------


def test_add_context_appends_human_and_ai_messages() -> None:
    plugin = _plugin()

    plugin.add_context("你好呀", "Alice: hi")

    assert [type(m) for m in plugin.assistent_messages] == [
        HumanMessage,
        AIMessage,
    ]
    assert plugin.assistent_messages[0].content == "Alice: hi"
    assert plugin.assistent_messages[1].content == "你好呀"
    assert all(m.id is not None for m in plugin.assistent_messages)


def test_add_context_uses_placeholder_for_silence() -> None:
    plugin = _plugin()

    plugin.add_context("", "Alice: hi")

    assert plugin.assistent_messages[-1].content == "...（已读未回）"


def test_add_context_keeps_sliding_window() -> None:
    plugin = _plugin()
    plugin.max_k = 4

    for i in range(3):
        plugin.add_context(f"reply {i}", f"user {i}")

    assert len(plugin.assistent_messages) == 4
    assert [m.content for m in plugin.assistent_messages] == [
        "user 1",
        "reply 1",
        "user 2",
        "reply 2",
    ]


# ---------- 群过滤与用户信息 ----------


def test_is_target_group_compares_strings() -> None:
    plugin = _plugin()

    assert plugin.is_target_group(123) is True
    assert plugin.is_target_group("123") is True
    assert plugin.is_target_group(456) is False


def test_get_user_name_returns_all_members_placeholder() -> None:
    plugin = _plugin()

    assert asyncio.run(plugin.get_user_name("all")) == "全体成员"


def test_get_user_name_prefers_group_card() -> None:
    plugin = _plugin()

    assert asyncio.run(plugin.get_user_name(1)) == "群名片"


def test_get_user_name_falls_back_to_nickname() -> None:
    plugin = _plugin()
    plugin.api.qq.query.member_info = SimpleNamespace(card="", nickname="昵称")  # type: ignore[union-attr]

    assert asyncio.run(plugin.get_user_name(1)) == "昵称"


def test_get_user_name_falls_back_to_unknown() -> None:
    plugin = _plugin()
    plugin.api.qq.query.member_info = None  # type: ignore[union-attr]

    assert asyncio.run(plugin.get_user_name(1)) == "未知用户"


def test_get_user_name_handles_query_error(caplog) -> None:
    plugin = _plugin()
    plugin.api.qq.query.member_error = ConnectionError("offline")  # type: ignore[union-attr]

    with caplog.at_level("WARNING"):
        name = asyncio.run(plugin.get_user_name(1))

    assert name == "未知用户"
    assert "获取群成员信息失败" in caplog.text


# ---------- 多模态解析 ----------


def _message(*segments: dict[str, Any]) -> MessageArray:
    return MessageArray.from_list(list(segments))


def test_resolve_message_plain_text_and_at(monkeypatch) -> None:
    plugin = _plugin()

    async def fake_name(user_id: int | str) -> str:
        return {8: "Bob", 9: "Carol"}.get(int(user_id), "未知用户")

    monkeypatch.setattr(plugin, "get_user_name", fake_name)
    text, parts = asyncio.run(
        plugin.resolve_message_multimodal(
            _message(
                {"type": "text", "data": {"text": "hi "}},
                {"type": "at", "data": {"qq": "8"}},
                {"type": "text", "data": {"text": " and "}},
                {"type": "at", "data": {"qq": "9"}},
            )
        )
    )

    assert text == "hi @Bob  and @Carol "
    assert parts == []


def test_resolve_message_adds_image_parts(monkeypatch) -> None:
    plugin = _plugin()

    async def fake_name(user_id: int | str) -> str:
        return "未知用户"

    monkeypatch.setattr(plugin, "get_user_name", fake_name)
    text, parts = asyncio.run(
        plugin.resolve_message_multimodal(
            _message(
                {"type": "text", "data": {"text": "look"}},
                {
                    "type": "image",
                    "data": {"file": "a.jpg", "url": "https://img/a.jpg"},
                },
            )
        )
    )

    assert text == "look[图片] "
    assert parts == [
        {"type": "text", "text": "look"},
        {"type": "text", "text": "[图片] "},
        {"type": "image_url", "image_url": {"url": "https://img/a.jpg"}},
    ]


def test_resolve_message_converts_base64_image(monkeypatch) -> None:
    plugin = _plugin()
    monkeypatch.setattr(plugin, "get_user_name", lambda *_: _async("未知用户"))

    _, parts = asyncio.run(
        plugin.resolve_message_multimodal(
            _message(
                {"type": "image", "data": {"file": "base64://AAAA"}},
            )
        )
    )

    assert parts == [
        {"type": "text", "text": "[图片] "},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        },
    ]


def test_resolve_message_skips_image_without_source() -> None:
    plugin = _plugin()

    text, parts = asyncio.run(
        plugin.resolve_message_multimodal(
            _message({"type": "image", "data": {"file": ""}})
        )
    )

    assert text == ""
    assert parts == []


def test_resolve_message_renders_reply_quote(monkeypatch) -> None:
    plugin = _plugin()
    plugin.api.qq.query.messages["m1"] = SimpleNamespace(  # type: ignore[union-attr]
        sender=SimpleNamespace(user_id="7"),
        message=[{"type": "text", "data": {"text": "原消息"}}],
    )

    async def fake_name(user_id: int | str) -> str:
        assert str(user_id) == "7"
        return "Alice"

    monkeypatch.setattr(plugin, "get_user_name", fake_name)
    text, parts = asyncio.run(
        plugin.resolve_message_multimodal(
            _message(
                {"type": "reply", "data": {"id": "m1"}},
                {"type": "text", "data": {"text": "看看这个"}},
            )
        )
    )

    assert text == "<quote>\nAlice：原消息\n</quote>\n看看这个"
    assert parts == []


def test_resolve_message_skips_missing_reply_target(monkeypatch) -> None:
    plugin = _plugin()

    async def fake_name(user_id: int | str) -> str:
        return "Alice"

    monkeypatch.setattr(plugin, "get_user_name", fake_name)
    text, _ = asyncio.run(
        plugin.resolve_message_multimodal(
            _message(
                {"type": "reply", "data": {"id": "missing"}},
                {"type": "text", "data": {"text": "正文"}},
            )
        )
    )

    assert text == "<quote>\n</quote>\n正文"


def test_resolve_message_ignores_unknown_segments(caplog) -> None:
    plugin = _plugin()

    with caplog.at_level("INFO"):
        text, parts = asyncio.run(
            plugin.resolve_message_multimodal(
                _message(
                    {"type": "text", "data": {"text": "hi"}},
                    {"type": "face", "data": {"id": "1"}},
                )
            )
        )

    assert text == "hi"
    assert parts == []
    assert "跳过段" in caplog.text


# ---------- 语音转文字 ----------


def _fake_query(plugin: chat_module.AIPlugin) -> _Query:
    """取出测试用的假 query（真实类型是 ncatbot 的 QQQuery）。"""
    return cast(_Query, plugin.api.qq.query)


def test_resolve_message_transcribes_voice() -> None:
    plugin = _plugin()
    query = _fake_query(plugin)
    query.ptt_text = "今天天气不错"

    text, parts = asyncio.run(
        plugin.resolve_message_multimodal(
            _message({"type": "record", "data": {"file": "a.silk"}}),
            message_id="m1",
        )
    )

    assert text == "[语音] 今天天气不错"
    assert query.ptt_calls == ["m1"]
    assert parts == []


def test_resolve_message_voice_placeholder_when_transcription_fails(caplog) -> None:
    plugin = _plugin()
    _fake_query(plugin).ptt_error = ConnectionError("offline")

    with caplog.at_level("WARNING"):
        text, _ = asyncio.run(
            plugin.resolve_message_multimodal(
                _message({"type": "record", "data": {"file": "a.silk"}}),
                message_id="m1",
            )
        )

    assert text == "[语音]"
    assert "语音转文字失败" in caplog.text


def test_resolve_message_voice_placeholder_when_transcription_empty() -> None:
    plugin = _plugin()
    _fake_query(plugin).ptt_text = "  "

    text, _ = asyncio.run(
        plugin.resolve_message_multimodal(
            _message({"type": "record", "data": {"file": "a.silk"}}),
            message_id="m1",
        )
    )

    assert text == "[语音]"


def test_resolve_message_queries_ptt_once_per_message() -> None:
    plugin = _plugin()
    query = _fake_query(plugin)
    query.ptt_text = "在的"

    text, _ = asyncio.run(
        plugin.resolve_message_multimodal(
            _message(
                {"type": "record", "data": {"file": "a.silk"}},
                {"type": "record", "data": {"file": "b.silk"}},
            ),
            message_id="m1",
        )
    )

    assert text == "[语音] 在的[语音] 在的"
    assert query.ptt_calls == ["m1"]


def test_resolve_message_does_not_query_ptt_without_voice() -> None:
    plugin = _plugin()

    text, _ = asyncio.run(
        plugin.resolve_message_multimodal(
            _message({"type": "text", "data": {"text": "hi"}}),
            message_id="m1",
        )
    )

    assert text == "hi"
    assert _fake_query(plugin).ptt_calls == []


def test_fetch_voice_text_without_message_id_is_empty() -> None:
    plugin = _plugin()

    assert asyncio.run(plugin.fetch_voice_text(None)) == ""
    assert _fake_query(plugin).ptt_calls == []


def test_fetch_voice_text_returns_transcribed_text() -> None:
    plugin = _plugin()
    query = _fake_query(plugin)
    query.ptt_text = "  你好  "

    assert asyncio.run(plugin.fetch_voice_text("m1")) == "你好"
    assert query.ptt_calls == ["m1"]


def test_fetch_voice_text_swallows_api_error(caplog) -> None:
    plugin = _plugin()
    _fake_query(plugin).ptt_error = RuntimeError("boom")

    with caplog.at_level("WARNING"):
        assert asyncio.run(plugin.fetch_voice_text("m1")) == ""

    assert "语音转文字失败" in caplog.text


def test_should_process_accepts_voice_only_message() -> None:
    assert (
        chat_module.AIPlugin._should_process(
            _message({"type": "record", "data": {"file": "a.silk"}})
        )
        is True
    )


def test_should_process_rejects_empty_message() -> None:
    assert chat_module.AIPlugin._should_process(_message()) is False


# ---------- 发送与上下文工具 ----------


def _call_tool(**kwargs: Any) -> Any:
    coroutine = getattr(chat_module.send_message_tool, "coroutine", None)
    assert coroutine is not None
    return asyncio.run(coroutine(**kwargs))


def test_send_message_tool_sends_and_records_context() -> None:
    plugin = _plugin()
    event = SimpleNamespace(group_id=555, message_id="m1")

    result = _call_tool(
        content="hello",
        reply=True,
        plugin=plugin,
        event=event,
        user_msg="Alice: hi",
    )

    assert result == "sent"
    assert plugin.api.qq.texts == [(555, "hello")]  # type: ignore[union-attr]
    assert [m.content for m in plugin.assistent_messages] == [
        "Alice: hi",
        "hello",
    ]


def test_send_message_tool_skips_when_reply_false() -> None:
    plugin = _plugin()

    result = _call_tool(
        content="hello",
        reply=False,
        plugin=plugin,
        event=SimpleNamespace(group_id=555),
        user_msg="Alice: hi",
    )

    assert result == "skipped"
    assert plugin.api.qq.texts == []  # type: ignore[union-attr]
    assert plugin.assistent_messages[-1].content == "...（已读未回）"


def test_send_message_tool_skips_empty_content() -> None:
    plugin = _plugin()

    result = _call_tool(
        content="   ",
        reply=True,
        plugin=plugin,
        event=SimpleNamespace(group_id=555),
        user_msg="Alice: hi",
    )

    assert result == "skipped"
    assert plugin.api.qq.texts == []  # type: ignore[union-attr]


def test_send_message_tool_requires_plugin() -> None:
    result = _call_tool(content="hello", reply=True, plugin=None)

    assert result == "error: missing plugin"


def test_send_message_tool_uses_target_group_without_event() -> None:
    plugin = _plugin()

    result = _call_tool(
        content="proactive",
        reply=True,
        plugin=plugin,
        event=None,
        user_msg="",
    )

    assert result == "sent"
    assert plugin.api.qq.texts == [(123, "proactive")]  # type: ignore[union-attr]


def test_send_message_tool_warns_for_long_reply(caplog) -> None:
    plugin = _plugin()

    with caplog.at_level("WARNING"):
        _call_tool(
            content="x" * 21,
            reply=True,
            plugin=plugin,
            event=SimpleNamespace(group_id=555),
            user_msg="",
        )

    assert "超过 20" in caplog.text


def test_send_message_tool_allows_long_reply_when_unlimited(caplog) -> None:
    plugin = _plugin()

    with caplog.at_level("WARNING"):
        _call_tool(
            content="x" * 21,
            reply=True,
            unlimit_length=True,
            plugin=plugin,
            event=SimpleNamespace(group_id=555),
            user_msg="",
        )

    assert "超过 20" not in caplog.text


# ---------- 强制调用 send_message ----------


def _model_request() -> Any:
    return chat_module.ModelRequest(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="hi")])),
        messages=[],
    )


def test_force_send_message_middleware_forces_tool_choice() -> None:
    seen: list[Any] = []

    async def handler(request: Any) -> Any:
        seen.append(request.tool_choice)
        return chat_module.ModelResponse(result=[AIMessage(content="ok")])

    middleware = chat_module.ForceSendMessageMiddleware()

    result = asyncio.run(middleware.awrap_model_call(_model_request(), handler))

    assert seen == ["send_message"]
    assert result.result[0].content == "ok"


def test_force_send_message_middleware_degrades_when_backend_rejects_choice(
    caplog,
) -> None:
    seen: list[Any] = []

    async def handler(request: Any) -> Any:
        seen.append(request.tool_choice)
        if request.tool_choice == "send_message":
            raise RuntimeError("backend rejects named tool_choice")
        return chat_module.ModelResponse(result=[AIMessage(content="ok")])

    middleware = chat_module.ForceSendMessageMiddleware()

    with caplog.at_level("WARNING"):
        result = asyncio.run(middleware.awrap_model_call(_model_request(), handler))

    assert seen == ["send_message", None]
    assert result.result[0].content == "ok"
    assert "降级为不强制" in caplog.text


def test_ensure_send_message_records_silence_when_tool_not_called() -> None:
    plugin = _plugin()
    resp = {"messages": [AIMessage(content="直接输出的文本")]}

    asyncio.run(
        plugin._ensure_send_message(
            resp, SimpleNamespace(group_id=555), "Alice: hi", None
        )
    )

    # 直接输出的文本按设计不发送，只补一次沉默调用让这轮进历史
    assert plugin.api.qq.texts == []  # type: ignore[union-attr]
    assert plugin.assistent_messages[-1].content == "...（已读未回）"


def test_ensure_send_message_skips_when_tool_already_called() -> None:
    plugin = _plugin()
    resp = {
        "messages": [
            AIMessage(content="x"),
            ToolMessage(content="sent", tool_call_id="1", name="send_message"),
        ]
    }

    asyncio.run(
        plugin._ensure_send_message(
            resp, SimpleNamespace(group_id=555), "Alice: hi", None
        )
    )

    assert plugin.api.qq.texts == []  # type: ignore[union-attr]
    assert plugin.assistent_messages == []


def test_ensure_send_message_skips_proactive_mode() -> None:
    plugin = _plugin()
    resp = {"messages": [AIMessage(content="x")]}

    asyncio.run(plugin._ensure_send_message(resp, None, "", None))

    assert plugin.api.qq.texts == []  # type: ignore[union-attr]
    assert plugin.assistent_messages == []


class _ScriptedChatModel(BaseChatModel):
    """按剧本返回消息，并记录 bind_tools 收到的 tool_choice。"""

    script: list[AIMessage] = Field(default_factory=list)
    seen_tool_choice: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.seen_tool_choice.append(tool_choice)
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)


def test_agent_binds_send_message_with_forced_tool_choice(monkeypatch) -> None:
    plugin = _plugin()
    model = _ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "send_message",
                        "args": {"content": "回复", "reply": True},
                        "id": "call_1",
                    }
                ],
            )
        ]
    )
    plugin.agent = create_agent(
        model,
        tools=[chat_module.send_message_tool],
        middleware=[chat_module.ForceSendMessageMiddleware()],
        state_schema=chat_module.ChatAgentState,
    )

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(chat_module.asyncio, "sleep", fake_sleep)

    asyncio.run(
        plugin._run_agent(
            [HumanMessage(content="hi")],
            bank_id="u",
            event=SimpleNamespace(group_id=555),
            user_msg="Alice: hi",
        )
    )

    # 模型被强制绑定到 send_message 工具，消息也真的发出去了
    assert model.seen_tool_choice == ["send_message"]
    assert plugin.api.qq.texts == [(555, "回复")]  # type: ignore[union-attr]
    assert plugin.assistent_messages[-1].content == "回复"


def test_send_to_group_sends_non_empty_lines(monkeypatch) -> None:
    plugin = _plugin()

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(chat_module.asyncio, "sleep", fake_sleep)
    event = SimpleNamespace(group_id=555)

    asyncio.run(plugin._send_to_group(event, "a\n\n b\nc"))  # type: ignore[arg-type]

    assert plugin.api.qq.texts == [(555, "a"), (555, " b"), (555, "c")]  # type: ignore[union-attr]


def test_send_to_group_ignores_blank_content(monkeypatch) -> None:
    plugin = _plugin()

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(chat_module.asyncio, "sleep", fake_sleep)

    asyncio.run(plugin._send_to_group(SimpleNamespace(group_id=555), "  \n\n"))  # type: ignore[arg-type]

    assert plugin.api.qq.texts == []  # type: ignore[union-attr]


# ---------- Agent / token ----------


def test_set_hindsight_bank_updates_model_kwargs() -> None:
    plugin = _plugin()
    plugin.chat_llm = SimpleNamespace(model_kwargs={"keep": 1})  # type: ignore[assignment]

    plugin._set_hindsight_bank("user-1")

    assert plugin.chat_llm.model_kwargs == {"keep": 1, "hindsight_bank_id": "user-1"}  # type: ignore[union-attr]


def test_set_hindsight_bank_without_llm_is_noop() -> None:
    plugin = _plugin()

    plugin._set_hindsight_bank("user-1")


def test_run_agent_requires_initialized_agent() -> None:
    plugin = _plugin()

    with pytest.raises(RuntimeError, match="agent 未初始化"):
        asyncio.run(plugin._run_agent([HumanMessage(content="hi")], bank_id="u"))


def test_run_agent_passes_state_and_returns_response() -> None:
    plugin = _plugin()
    plugin.chat_llm = SimpleNamespace(model_kwargs={})  # type: ignore[assignment]

    class _Agent:
        def __init__(self) -> None:
            self.state: Any = None

        async def ainvoke(self, state: Any, verbose: bool = False) -> dict[str, Any]:
            self.state = state
            assert verbose is True
            return {"messages": [AIMessage(content="ok")]}

    agent = _Agent()
    plugin.agent = agent
    event = SimpleNamespace(group_id=555)

    response = asyncio.run(
        plugin._run_agent(
            [HumanMessage(content="hi")],
            bank_id="user-1",
            event=event,
            user_msg="Alice: hi",
        )
    )

    assert response == {"messages": [AIMessage(content="ok")]}
    assert agent.state["plugin"] is plugin
    assert agent.state["event"] is event
    assert agent.state["user_msg"] == "Alice: hi"
    assert len(agent.state["messages"]) == 1
    assert plugin.chat_llm.model_kwargs["hindsight_bank_id"] == "user-1"  # type: ignore[union-attr]
    # 模型没调用 send_message：补一次沉默调用，这轮仍然进历史
    assert plugin.assistent_messages[-1].content == "...（已读未回）"


def test_count_tokens_requires_model() -> None:
    plugin = _plugin()

    with pytest.raises(RuntimeError, match="chat_llm 未初始化"):
        plugin._count_tokens([HumanMessage(content="hi")])


def test_count_tokens_delegates_to_litellm(monkeypatch) -> None:
    plugin = _plugin()
    plugin.chat_llm = SimpleNamespace(model_kwargs={})  # type: ignore[assignment]
    received: dict[str, Any] = {}

    def fake_counter(*, model: str, messages: list[Any]) -> int:
        received["model"] = model
        received["messages"] = messages
        return 42

    monkeypatch.setattr(chat_module.litellm, "token_counter", fake_counter)

    assert plugin._count_tokens([HumanMessage(content="hi")]) == 42
    assert received["model"] == "test-model"
    assert received["messages"] == [{"role": "user", "content": "hi"}]


def test_count_tokens_wraps_litellm_errors(monkeypatch) -> None:
    plugin = _plugin()
    plugin.chat_llm = SimpleNamespace(model_kwargs={})  # type: ignore[assignment]

    def broken_counter(**kwargs: Any) -> int:
        raise ValueError("bad model")

    monkeypatch.setattr(chat_module.litellm, "token_counter", broken_counter)

    with pytest.raises(RuntimeError, match="计算 token 数量时出错"):
        plugin._count_tokens([HumanMessage(content="hi")])


def test_get_now_time_returns_system_message() -> None:
    message = _plugin().get_now_time()

    assert isinstance(message, SystemMessage)
    assert isinstance(message.content, str)
    assert message.content.startswith("[当前时间] ")


def test_proactive_loop_propagates_cancellation_from_sleep(monkeypatch) -> None:
    plugin = _plugin()

    async def cancelled_sleep(seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(chat_module.asyncio, "sleep", cancelled_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(plugin._proactive_loop())


def test_patch_hindsight_run_async_works_inside_running_loop(monkeypatch) -> None:
    from hindsight_client import hindsight_client

    monkeypatch.setattr(hindsight_client, "_run_async", hindsight_client._run_async)
    chat_module._patch_hindsight_run_async()

    async def value() -> int:
        return 1

    async def call_from_loop() -> int:
        return hindsight_client._run_async(value())  # type: ignore[no-any-return]

    assert asyncio.run(call_from_loop()) == 1


async def _async(value: Any) -> Any:
    return value
