import os
import pickle
import math
import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import litellm
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
)

# 上下文窗口按 token 计，不再按消息条数
DEFAULT_TOKEN_MAX = 4096
DEFAULT_TOKEN_MIN = 256

# 图片按 OpenAI 视觉默认开销计入预算，避免低估导致请求超长。
# litellm 默认只给图片算 93 token，偏高估成 258 更安全。
# 直接写成字面量而不是 **dict 展开：展开的话 pyright 无法确定参数类型。
IMAGE_TOKEN_COUNT_OPT = True

# 历史里用的是 LangChain 消息对象，统计 token 时要转回 OpenAI dict 格式
MessageLike = Union[AIMessage, HumanMessage, SystemMessage, Dict[str, Any]]

# 让 select_context_by_tokens 原样保留传入的消息类型
_M = TypeVar("_M", bound=MessageLike)

# 逐条消息的 token 计数缓存。assistent_messages 里的消息是追加且不可变的，
# 所以同一批消息在多轮里会被反复统计，缓存能省掉大量重复分词。
_TOKEN_CACHE: Dict[tuple, int] = {}
_TOKEN_CACHE_MAX = 2048


def _to_openai_dict(message: Union[BaseMessage, Dict[str, Any]]) -> Dict[str, Any]:
    """把 BaseMessage / dict 统一转成 OpenAI messages 元素。

    langchain 的 type（system/human/ai）和 OpenAI 的 role（system/user/assistant）
    命名不一样，litellm 只认后者。
    """
    if isinstance(message, BaseMessage):
        converted = message_to_dict(message)
        # message_to_dict 返回 {"type": ..., "data": {...}}，
        # 拆出 data 里的 role/content 再把 type 映射成 role
        data = converted.get("data") or {}
        role = data.get("role")
        if role is None:
            # langchain type -> openai role
            role = {
                "system": "system",
                "human": "user",
                "ai": "assistant",
            }.get(converted.get("type", ""), "user")
        return {"role": role, "content": data.get("content", "")}
    return message


def _cache_key(message: Union[BaseMessage, Dict[str, Any]]) -> tuple:
    normalized = _to_openai_dict(message)
    content = normalized.get("content", "")
    if not isinstance(content, str):
        # 多模态 content（list[dict]）用 repr 兜底
        content = repr(content)
    return (normalized.get("role", ""), content)


def count_tools_tokens(tools: Sequence[Any], tool_name: str) -> int:
    """统计工具定义 + tool_choice 占用的 token 数。

    工具 schema 每轮都要发，和 system prompt 一样是固定开销，
    所以要计入上下文预算。tools 传 langchain 的 BaseTool 列表。
    """
    from litellm.types.utils import ChatCompletionToolParam
    from langchain_core.utils.function_calling import convert_to_openai_tool

    try:
        # convert_to_openai_tool 声明返回 dict[str, Any]，这里显式重建成
        # TypedDict，让 pyright 能校验结构
        converted: List[ChatCompletionToolParam] = [
            ChatCompletionToolParam(**convert_to_openai_tool(t)) for t in tools
        ]
        return int(
            litellm.token_counter(
                messages=[{"role": "system", "content": ""}],
                tools=converted,
                tool_choice={
                    "type": "function",
                    "function": {"name": tool_name},
                },
                use_default_image_token_count=IMAGE_TOKEN_COUNT_OPT,
            )
        )
    except Exception:
        # 统计失败不应该影响正常对话
        return 0


def count_message_tokens(message: Union[BaseMessage, Dict[str, Any]]) -> int:
    """统计单条消息的 token 数（含 role/格式开销）。

    message 可以是任意 LangChain BaseMessage 子类，也可以是 OpenAI dict。
    带缓存；multimodal content（图片）会按默认视觉开销计入。
    """
    key = _cache_key(message)
    cached = _TOKEN_CACHE.get(key)
    if cached is not None:
        return cached
    tokens = int(
        litellm.token_counter(
            messages=[_to_openai_dict(message)],
            use_default_image_token_count=IMAGE_TOKEN_COUNT_OPT,
        )
    )
    if len(_TOKEN_CACHE) >= _TOKEN_CACHE_MAX:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[key] = tokens
    return tokens


def count_tokens(content: Any) -> int:
    """统计任意内容（str / 消息 / 消息列表）的 token 数。

    通用入口，接受任何 BaseMessage 子类；count_message_tokens 才是窄类型。
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return int(litellm.token_counter(text=content))
    if isinstance(content, (BaseMessage, dict)):
        return count_message_tokens(content)
    if isinstance(content, Sequence):
        return sum(
            count_message_tokens(m) for m in content if isinstance(m, (BaseMessage, dict))
        )
    return int(litellm.token_counter(text=str(content)))


class TemperatureController:
    """按真实时间控制温度、上下文 token 预算和重加热。"""

    def __init__(
        self,
        token_max: int = DEFAULT_TOKEN_MAX,
        token_min: int = DEFAULT_TOKEN_MIN,
        temperature_initial: float = 1.0,
        temperature_max: float = 1.5,
        temperature_min: float = 0.05,
        temperature_tau: float = 1800.0,
        reheat_multiplier: float = 4.0,
        reheat_cooldown: float = 60.0,
        state_path: str | Path = "data/chat_bot_temperature.pkl",
    ) -> None:
        self.token_max = int(token_max)
        self.token_min = min(int(token_min), self.token_max)
        self.temperature_initial = temperature_initial
        self.temperature_max = temperature_max
        self.temperature_min = temperature_min
        self.temperature_tau = temperature_tau
        self.reheat_multiplier = reheat_multiplier
        self.reheat_cooldown = reheat_cooldown
        self.state_path = Path(state_path)
        self.temperature_base = temperature_initial
        self.temperature_base_time = time.monotonic()
        self.last_reheat_time = 0.0
        self._load_state()

    def _load_state(self) -> None:
        try:
            with self.state_path.open("rb") as state_file:
                state = pickle.load(state_file)
            saved_at = float(state["saved_at"])
            elapsed = max(0.0, time.time() - saved_at)
            self.temperature_base = max(
                self.temperature_min,
                min(self.temperature_max, float(state["temperature_base"])),
            )
            self.temperature_base_time = time.monotonic() - elapsed
            last_reheat_at = state.get("last_reheat_at")
            if last_reheat_at is not None:
                self.last_reheat_time = time.monotonic() - max(
                    0.0, time.time() - float(last_reheat_at)
                )
        except (
            FileNotFoundError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            OSError,
            pickle.PickleError,
        ):
            self._save_state()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_suffix(
            self.state_path.suffix + ".tmp"
        )
        state = {
            "saved_at": time.time(),
            "temperature_base": self.temperature_base,
            "last_reheat_at": (
                time.time() - (time.monotonic() - self.last_reheat_time)
                if self.last_reheat_time
                else None
            ),
        }
        with temporary_path.open("wb") as state_file:
            pickle.dump(state, state_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, self.state_path)

    def current_temperature(self, now: float | None = None) -> float:
        if now is None:
            now = time.monotonic()
        elapsed = max(0.0, now - self.temperature_base_time)
        return self.temperature_min + (
            self.temperature_base - self.temperature_min
        ) * math.exp(-elapsed / self.temperature_tau)

    def temperature_to_token_budget(self, temperature: float) -> int:
        """把温度映射成一个 token 预算（token_min .. token_max）。

        温度高 -> 预算大，温度低 -> 预算小。取代旧的『消息条数』窗口。
        """
        temperature = max(self.temperature_min, min(self.temperature_max, temperature))
        position = (
            math.log(temperature) - math.log(self.temperature_min)
        ) / (
            math.log(self.temperature_initial) - math.log(self.temperature_min)
        )
        position = max(0.0, min(1.0, position))
        budget = self.token_min + position * (self.token_max - self.token_min)
        return min(self.token_max, max(self.token_min, int(budget)))

    def select_context_by_tokens(
        self,
        messages: Sequence[_M],
        budget: int,
        reserve: int = 0,
    ) -> List[_M]:
        """从 messages 尾部往前取，取到 token 预算用满为止。

        参数
        ----
        messages: 历史消息列表（LangChain BaseMessage 或 OpenAI dict），按时间正序。
        budget:   分配给历史消息的 token 预算。
        reserve:  为 system prompt / tools / 当前用户消息预留的 token。
                  这些不计入本函数，但会从可用空间里扣掉。

        返回
        ----
        连续的一段尾部消息；保证越靠近当前轮次的消息越不会被丢掉。
        """
        if not messages:
            return []
        remaining = int(budget) - int(reserve)
        if remaining <= 0:
            return []

        selected: List[_M] = []
        used = 0
        for message in reversed(messages):
            cost = count_message_tokens(message)
            if used + cost > remaining:
                # 预算用完就停：保持尾部连续，不做稀疏采样，
                # 否则上下文会出现前后不连贯的对话片段。
                break
            selected.append(message)
            used += cost
        selected.reverse()
        return selected

    def count_total_tokens(self, messages: Sequence[MessageLike]) -> int:
        """统计一整组消息的 token 总数。"""
        return sum(
            count_message_tokens(m)
            for m in messages
            if isinstance(m, (BaseMessage, dict))
        )
        """统计一整组消息的 token 总数。"""
        return sum(
            count_message_tokens(m)
            for m in messages
            if isinstance(m, (BaseMessage, dict))
        )

    def reheat(self, now: float | None = None) -> float:
        if now is None:
            now = time.monotonic()
        temperature = self.current_temperature(now)
        self.temperature_base = min(
            self.temperature_max, self.reheat_multiplier * temperature
        )
        self.temperature_base_time = now
        self.last_reheat_time = now
        self._save_state()
        return self.temperature_base

    def should_reheat(
        self, message: str, temperature: float, now: float | None = None
    ) -> bool:
        if now is None:
            now = time.monotonic()
        if now - self.last_reheat_time < self.reheat_cooldown:
            return False
        old_memory_markers = ("之前", "刚才", "上次", "记得", "忘了", "忘记")
        complex_query = len(message) >= 80 or any(
            marker in message for marker in old_memory_markers
        )
        return complex_query and temperature <= self.temperature_max * 0.65
