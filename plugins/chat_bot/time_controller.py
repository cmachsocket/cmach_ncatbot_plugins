import os
import pickle
import math
import time

from pathlib import Path
from typing import Union

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

# 上下文窗口按 token 计，不再按消息条数。
# token_base 是基准值，实际预算 = token_base × 倍率。
DEFAULT_TOKEN_BASE = 4096
# 倍率区间：温度低于 0.5 全部饱和到 0.5，高于 1.5 饱和到 1.5
DEFAULT_MULTIPLIER_MIN = 0.5
DEFAULT_MULTIPLIER_MAX = 1.5

class TemperatureController:
    """按真实时间控制温度、上下文 token 预算和重加热。"""

    def __init__(
        self,
        token_base: int = DEFAULT_TOKEN_BASE,
        multiplier_min: float = DEFAULT_MULTIPLIER_MIN,
        multiplier_max: float = DEFAULT_MULTIPLIER_MAX,
        temperature_initial: float = 1.0,
        temperature_max: float = 1.5,
        temperature_min: float = 0.05,
        temperature_tau: float = 1800.0,
        reheat_multiplier: float = 4.0,
        reheat_cooldown: float = 60.0,
        state_path: str | Path = "data/chat_bot_temperature.pkl",
    ) -> None:
        # token_base 是基准预算，实际预算 = token_base × 倍率
        self.token_base = int(token_base)
        self.multiplier_min = float(multiplier_min)
        self.multiplier_max = float(multiplier_max)
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

    def temperature_to_multiplier(self, temperature: float) -> float:
        """把温度换算成预算倍率。

        倍率 = clamp(温度, 0.5, 1.5)。温度低于 0.5 的部分全部饱和到 0.5 ——
        温度本身能降到 0.05，但那样预算只有基准的 5%，等于没有上下文；
        真正需要表达的是「冷下来就少看点历史」，0.5 倍（2048 token）已经足够。
        """
        return max(self.multiplier_min, min(self.multiplier_max, temperature))

    def temperature_to_token_budget(self, temperature: float) -> int:
        """基准 token 预算 × 温度倍率。

        context_budget（4096）是基准值，预算 = 基准 × clamp(温度, 0.5, 1.5)，
        所以实际范围是 2048 ~ 6144。温度高 -> 预算大，温度低 -> 预算小。
        """
        multiplier = self.temperature_to_multiplier(temperature)
        return max(1, int(self.token_base * multiplier))

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
