import os
import pickle
import math
import time

from pathlib import Path
    
class TemperatureController:
    """按真实时间控制温度、上下文窗口和重加热。"""

    def __init__(
        self,
        window_max: int = 40,
        window_min: int = 4,
        temperature_initial: float = 1.0,
        temperature_max: float = 1.5,
        temperature_min: float = 0.05,
        temperature_tau: float = 1800.0,
        reheat_multiplier: float = 4.0,
        reheat_cooldown: float = 60.0,
        state_path: str | Path = "data/chat_bot_temperature.pkl",
    ) -> None:
        self.window_max = window_max
        self.window_min = window_min
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

    def temperature_to_window(self, temperature: float) -> int:
        temperature = max(self.temperature_min, min(self.temperature_max, temperature))
        position = (
            math.log(temperature) - math.log(self.temperature_min)
        ) / (
            math.log(self.temperature_initial) - math.log(self.temperature_min)
        )
        position = max(0.0, min(1.0, position))
        window = self.window_min + position * (self.window_max - self.window_min)
        return min(self.window_max, max(self.window_min, int(window)))

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
