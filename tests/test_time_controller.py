"""TemperatureController 的温度、预算与持久化测试。"""

import math
import time

import pytest

from plugins.chat_bot.time_controller import (
    DEFAULT_MULTIPLIER_MAX,
    DEFAULT_MULTIPLIER_MIN,
    DEFAULT_TOKEN_BASE,
    TemperatureController,
)


def _controller(tmp_path, **kwargs) -> TemperatureController:
    return TemperatureController(
        state_path=tmp_path / "temperature.pkl",
        **kwargs,
    )


def test_initial_temperature_equals_initial_base(tmp_path) -> None:
    controller = _controller(tmp_path, temperature_initial=1.2)

    assert controller.current_temperature(
        controller.temperature_base_time
    ) == pytest.approx(1.2)


def test_temperature_decays_exponentially_toward_minimum(tmp_path) -> None:
    controller = _controller(
        tmp_path,
        temperature_initial=1.5,
        temperature_min=0.05,
        temperature_tau=100.0,
    )
    start = controller.temperature_base_time

    after_tau = controller.current_temperature(start + 100.0)
    expected = 0.05 + (1.5 - 0.05) * math.exp(-1.0)

    assert after_tau == pytest.approx(expected)
    assert after_tau > controller.temperature_min
    assert controller.current_temperature(start + 10_000.0) == pytest.approx(
        controller.temperature_min
    )


def test_temperature_multiplier_is_clamped(tmp_path) -> None:
    controller = _controller(tmp_path)

    assert controller.temperature_to_multiplier(0.05) == DEFAULT_MULTIPLIER_MIN
    assert controller.temperature_to_multiplier(0.8) == pytest.approx(0.8)
    assert controller.temperature_to_multiplier(5.0) == DEFAULT_MULTIPLIER_MAX


def test_token_budget_uses_base_times_multiplier(tmp_path) -> None:
    controller = _controller(tmp_path, token_base=4096)

    assert controller.temperature_to_token_budget(1.0) == 4096
    assert controller.temperature_to_token_budget(0.01) == 2048
    assert controller.temperature_to_token_budget(10.0) == 6144


def test_token_budget_never_below_one(tmp_path) -> None:
    controller = _controller(tmp_path, token_base=1, multiplier_min=0.1)

    assert controller.temperature_to_token_budget(0.0) == 1


def test_default_token_base_is_used(tmp_path) -> None:
    controller = _controller(tmp_path)

    assert controller.token_base == DEFAULT_TOKEN_BASE


def test_reheat_multiplies_current_temperature_and_caps_at_max(tmp_path) -> None:
    controller = _controller(
        tmp_path,
        temperature_initial=0.4,
        temperature_max=1.5,
        reheat_multiplier=4.0,
        temperature_tau=100.0,
    )
    now = controller.temperature_base_time
    current = controller.current_temperature(now)

    reheated = controller.reheat(now)

    assert reheated == pytest.approx(min(1.5, 4.0 * current))
    assert controller.current_temperature(now) == pytest.approx(reheated)
    assert controller.last_reheat_time == now


def test_reheat_is_capped_at_temperature_max(tmp_path) -> None:
    controller = _controller(
        tmp_path,
        temperature_initial=1.4,
        temperature_max=1.5,
        reheat_multiplier=4.0,
    )
    now = controller.temperature_base_time

    assert controller.reheat(now) == pytest.approx(1.5)


def test_should_reheat_requires_complex_query(tmp_path) -> None:
    controller = _controller(tmp_path, temperature_initial=0.8)

    assert controller.should_reheat("你好", 0.8, now=100.0) is False
    assert controller.should_reheat("x" * 80, 0.8, now=100.0) is True
    assert controller.should_reheat("你还记得之前说的事吗", 0.8, now=100.0) is True


def test_should_reheat_respects_cooldown(tmp_path) -> None:
    controller = _controller(tmp_path, temperature_initial=0.8)
    controller.last_reheat_time = 1000.0

    assert controller.should_reheat("x" * 80, 0.8, now=1030.0) is False
    assert controller.should_reheat("x" * 80, 0.8, now=1060.0) is True


def test_should_reheat_skips_hot_temperature(tmp_path) -> None:
    controller = _controller(tmp_path, temperature_max=1.5)

    assert controller.should_reheat("x" * 80, 1.5, now=100.0) is False
    assert controller.should_reheat("x" * 80, 0.9, now=100.0) is True


def test_state_is_persisted_and_reloaded(tmp_path) -> None:
    state_path = tmp_path / "temperature.pkl"
    controller = TemperatureController(
        state_path=state_path,
        temperature_initial=1.0,
        temperature_max=1.5,
    )
    now = controller.temperature_base_time
    controller.reheat(now)
    saved_base = controller.temperature_base

    restored = TemperatureController(state_path=state_path)

    assert state_path.exists()
    assert restored.temperature_base == pytest.approx(saved_base)
    assert restored.last_reheat_time > 0


def test_elapsed_time_since_save_is_applied_on_load(tmp_path) -> None:
    state_path = tmp_path / "temperature.pkl"
    controller = TemperatureController(state_path=state_path, temperature_initial=1.0)
    controller._save_state()
    saved_at = time.time() - 3600
    controller.temperature_base = 1.2
    controller._save_state()
    state = {
        "saved_at": saved_at,
        "temperature_base": 1.2,
        "last_reheat_at": None,
    }
    import pickle

    with state_path.open("wb") as fh:
        pickle.dump(state, fh)

    restored = TemperatureController(state_path=state_path)

    # 保存时间在一小时前，加载后应认为温度已经衰减了一小时
    elapsed = time.monotonic() - restored.temperature_base_time
    assert elapsed == pytest.approx(3600, abs=5)
    assert restored.current_temperature() < 1.2


def test_corrupt_state_file_falls_back_to_default(tmp_path) -> None:
    state_path = tmp_path / "temperature.pkl"
    state_path.write_bytes(b"not a pickle")

    controller = TemperatureController(state_path=state_path, temperature_initial=1.0)

    assert controller.temperature_base == 1.0
    assert state_path.exists()
