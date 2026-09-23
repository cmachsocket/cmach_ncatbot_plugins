"""persona.py — 社交动力学引擎

把机器人建模成一个耦合的多维连续状态系统：

    ┌──────────────┐
    │ attention A  │ ←── 消息密度、@ 触发
    └──────┬───────┘
           │
    ┌──────────────┐
    │   mood E     │ ←── 消息情感 + 记忆
    └──────┬───────┘
           │
    ┌──────────────┐
    │  energy N    │ ←── 发言消耗 + 时间回血 + 群里越吵耗越快
    └──────┬───────┘
           │
    ┌──────────────┐
    │ affection F  │ ←── 跟某人的互动累积 + 长期衰减
    └──────┬───────┘
           │
           ▼
        发言决策 σ(线性组合)

学术血统：
    - Affective Computing (Picard, 1997)         valence-arousal 二维情绪
    - BDI Model (Bratman, 1987)                   belief-desire-intention
    - Multiple Resource Theory (Wickens)          注意力/能量资源竞争
    - Cognitive Load Theory (Sweller, 1988)       fatigue 概念
    - Rescorla-Wagner                             affection 增量更新
    - Russell Circumplex Model (1980)             mood/arousal 正交
    - Emotional Contagion (Hatfield, 1993)        情绪传染
    - Ebbinghaus 遗忘曲线                          时间常数衰减
    - Sacks Turn-Taking (1974)                    抢话抑制

参数为工程经验值，不是学术结论。
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# 单群一组状态
# ============================================================================

@dataclass
class GroupDynamics:
    """一个群内机器人持有的全部动力学状态"""

    # ---- 全局连续状态 ----
    attention: float = 0.5         # ∈ [0, 1]   当前注意力集中度
    energy: float = 1.0            # ∈ [0, 1]   能量/精神
    mood: float = 0.0              # ∈ [-1, 1]  valence：-1 痛苦, +1 愉悦
    arousal: float = 0.2           # ∈ [0, 1]   arousal：0 困, 1 激动

    # ---- per-user 状态 ----
    affection: defaultdict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: 0.4)
    )                              # ∈ [0, 1]   对某人的好感
    trust: defaultdict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: 0.5)
    )                              # ∈ [0, 1]   对某人的信任
    fatigue: dict[str, float] = field(
        default_factory=dict
    )                              # ∈ [0, 1]   跟某人互动的疲劳

    # ---- 时间戳 ----
    last_spoke_global: float = 0.0
    last_spoke_to: dict[str, float] = field(default_factory=dict)
    last_tick: float = field(default_factory=time.monotonic)

    # ---- 对某 user 的近期消息时间（用于疲劳判断）----
    recency_to_user: defaultdict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=20))
    )

    # ---- 短期记忆 ----
    said_recently: deque[int] = field(default_factory=lambda: deque(maxlen=20))

    # ---- 群节奏 ----
    pulse: deque[float] = field(default_factory=lambda: deque(maxlen=400))


# ============================================================================
# 引擎
# ============================================================================

class SocialDynamics:
    """社交动力学主引擎：每个 group_id 维护一组 GroupDynamics"""

    # ---- 时间常数（秒）----
    TAU_ATTENTION = 30.0          # 注意力跟踪群密度的快慢
    TAU_ENERGY_REST = 600.0       # 能量自然回血 ~ 10min
    TAU_MOOD = 90.0                # 情绪回归基线 ~ 1.5min
    TAU_AROUSAL = 45.0             # 唤起回归 ~ 45s
    TAU_AFFECTION_LONG = 3600.0    # 亲密度回归 ~ 1h
    TAU_TRUST_LONG = 7200.0        # 信任回归 ~ 2h
    TAU_FATIGUE = 600.0            # 疲劳半衰 ~ 10min

    # ---- 决策常数（logistic 偏置项）----
    DECIDE_BIAS = -2.0             # 基础不想回的偏置，越大越沉默
    DECIDE_FORCED_AT_ME = True     # 被 @ 时是否强制回

    def __init__(self, seed: Optional[int] = None):
        self.groups: dict[str, GroupDynamics] = defaultdict(GroupDynamics)
        self._rng = random.Random(seed)
        self._bot_id: str = ""   # 外部调用 set_bot_id() 注入
        self._said_text: dict[str, list[str]] = {}  # 复读禁忌：原话环形 buffer

    def set_bot_id(self, bot_id: str) -> None:
        """让引擎知道自己的 QQ id，用来判断消息里的 @"""
        self._bot_id = bot_id

    # ------------------------------------------------------------------
    # 内部：时间步进
    # ------------------------------------------------------------------
    def tick(self, group_id: str, now: Optional[float] = None) -> None:
        """每帧调一次：让所有状态自然演化"""
        now = now if now is not None else time.monotonic()
        g = self.groups[group_id]
        dt = min(60.0, now - g.last_tick)
        g.last_tick = now

        # 1. 注意力跟踪群密度
        density = self._density(g, now, window=20)
        target_a = 0.3 + 0.6 * min(1.0, density / 5.0)
        g.attention += (target_a - g.attention) * (1 - math.exp(-dt / self.TAU_ATTENTION))

        # 2. 能量：热闹消耗、安静回血
        k_busy = 0.015
        k_rest = 0.004
        dN = (-k_busy * density + k_rest * (1 - min(1.0, density / 8))) * dt
        g.energy = max(0.0, min(1.0, g.energy + dN))

        # 3. 情绪回归
        g.mood += (0.0 - g.mood) * (1 - math.exp(-dt / self.TAU_MOOD))

        # 4. 唤起回归
        g.arousal += (0.2 - g.arousal) * (1 - math.exp(-dt / self.TAU_AROUSAL))

        # 5. 亲密度 / 信任 慢速回归到中性
        for uid in list(g.affection.keys()):
            g.affection[uid] += (0.3 - g.affection[uid]) * (
                1 - math.exp(-dt / self.TAU_AFFECTION_LONG)
            )
            g.affection[uid] = max(0.0, min(1.0, g.affection[uid]))
        for uid in list(g.trust.keys()):
            g.trust[uid] += (0.5 - g.trust[uid]) * (
                1 - math.exp(-dt / self.TAU_TRUST_LONG)
            )
            g.trust[uid] = max(0.0, min(1.0, g.trust[uid]))

        # 6. 疲劳衰减
        for uid in list(g.fatigue.keys()):
            g.fatigue[uid] *= math.exp(-dt / self.TAU_FATIGUE)
            if g.fatigue[uid] < 0.005:
                g.fatigue.pop(uid, None)

        # 7. 砍过期 pulse（> 10min）
        cutoff = now - 600
        while g.pulse and g.pulse[0] < cutoff:
            g.pulse.popleft()

    # ------------------------------------------------------------------
    # 外部事件：消息进来
    # ------------------------------------------------------------------
    def on_incoming_message(
        self,
        group_id: str,
        user_id: str,
        text: str,
        has_image: bool,
        is_at_me: bool,
        now: Optional[float] = None,
    ) -> None:
        """每收到一条群消息都调一次"""
        now = now if now is not None else time.monotonic()
        g = self.groups[group_id]
        self.tick(group_id, now)
        g.pulse.append(now)

        # 唤起：@、图片、文字长度、情绪词
        bump = 0.0
        if is_at_me:
            bump += 0.6
        if has_image:
            bump += 0.2
        if text:
            bump += min(0.3, len(text) / 200.0)
        for kw, v in {
            "哈哈": 0.15, "笑": 0.08, "！": 0.05, "?": 0.05, "？": 0.05,
            "哭": 0.10, "😭": 0.10, "火": 0.10, "气": 0.08,
        }.items():
            if kw in text:
                bump += v
        g.arousal = min(1.0, g.arousal + bump)

        # 情绪传染：信任的人影响放大
        delta = self._sentiment_delta(text)
        trust = g.trust.get(user_id, 0.5)
        delta *= 0.6 + 0.8 * trust
        g.mood = max(-1.0, min(1.0, g.mood * 0.7 + delta))

        # 亲密度累积
        if is_at_me or has_image:
            g.affection[user_id] = min(1.0, g.affection[user_id] + 0.04)
            g.trust[user_id] = min(1.0, g.trust[user_id] + 0.02)
        else:
            g.affection[user_id] = min(1.0, g.affection[user_id] + 0.003)

        # 疲劳：连续互动（看 recency 队列而不是 last_spoke_to，
        #     避免『首次互动也算疲劳』的 bug）
        g.recency_to_user[user_id].append(now)
        recent_60s = sum(1 for t in g.recency_to_user[user_id] if now - t <= 60)
        # 60s 内累计 4 条以上 → 疲劳上升
        if recent_60s >= 4:
            g.fatigue[user_id] = min(1.0, g.fatigue.get(user_id, 0.0) + 0.10 * recent_60s)
        else:
            g.fatigue[user_id] = g.fatigue.get(user_id, 0.0) * 0.9

        # 被 @ → 强制注意力
        if is_at_me:
            g.attention = max(g.attention, 0.9)

    # ------------------------------------------------------------------
    # 外部事件：自己说话了
    # ------------------------------------------------------------------
    def on_self_spoke(
        self,
        group_id: str,
        user_id: str,
        content: str,
        now: Optional[float] = None,
    ) -> None:
        now = now if now is not None else time.monotonic()
        g = self.groups[group_id]
        self.tick(group_id, now)

        g.last_spoke_global = now
        g.last_spoke_to[user_id] = now

        # 复读禁忌：保存原话
        self.record_said_text(group_id, content)

        cost = 0.04 + 0.005 * len(content)
        g.energy = max(0.0, g.energy - cost)
        g.arousal = max(0.0, g.arousal - 0.05)
        g.mood = min(1.0, g.mood + 0.04)  # 说完带点满足感

        h = hash(self._normalize_for_repeat(content))
        g.said_recently.append(h)

    # ------------------------------------------------------------------
    # 外部事件：自己选择沉默
    # ------------------------------------------------------------------
    def on_self_skipped(self, group_id: str, user_id: str) -> None:
        g = self.groups[group_id]
        g.attention *= 0.95
        g.fatigue[user_id] = g.fatigue.get(user_id, 0.0) * 0.95

    # ------------------------------------------------------------------
    # 决策：要不要回这条消息
    # ------------------------------------------------------------------
    def decide_reply(
        self,
        group_id: str,
        user_id: str,
        text: str,
        has_image: bool,
        is_at_me: bool,
        now: Optional[float] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """返回 (是否回, 决策信息)"""
        now = now if now is not None else time.monotonic()
        self.tick(group_id, now)
        g = self.groups[group_id]

        if is_at_me and self.DECIDE_FORCED_AT_ME:
            return True, {
                "forced": "at_me",
                "z": float("inf"),
                "p": 1.0,
                "attention": g.attention,
                "energy": g.energy,
                "mood": g.mood,
                "arousal": g.arousal,
                "affection": g.affection.get(user_id, 0.4),
                "trust": g.trust.get(user_id, 0.5),
                "fatigue": g.fatigue.get(user_id, 0.0),
                "trigger": 0.0,
            }

        affection = g.affection.get(user_id, 0.4)
        fatigue_v = g.fatigue.get(user_id, 0.0)
        trust = g.trust.get(user_id, 0.5)

        # 触发强度
        trigger = 0.0
        if has_image:
            trigger += 0.3
        for kw in ("桃", "黑羽", "哈？", "杂鱼", "笨蛋"):
            if kw in text:
                trigger += 0.4
        for kw in ("?", "？", "笑", "哈", "啊"):
            if kw in text:
                trigger += 0.15
        if "!" in text or "！" in text:
            trigger += 0.1
        if 8 <= len(text) <= 30:
            trigger += 0.1

        # 抢话抑制
        if g.pulse:
            density_10s = sum(1 for t in g.pulse if now - t <= 10)
            if density_10s >= 5 and now - g.last_spoke_global > 30:
                trigger -= 0.4

        # @ 其他人不插嘴
        if text.count("@") >= 1 and not is_at_me:
            trigger -= 0.3

        # logistic 综合
        z = (
            self.DECIDE_BIAS
            + 1.6 * g.attention
            + 1.4 * g.energy
            + 1.8 * affection
            + 2.2 * trigger
            + 0.4 * trust
            - 2.0 * fatigue_v
            + 0.6 * (g.mood + 1) / 2.0
            + 0.4 * g.arousal
        )
        try:
            p = 1 / (1 + math.exp(-z))
        except OverflowError:
            p = 0.0 if z < 0 else 1.0

        # 硬阈值：能量枯竭
        if g.energy < 0.05:
            return False, {"p": p, "blocked": "no_energy", "z": z}

        # 深夜 + 能量低 → 抑制
        hour = time.localtime(now).tm_hour
        if 0 <= hour < 7 and g.energy < 0.4:
            p *= 0.3

        should = self._rng.random() < p
        info = {
            "p": p,
            "z": z,
            "attention": g.attention,
            "energy": g.energy,
            "mood": g.mood,
            "affection": affection,
            "trust": trust,
            "fatigue": fatigue_v,
            "trigger": trigger,
            "arousal": g.arousal,
        }
        return should, info

    # ------------------------------------------------------------------
    # 决策：主动想说话的欲望
    # ------------------------------------------------------------------
    def proactive_impulse(self, group_id: str, now: Optional[float] = None) -> float:
        """返回 [0, 1]，外部按心跳采样"""
        now = now if now is not None else time.monotonic()
        self.tick(group_id, now)
        g = self.groups[group_id]

        density = self._density(g, now, window=120)
        quiet = 1 - min(1.0, density / 3.0)

        z = (
            -3.0
            + 2.2 * quiet
            + 1.5 * g.energy
            + 1.2 * (g.mood + 1) / 2.0
            + 0.5 * (1 - g.attention)
        )
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 0.0 if z < 0 else 1.0

    # ------------------------------------------------------------------
    # 决策：打字延迟
    # ------------------------------------------------------------------
    def typing_latency(
        self,
        group_id: str,
        content_len: int,
        now: Optional[float] = None,
    ) -> float:
        """think + type 两段，含偶尔『打了删』"""
        now = now if now is not None else time.monotonic()
        g = self.groups[group_id]

        think = self._rng.uniform(0.8, 2.5) * (1.0 + (1.0 - g.attention))

        cps = (
            2.5
            + 1.5 * g.energy
            + 0.8 * (g.mood + 1) / 2.0
            + 0.6 * g.arousal
        )
        if g.fatigue:
            avg_fatigue = sum(g.fatigue.values()) / len(g.fatigue)
            cps -= 1.0 * avg_fatigue
        cps = max(1.0, cps)

        type_t = content_len / cps

        hour = time.localtime(now).tm_hour
        if 0 <= hour < 7:
            type_t *= 1.5

        if self._rng.random() < 0.12:
            type_t += self._rng.uniform(2.0, 5.0)

        return min(think + type_t, 15.0)

    # ------------------------------------------------------------------
    # 复读抑制
    # ------------------------------------------------------------------
    def is_repeating(self, group_id: str, content: str) -> bool:
        h = hash(self._normalize_for_repeat(content))
        return h in self.groups[group_id].said_recently

    def said_samples(self, group_id: str, n: int = 4) -> list[str]:
        """返回最近说过的若干条原话（去掉重复 hash 的不可逆信息），供 prompt 注入禁忌。"""
        g = self.groups[group_id]
        # said_recently 存的是 hash，无法逆推原话
        # 因此这里依赖外部在 on_self_spoke 时把原话存到一个环形 buffer。
        return list(self._said_text.get(group_id, []))[-n:]

    def record_said_text(self, group_id: str, text: str) -> None:
        """外部在 self.on_self_spoke 后调一下，把原话存进来用于复读禁忌。"""
        buf = self._said_text.setdefault(group_id, [])
        buf.append(text)
        if len(buf) > 16:
            del buf[:-16]

    # ------------------------------------------------------------------
    # 调试
    # ------------------------------------------------------------------
    def dump(self, group_id: str) -> dict[str, object]:
        g = self.groups[group_id]
        return {
            "attention": round(g.attention, 3),
            "energy": round(g.energy, 3),
            "mood": round(g.mood, 3),
            "arousal": round(g.arousal, 3),
            "affection_top": sorted(
                g.affection.items(), key=lambda x: -x[1]
            )[:5],
            "fatigue_top": sorted(
                g.fatigue.items(), key=lambda x: -x[1]
            )[:5],
            "trust_top": sorted(
                g.trust.items(), key=lambda x: -x[1]
            )[:5],
            "since_last_spoke": (
                round(time.monotonic() - g.last_spoke_global, 1)
                if g.last_spoke_global else None
            ),
            "recent_density": len(g.pulse),
        }

    # ------------------------------------------------------------------
    # 内部 helpers
    # ------------------------------------------------------------------
    def _density(self, g: GroupDynamics, now: float, window: float) -> float:
        """群消息密度（条/分钟）"""
        c = sum(1 for t in g.pulse if now - t <= window)
        return c / max(1.0, window / 60.0)

    def _sentiment_delta(self, text: str) -> float:
        score = 0.0
        pos = {
            "哈哈": 0.3, "笑死": 0.25, "好耶": 0.3, "可爱": 0.15,
            "棒": 0.15, "强": 0.15, "喜欢": 0.2, "爱": 0.2,
            "嘿嘿": 0.2, "嘻": 0.1, ":)": 0.1, "😊": 0.15,
        }
        neg = {
            "讨厌": -0.3, "无语": -0.25, "滚": -0.4, "烦": -0.25,
            "气": -0.2, "哭": -0.2, "😭": -0.3, "艹": -0.3,
            "妈的": -0.4, "操": -0.3, "傻": -0.15, "笨": -0.1,
        }
        for k, v in pos.items():
            if k in text:
                score += v
        for k, v in neg.items():
            if k in text:
                score += v
        score += min(0.2, text.count("!") * 0.05)
        score += min(0.2, text.count("！") * 0.05)
        score -= min(0.2, text.count("?") * 0.04)
        score -= min(0.2, text.count("？") * 0.04)
        return max(-1.0, min(1.0, score))

    def _normalize_for_repeat(self, s: str) -> str:
        return s.strip().rstrip("！。？～~，,.")