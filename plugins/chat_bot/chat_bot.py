from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.plugin import NcatBotPlugin
import hindsight_litellm
from .time_controller import TemperatureController
import json
import asyncio
import concurrent.futures
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from ncatbot.utils import get_log

from ncatbot.types import MessageArray,Reply,PlainText,At,Image

LOG = get_log("AIPlugin")

def _patch_hindsight_run_async() -> None:
    """Monkey patch hindsight_client._run_async to be safe inside a running event loop.

    hindsight_client.Hindsight.recall/reflect 是同步包装，它们内部调用 _run_async(coro)
    在已运行的 event loop 里再 run_until_complete 会抛 RuntimeError，协程泄漏并触发
    RuntimeWarning。本函数在 on_load 时替换 _run_async：当前 loop 未运行则维持原行为，
    已在运行则把协程丢到独立线程的新 loop 里跑。
    """
    import hindsight_client.hindsight_client as _hc

    def _safe_run_async(coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            def _runner():
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_runner).result()
        return loop.run_until_complete(coro)

    _hc._run_async = _safe_run_async

SOUL_PROMPT = (Path(__file__).resolve().parent / "SOUL.txt").read_text(
    encoding="utf-8"
)

SYSTEM_PROMPT = SOUL_PROMPT + \
"""

GUIDELINES:
你在一个群聊里面，你收到的消息不一定是发给你的，你需要根据上下文和语境来判断是否回复。
发送消息时，**必须**调用 send_message 工具；任何直接输出的文字都会被**完全丢弃**，不会作为消息内容发送!!!
你可以自行决定是否调用 send_message 工具，或者直接忽略用户消息。不需要每一条都回复，像人一样选择性回复就行。
"""

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "向当前群聊发送一条消息",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要发送给用户的消息内容",
                    },
                    "reply": {
                        "type": "boolean",
                        "description": "是否回复当前用户的消息；true 为回复，false 为不回复",
                        "default": False,
                    },
                },
                "required": ["content", "reply"],
            },
        },
    },
]




class AIPlugin(NcatBotPlugin):
    """AI 适配器基础用法示例"""
    name = "hello_world_ai"
    hindsight: Any = None
    hindsight_port: int = 7071
    target_group_id: int = 1093424135
    bot_id: Optional[str] = None
    assistent_messages: List[Dict[str, str]] = []  # 用于存储上下文
    max_k: int = 60  # 历史消息缓存上限
    dynamic_k: int = 40  # 温度动态调节的基准上下文窗口
    _bg_task: Optional[asyncio.Task[None]] = None
    # 上次收到真实用户消息的 monotonic 时间，用于主动说话沉默计时
    _last_user_msg_at: float = 0.0

    # ---- 主动说话参数 ----
    # 群沉默超过这个秒数才开始计算主动说话概率
    PROACTIVE_SILENCE_THRESHOLD_S: float = 3.0 * 3600.0
    # 沉默刚跨过阈值时的初始概率
    PROACTIVE_P_START: float = 0.01
    # 每多沉默一小时，概率增加多少（线性）
    PROACTIVE_P_SLOPE_PER_HOUR: float = 0.05
    # 心跳间隔（小时）
    PROACTIVE_TICK_HOURS: float = 1.0

    async def on_load(self) -> None:
        self.assistent_messages = []
        self._last_user_msg_at = time.monotonic()
        self.temperature_controller = TemperatureController(window_max=self.dynamic_k)
        self.hindsight_port = self.get_config("HINDSIGHT_PORT", 7071)
        self.target_group_id = self.get_config("TARGET_GROUP_ID", 1093424135)
        #self.hindsight = Hindsight(base_url=f"http://localhost:{self.hindsight_port}")
        _patch_hindsight_run_async()
        hindsight_litellm.configure(hindsight_api_url=f"http://localhost:{self.hindsight_port}")
        hindsight_litellm.set_defaults(bank_id="default-bank")  # 设置一个默认值
        hindsight_litellm.enable()
        info = await self.api.qq.query.get_login_info()
        self.bot_id = info.user_id
        # 主动说话心跳任务（重载时先取消旧的，避免协程堆积）
        if self._bg_task is not None and not self._bg_task.done():
            self._bg_task.cancel()
        self._bg_task = asyncio.create_task(self._proactive_loop())

    @registrar.qq.on_group_message()
    async def ai_chat(self, event: GroupMessageEvent) -> None:
        """AI 对话：LLM 通过 send_message 工具自行决定『要不要回/说什么』"""
        if not self.is_target_group(event.group_id):
            return
        raw_text = event.message.text.strip()
        if not raw_text:
            return

        uid = str(event.user_id)
        now = time.monotonic()
        # 记录真实用户消息时间，供主动说话心跳判断『沉默多久』
        self._last_user_msg_at = now

        at_list = event.message.filter_at()
        is_at_me = any(str(at.user_id) == self.bot_id for at in at_list)
        has_image = bool(event.message.filter(Image))

        user_name = await self.get_user_name(event.user_id)
        user_message = await self.resolve_message(event.message)
        if not user_message.strip():
            return
        prefixed = f"{user_name}: {user_message}"

        temperature = self.temperature_controller.current_temperature(now)
        if self.temperature_controller.should_reheat(prefixed, temperature, now):
            temperature = self.temperature_controller.reheat(now)
        context_window = self.temperature_controller.temperature_to_window(temperature)

        system_chat = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_chat = [{"role": "user", "content": prefixed}]

        resp = await self.api.ai.chat(
            system_chat + self.assistent_messages[-context_window:] + user_chat,
            tools=TOOLS_SCHEMA,
            tool_choice={
                "type": "function",
                "function": {"name": "send_message"},
            },
            hindsight_bank_id=uid,
        )
        message = resp.choices[0].message
        LOG.info(f"AI content:{resp.choices[0].message.content}")
        if not message.tool_calls:
            # 模型自己选择沉默
            LOG.info("模型选择不回复：没有调用 send_message 工具")
            self.add_context(bot_content="", message=prefixed)
            return

        last_content = ""
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                self.logger.warning("send_message 参数不是有效 JSON")
                continue
            if arguments.get("reply") is False:  # 严格只匹配 False，不匹配 None
                LOG.info("模型选择不回复：Reply=False")
                self.add_context(bot_content="", message=prefixed)
                return

            raw_content = arguments.get("content") or ""

            if not raw_content.strip():
                continue

            await self._send_to_group(event, raw_content)
            last_content = raw_content

        self.add_context(
            bot_content=last_content,
            message=prefixed,
        )

    async def _proactive_loop(self) -> None:
        """主动说话心跳：不依赖用户消息。

        策略：
          - 群沉默 < 3h：不主动说
          - 沉默 >= 3h 后，每小时判定一次
          - 概率 p(silence_h) = 0.01 + 0.05 * silence_h（线性），上限 1.0
          - 平均第一次主动说话 ~ 沉默跨过阈值后约 5h（总 ~8h）
        """
        tick_s = self.PROACTIVE_TICK_HOURS * 3600.0
        while True:
            await asyncio.sleep(tick_s)
            try:
                now = time.monotonic()
                silence_s = now - self._last_user_msg_at
                silence_h = (silence_s - self.PROACTIVE_SILENCE_THRESHOLD_S) / 3600.0
                if silence_h <= 0:
                    # 还没沉默够 3 小时，不主动开话题
                    continue
                p = min(
                    1.0,
                    self.PROACTIVE_P_START
                    + self.PROACTIVE_P_SLOPE_PER_HOUR * silence_h,
                )
                if random.random() >= p:
                    continue
                if not self.assistent_messages:
                    continue
                # 让模型自创一句
                prompt_msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT
                        + "\n[模式] 主动发起话题。随便说点啥——想起的事、对群友的吐槽、自嘲。"}
                ] + self.assistent_messages[-8:]
                resp = await self.api.ai.chat(
                    prompt_msgs,
                    tools=TOOLS_SCHEMA,
                    tool_choice={"type": "function", "function": {"name": "send_message"}},
                    hindsight_bank_id="self",
                )
                msg = resp.choices[0].message
                LOG.info(f"AI content:{resp.choices[0].message.content}")
                if not msg.tool_calls:
                    continue
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        continue
                    content = args.get("content") or ""
                    if not content.strip():
                        continue
                    await self.api.qq.send_group_text(self.target_group_id, content)
                    # 主动说话也算 bot 自己说过话，不算用户消息
                    self.add_context(
                        bot_content=content,
                        message=content,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning("proactive loop error: %s", e)

    async def _send_to_group(
        self, event: GroupMessageEvent, content: str
    ) -> None:
        """向当前群聊发送一条消息。"""
        if not content.strip():
            return
        await self.api.qq.send_group_text(event.group_id, content)
    def is_target_group(self, group_id: int | str) -> bool:
        """检查消息是否来自目标群聊。"""
        return str(group_id) == str(self.target_group_id)

    def add_context(self, bot_content: str, message: str) -> None:
        """记录一轮对话到上下文历史。

        message 是这一轮用户的发言（已是 "{name}: {text}" 形式）。
        bot_content 是机器人的回复；空字符串表示机器人选择沉默，
        此时插入统一占位符 "...（已读未回）"，让对话轮次保持完整，
        避免 LLM 下一轮重复尝试回复同一条消息。
        """
        self.assistent_messages.append({"role": "user", "content": message})
        bot_text = bot_content if bot_content else "...（已读未回）"
        self.assistent_messages.append({"role": "assistant", "content": bot_text})
        # 滑动窗口
        if len(self.assistent_messages) > self.max_k:
            self.assistent_messages = self.assistent_messages[-self.max_k:]
    async def get_user_name(self, user_id: int | str) -> str:
        """查询用户在该群的显示名，优先群昵称，回退到 QQ 昵称"""
        try:
            if(user_id == "all"):
                return "全体成员"
            member_info = await self.api.qq.query.get_group_member_info(
                self.target_group_id, user_id
            )
        except Exception as e:
            self.logger.warning("获取群成员信息失败 user_id=%s: %s", user_id, e)
            return "未知用户"

        if not member_info:
            return "未知用户"

        # card 可能为 "" 或 None，nickname 通常是 QQ 昵称
        card: str = getattr(member_info, "card", "") or ""
        nickname: str = getattr(member_info, "nickname", "") or ""
        return card or nickname or "未知用户"
    async def resolve_message(self, messages: MessageArray) -> str:
        """解析消息内容"""
        message = ""
        reply_msg = "<quote>\n"
        reply_ids = messages.filter(Reply)
        for reply in reply_ids:
            message_data=await self.api.qq.query.get_msg(reply.id)
            if message_data is None:
                continue
            if message_data.sender is not None and message_data.sender.user_id is not None:
                name = await self.get_user_name(message_data.sender.user_id)
            else:
                name = "未知用户"
            # get_msg 返回 MessageData，消息段列表位于其 message 属性中。
            reply_msg += f"{name}：{MessageArray.from_list(message_data.message or []).text}\n"
        reply_msg += "</quote>\n"
        if(len(reply_ids) > 0):
            message += reply_msg
        for msg in messages:
            if isinstance(msg, PlainText):
                message += msg.text
            elif isinstance(msg, At):
                name = await self.get_user_name(msg.user_id)
                message += f"@{name} "
        LOG.info(f"resolve_message: {message}")
        return message