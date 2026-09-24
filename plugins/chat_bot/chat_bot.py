from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.plugin import NcatBotPlugin
import hindsight_litellm
from regex import F
from .time_controller import TemperatureController
from .persona import SocialDynamics
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

    async def on_load(self) -> None:
        self.assistent_messages = []
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
        # 社交动力学：决定『要不要回』
        self.persona = SocialDynamics()
        self.persona.set_bot_id(str(self.bot_id))
        # 统计：用于评估拟人度
        self._stats = {"prompt": 0, "replied": 0, "skipped": 0}
        # 主动说话心跳任务（重载时先取消旧的，避免协程堆积）
        if self._bg_task is not None and not self._bg_task.done():
            self._bg_task.cancel()
        self._bg_task = asyncio.create_task(self._proactive_loop())

    @registrar.qq.on_group_message()
    async def ai_chat(self, event: GroupMessageEvent) -> None:
        """AI 对话：动力学决定『要不要回』，LLM 决定『说什么』"""
        if not self.is_target_group(event.group_id):
            return
        raw_text = event.message.text.strip()
        if not raw_text:
            return

        gid = str(event.group_id)
        uid = str(event.user_id)
        now = time.monotonic()
        self._stats["prompt"] += 1

        at_list = event.message.filter_at()
        is_at_me = any(str(at.user_id) == self.bot_id for at in at_list)
        has_image = bool(event.message.filter(Image))

        # 1) 把消息喂给动力学
        self.persona.on_incoming_message(
            group_id=gid,
            user_id=uid,
            text=raw_text,
            has_image=has_image,
            is_at_me=is_at_me,
            now=now,
        )

        # 2) 求解『该不该回』
        should_reply, decision_info = self.persona.decide_reply(
            group_id=gid,
            user_id=uid,
            text=raw_text,
            has_image=has_image,
            is_at_me=is_at_me,
            now=now,
        )

        if not should_reply:
            self._stats["skipped"] += 1
            self.persona.on_self_skipped(gid, uid)
            LOG.info(
                "persona 丢弃消息 gid=%s uid=%s text=%r decision=%s",
                gid, uid, raw_text, decision_info,
            )
            # 沉默也写一轮对话进历史（用 uid 作占位名，跳过昂贵的
            # get_user_name/resolve_message），让下一轮上下文保持完整。
            self.add_context(bot_content="", message=f"{uid}: {raw_text}")
            return

        # 3) 准备 prompt
        user_name = await self.get_user_name(event.user_id)
        user_message = await self.resolve_message(event.message)
        if not user_message.strip():
            return
        prefixed = f"{user_name}: {user_message}"

        temperature = self.temperature_controller.current_temperature(now)
        if self.temperature_controller.should_reheat(prefixed, temperature, now):
            temperature = self.temperature_controller.reheat(now)
        context_window = self.temperature_controller.temperature_to_window(temperature)

        decision_ctx = self._build_decision_ctx(decision_info, is_at_me, user_name)

        system_chat = [
            {"role": "system", "content": SYSTEM_PROMPT + decision_ctx}
        ]
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
            LOG.info(
                "模型选择不回复：没有调用 send_message 工具")
            self._stats["skipped"] += 1
            self.persona.on_self_skipped(gid, uid)
            self.add_context(bot_content="", message=prefixed)
            return

        # 4) 处理工具调用
        last_content = ""
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                self.logger.warning("send_message 参数不是有效 JSON")
                continue
            if arguments.get("reply") is False:  # 严格只匹配 False，不匹配 None
                # 模型选择不回复
                LOG.info(
                    "模型选择不回复：Reply=False"
                )
                self._stats["skipped"] += 1
                self.persona.on_self_skipped(gid, uid)
                self.add_context(bot_content="", message=prefixed)
                return
                
            raw_content = arguments.get("content") or ""
            
            if not raw_content.strip():
                continue
            
            # 打字延迟
            delay = self.persona.typing_latency(gid, len(raw_content), now)
            if delay > 0:
                await asyncio.sleep(delay)

            await self._send_to_group(event, raw_content)

            # 记录动力学状态
            self.persona.on_self_spoke(gid, uid, raw_content, now)
            self._stats["replied"] += 1
            last_content = raw_content

        self.add_context(
            bot_content=last_content,
            message=prefixed,
        )

    async def _proactive_loop(self) -> None:
        """主动说话心跳：不依赖用户消息"""
        while True:
            # 20~90 分钟一醒
            await asyncio.sleep(random.uniform(1200, 5400))
            try:
                gid = str(self.target_group_id)
                now = time.monotonic()
                impulse = self.persona.proactive_impulse(gid, now)
                if random.random() > impulse:
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
                    if self.persona.is_repeating(gid, content):
                        continue
                    delay = self.persona.typing_latency(gid, len(content), now)
                    await asyncio.sleep(delay)
                    await self.api.qq.send_group_text(self.target_group_id, content)
                    self.persona.on_self_spoke(gid, "self", content, now)
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

    def _build_decision_ctx(
        self,
        decision_info: dict[str, Any],
        is_at_me: bool,
        user_name: str,
    ) -> str:
        """把动力学的数值状态翻译成自然语言描述，给 LLM 看。

        设计原则：LLM 不应该看到浮点数——只看到『定性』的状态描述，
        这样不会让它过度模仿数值（比如看到 mood=+0.3 就硬塞颜文字）。
        数值仍由 persona 内部决策使用。
        """
        energy = float(decision_info.get("energy", 0.5))
        mood = float(decision_info.get("mood", 0.0))
        affection = float(decision_info.get("affection", 0.4))
        fatigue = float(decision_info.get("fatigue", 0.0))

        # 能量 → 疲倦度
        if energy > 0.7:
            energy_desc = "你现在精神很好"
        elif energy > 0.4:
            energy_desc = "你有点累"
        else:
            energy_desc = "你现在很疲倦"

        # 情绪 → 心情
        if mood > 0.3:
            mood_desc = "心情不错"
        elif mood > 0.1:
            mood_desc = "有点小开心"
        elif mood < -0.3:
            mood_desc = "心情不太好"
        elif mood < -0.1:
            mood_desc = "有点低落"
        else:
            mood_desc = "心情一般"

        # 亲密度 → 关系
        if affection > 0.7:
            rel_desc = f"你跟 {user_name} 很有好感"
        elif affection > 0.4:
            rel_desc = f"你跟 {user_name} 关系不错"
        else:
            rel_desc = f"你跟 {user_name} 试图保持友好"

        # 疲劳 → 语气长度提示
        if fatigue > 0.6:
            fatigue_desc = "刚才聊得有点多，简短点回就行"
        else:
            fatigue_desc = ""

        # 必须回 vs 可不回
        must_desc = "这条必须回" if is_at_me else "这条可回可不回"

        parts = [energy_desc + "，", mood_desc + "。", rel_desc + "。"]
        if fatigue_desc:
            parts.append(fatigue_desc + "。")
        parts.append(f"{must_desc}。")

        return "\n[内心状态] " + " ".join(parts) + "\n"
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