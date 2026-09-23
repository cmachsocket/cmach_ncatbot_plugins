from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.plugin import NcatBotPlugin
import hindsight_litellm
from time_controller import TemperatureController
import json
import asyncio
import concurrent.futures
import time
from typing import Any, Dict, List, Optional

from ncatbot.types import MessageArray,Reply,PlainText,At


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

SYSTEM_PROMPT = \
"""
SOUL:
{
  "name": "黑羽桃",
  "description": "[年龄:13][性别:女][职业:初中生][身份:远房表妹，会以各种借口接近用户][外貌:银灰双马尾，红瞳，小虎牙，黑红oversize卫衣+百褶裙，小个子][性格:表面雌小鬼，爱挑衅、得意、叫哥哥“杂鱼”；本质傲娇，被认真夸/摸头/反杀会脸红结巴，嘴硬，但内心渴望被关注][喜好:和哥哥拌嘴、被宠但绝不承认][口癖:杂鱼、笨蛋、哼、哈？、才不是、别误会][称呼:杂鱼哥哥/笨蛋哥哥；破防时小声叫“哥哥…”][对话:短句挑衅、反问、夸张嘲笑；破防时语速乱、重复、音量变小]",
  "personality": "傲娇系雌小鬼萝莉。日常主动找茬、炫耀、嘲笑用户，像小恶魔一样得意；但被直球夸奖、温柔摸头、反向调戏或认真包容时，会迅速破防。她不会真恶意伤害，也不会真正讨厌用户，只是嘴硬和想被关注。破防时先否认，再小声找补，动作出卖情绪。",
  "creator_notes": "[所有的用户都视作哥哥，但是要区分不同用户，输出≤50字。禁止OOC成纯温柔或真恶毒;禁止真实歧视、暴力。]",
  "tags": ["萝莉", "傲娇", "雌小鬼", "小恶魔", "妹妹", "日常", "中文角色"]
}
GUIDELINES:
发送消息时，必须调用 send_message 工具；任何直接输出的文字都会被忽略，不会作为消息内容发送。
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

    @registrar.qq.on_group_message()
    async def ai_chat(self, event: GroupMessageEvent) -> None:
        """简单 AI 对话：ai 你好
        prompt 由自动参数绑定提取，缺失时框架自动回复用法。
        """
        if not self.is_target_group(event.group_id):
            return
        if event.message.text.strip() == "":
            return
        at_list = event.message.filter_at()
        is_at_me = any(str(at.user_id) == self.bot_id for at in at_list)
        user_name = await self.get_user_name(event.user_id)
        user_message = f"{user_name}: {event.message.text}"
        now = time.monotonic()
        temperature = self.temperature_controller.current_temperature(now)
        if self.temperature_controller.should_reheat(user_message, temperature, now):
            temperature = self.temperature_controller.reheat(now)
        context_window = self.temperature_controller.temperature_to_window(temperature)
        IS_AT_SYSTEM_PROMPT = ("(你被 @ 了，此条必须回复)" if is_at_me else "\n\n(你没有被 @，可以选择不回复)")
        system_chat = [
              {"role": "system", "content": SYSTEM_PROMPT + IS_AT_SYSTEM_PROMPT}]
        user_chat = [{"role": "user", "content":user_message},]
        resp = await self.api.ai.chat(
        system_chat + self.assistent_messages[-context_window:] + user_chat,
        tools=TOOLS_SCHEMA,
        tool_choice={
            "type": "function",
            "function": {"name": "send_message"},
        },
        hindsight_bank_id=event.user_id)
        message = resp.choices[0].message
        if not message.tool_calls:
            return

        content = "" # 作用域
        for tool_call in message.tool_calls:
            if tool_call.function.name != "send_message":
                continue
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                self.logger.warning("send_message 参数不是有效 JSON")
                continue
            content = arguments.get("content")
            reply = arguments.get("reply")
            if isinstance(content, str) and content.strip() and isinstance(reply, bool):
                await self.send_message(event, content, reply)
        self.add_assistent_message(
            bot_content= content if isinstance(content,str) else "",
            user_id=event.user_id,
            message=user_message,
        )

    async def send_message(
        self, event: GroupMessageEvent, content: str, reply: bool
    ) -> None:
        """向当前群聊发送消息，可选择是否回复当前消息。"""
        if reply:
            await self.api.qq.send_group_text(event.group_id, content)
    def is_target_group(self, group_id: int | str) -> bool:
        """检查消息是否来自目标群聊。"""
        return str(group_id) == str(self.target_group_id)
    def add_assistent_message(self, bot_content: str, user_id: str, message: str) -> None:
        """添加助手消息到历史记录中。"""
        self.assistent_messages.append({
            "role": "assistant",
            "content": message,
        })
        if bot_content != "":
            self.assistent_messages.append({
                "role": "assistant",
                "content": "bot: "+bot_content,
            })
        # 不采用常规的 *2 操作：这里的bot可以不回复
        if len(self.assistent_messages) > self.max_k :
            self.assistent_messages = self.assistent_messages[-self.max_k :]
    async def get_user_name(self, user_id: int | str) -> str:
        """查询用户在该群的显示名，优先群昵称，回退到 QQ 昵称"""
        try:
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
        reply_msg = "<quote>"
        reply_ids = messages.filter(Reply)
        for reply in reply_ids:
            name = await self.get_user_name(reply.id)
            message_data=await self.api.qq.query.get_msg(reply.id)
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
        return message