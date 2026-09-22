from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.plugin import NcatBotPlugin
import hindsight_litellm
from hindsight_client import Hindsight
from hindsight_litellm.callbacks import HindsightCallback
import litellm
from datetime import datetime
import json
from typing import Any, Dict, List

SYSTEM_PROMPT = \
"""
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
                        "description": "是否回复当前用户的消息；true 为回复，false 为直接发送到群聊",
                    },
                },
                "required": ["content", "reply"],
            },
        },
    },
]

class AIHelloWorldPlugin(NcatBotPlugin):
    """AI 适配器基础用法示例"""

    name = "hello_world_ai"
    hindsight = None
    hindsight_port = 7071
    target_group_id = 1093424135
    bot_id = None
    async def on_load(self) -> None:
        self.hindsight_port = self.get_config("HINDSIGHT_PORT", 7071)
        self.target_group_id = self.get_config("TARGET_GROUP_ID", 1093424135)
        self.hindsight = Hindsight(base_url=f"http://localhost:{self.hindsight_port}") 
        hindsight_litellm.configure(hindsight_api_url=f"http://localhost:{self.hindsight_port}")
        hindsight_litellm.set_defaults(bank_id="default-bank")  # 设置一个默认值
        hindsight_litellm.enable()
        hindsight_callback = HindsightCallback()
        litellm.callbacks = [hindsight_callback]
        info = await self.api.qq.query.get_login_info()
        self.bot_id = info.user_id
    @registrar.qq.on_group_message()
    async def ai_memory(self, event: GroupMessageEvent) -> None:
        if not self.is_target_group(event.group_id):
            return
        if self.bot_id == event.user_id:
            self.logger.info("忽略自己发送的消息")
            return
        text = event.message.text
        id = event.user_id
        timestamp = datetime.fromtimestamp(event.time)
        #拼接所有at
        ats = ""
        for at in event.message.filter_at():
            ats += f" @ {at.user_id} , "
        if(self.hindsight == None):
            self.logger.error("Hindsight 未初始化")
            return
        self.hindsight.retain(
            bank_id = id,
            content = text,
            timestamp = timestamp,
            metadata = {"at" : ats}
        )

    @registrar.qq.on_group_message()
    async def ai_chat(self, event: GroupMessageEvent) -> None:
        """简单 AI 对话：ai 你好
        prompt 由自动参数绑定提取，缺失时框架自动回复用法。
        """
        if not self.is_target_group(event.group_id):
            return
        at_list = event.message.filter_at()
        is_at_me = any(str(at.user_id) == self.bot_id for at in at_list)
        IS_AT_SYSTEM_PROMPT = ("(你被 @ 了，此条必须回复)" if is_at_me else "\n\n(你没有被 @，可以选择不回复)")
        resp = await self.api.ai.chat([
              {"role": "system", "content": SYSTEM_PROMPT + IS_AT_SYSTEM_PROMPT},
              {"role": "user", "content": event.message.text},
              ],
        tools=TOOLS_SCHEMA,
        tool_choice={
            "type": "function",
            "function": {"name": "send_message"},
        },
        hindsight_bank_id=event.user_id)
        message = resp.choices[0].message
        if not message.tool_calls:
            return

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

    async def send_message(
        self, event: GroupMessageEvent, content: str, reply: bool
    ) -> None:
        """向当前群聊发送消息，可选择是否回复当前消息。"""
        if reply:
            await event.reply(content)
        else:
            await self.api.qq.send_group_text(event.group_id, content)
    def is_target_group(self, group_id) -> bool:
        """检查消息是否来自目标群聊。"""
        return str(group_id) == str(self.target_group_id)