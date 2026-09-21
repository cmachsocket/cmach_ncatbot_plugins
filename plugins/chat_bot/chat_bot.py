"""
ai/01_hello_world — AI 适配器基础用法

演示功能:
  - api.ai.chat(): Chat Completion（字符串 & messages 两种调用方式）
  - api.ai.embeddings(): 文本向量化
  - api.ai.image_generation(): 图像生成
  - 模型参数覆盖
  - 命令参数自动绑定（推荐用法）

前置配置:
  adapters:
    - type: ai
      config:
        api_key: "sk-xxxx"             # 或通过环境变量 OPENAI_API_KEY
        completion_model: "gpt-4"
        embedding_model: "text-embedding-3-small"
        image_model: "dall-e-3"
"""

from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.plugin import NcatBotPlugin


class AIHelloWorldPlugin(NcatBotPlugin):
    """AI 适配器基础用法示例"""

    name = "hello_world_ai"

    @registrar.qq.on_private_command("ai")
    async def ai_chat(self, event: GroupMessageEvent, prompt: str):
        """简单 AI 对话：ai 你好
        prompt 由自动参数绑定提取，缺失时框架自动回复用法。
        """
        resp = await self.api.ai.chat(prompt)
        answer = resp.choices[0].message.content
        await event.reply(answer)

    @registrar.qq.on_private_command("ai-multi")
    async def ai_multi_turn(self, event: GroupMessageEvent, prompt: str = "你好"):
        """多轮对话示例：ai-multi 你的问题"""
        resp = await self.api.ai.chat([
            {"role": "system", "content": "你是一个简洁的助手，回答不超过50字"},
            {"role": "user", "content": prompt},
        ])
        await event.reply(resp.choices[0].message.content)
