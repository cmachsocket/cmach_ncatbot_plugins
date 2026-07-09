"""classhelper — 班级小助手（ncatbot 版）

迁移自 08helper/helper.py：
- 监听源群 → LLM 决策 → 转发 / 加提醒 / 忽略
- 直接用 api.ai.chat(event.message, tools=..., tool_choice="auto")
- 所有规则集中在 SYSTEM_PROMPT
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

from ncatbot.plugin import NcatBotPlugin
from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.core.registry.hook import Hook, HookAction, HookContext, HookStage
from ncatbot.types import MessageArray, PlainText


# ============== 配置 ==============
SOURCE_GROUP_ID = 1019963716       # 监听源群
TARGET_GROUP_ID = 1042964394       # 转发目标群
MANAGER_USER_ID = 3077906125       # 异常上报
AI_MODEL = "MiniMax-M2.7"
PROCESSED_CACHE_MAX = 2000


# ============== System Prompt（所有规则集中在这）==============
SYSTEM_PROMPT = """你是班级小助手，负责审核源群消息。

收到用户消息后，判断需要做什么，调用合适的 tool：
- need_forward_message：消息对班级同学有用（通知/活动/重要信息）
- add_timer_message：消息提到未来某个时间要做某事
- do_nothing：闲聊/无关/两者都不需要

可以一次调用多个 tool。提醒时间必须晚于当前时间至少 1 分钟。

当前时间：{now}
"""


# ============== Tools Schema（暴露给 LLM 的工具定义）==============
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "need_forward_message",
            "description": "把这条消息转发到目标班级群",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_timer_message",
            "description": "添加一条定时提醒，到时间会在目标群自动发送",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "提醒内容"},
                    "year":   {"type": "integer"},
                    "month":  {"type": "integer"},
                    "day":    {"type": "integer"},
                    "hour":   {"type": "integer"},
                    "minute": {"type": "integer"},
                },
                "required": ["content", "year", "month", "day", "hour", "minute"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "do_nothing",
            "description": "不转发也不设置提醒",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ============== 工具函数 ==============

def _extract_text(msg_array) -> str:
    """从 event.message (MessageArray) 提取纯文本"""
    return "".join(
        seg.text for seg in msg_array if hasattr(seg, "text")
    ).strip()


# ============== 群过滤 hook ==============

def group_filter_hook(group_from: int) -> Hook:
    return GroupFilterHook(group_from=group_from)


class GroupFilterHook(Hook):
    stage = HookStage.BEFORE_CALL
    priority = 50

    def __init__(self, group_from: int):
        self.group_from = group_from

    async def execute(self, ctx: HookContext) -> HookAction:
        gid = getattr(ctx.event.data, "group_id", None)
        if gid is not None and int(gid) == self.group_from:
            return HookAction.CONTINUE
        return HookAction.SKIP


# ============== Plugin ==============

class HelperPlugin(NcatBotPlugin):

    name = "classhelper"
    version = "0.5.0"
    author = "cmach_socket"
    description = "班级小助手：function calling 决策 forward/reminder/nothing"

    def _init_(self) -> None:
        pass

    async def on_load(self) -> None:
        self.logger.info(f"{self.name} 已加载")

    async def on_close(self) -> None:
        self.logger.info(f"{self.name} 已卸载")

    # ---------- 群消息入口 ----------

    async def timer_task(self,content : str) -> None:
        await self.api.qq.send_group_text(TARGET_GROUP_ID,content)

    @registrar.on_group_message()
    @group_filter_hook(group_from=SOURCE_GROUP_ID)
    async def on_group_message(self, event: GroupMessageEvent) -> None:
        # 去重
        if event.message_id in self.processed_ids:
            return
        self.processed_ids.add(event.message_id)
        if len(self.processed_ids) > PROCESSED_CACHE_MAX:
            self.processed_ids = set(
                list(self.processed_ids)[-PROCESSED_CACHE_MAX // 2 :]
            )

        # 从 event.message 提取纯文本（用于跳过空消息 + 异常上报）
        text = _extract_text(event.message)
        if not text:
            return

        # SYSTEM_PROMPT 作为 PlainText 段塞到 event.message 前面
        # chat() 内部会把所有 PlainText 拼成一个 text content
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combined_message = MessageArray(
            [PlainText(text=SYSTEM_PROMPT.format(now=now_str) + "\n\n")]
            + list(event.message)
        )

        # 一次 chat() 调用
        try:
            resp = await self.api.ai.chat(
                combined_message,
                tools=TOOLS_SCHEMA,    # **kwargs → litellm
                tool_choice="auto",
                model=AI_MODEL,
            )
        except Exception:
            self.logger.exception("LLM 调用失败")
            return

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return

        # 派发 tool_calls（在 on_group_message 作用域里，直接用 event）
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            name = tc.function.name
            if name == "need_forward_message":
                # 直接转发原消息（保留图片/表情/at）
                try:
                    await self.api.qq.messaging.forward_group_single_msg(
                        group_id=TARGET_GROUP_ID,
                        message_id=event.message_id,
                    )
                    self.logger.info(f"已转发 message_id={event.message_id}")
                except Exception:
                    self.logger.exception("转发失败")
                    try:
                        await self.api.qq.post_private_msg(
                            user_id=MANAGER_USER_ID,
                            text=f"转发失败\n原消息：{text[:200]}",
                        )
                    except Exception:
                        pass

            elif name == "add_timer_message":
                try:
                    t = datetime.datetime(
                        args["year"], args["month"], args["day"],
                        args["hour"], args["minute"],
                    )
                    content = args["content"]
                    if t <= datetime.datetime.now():
                        self.logger.warning(f"忽略过期提醒：{t}")
                        continue
                    self.add_scheduled_task(timer_task, ,conditions=[])
                    self.logger.info(f"已添加提醒：{t:%Y-%m-%d %H:%M} - {content}")
                except Exception:
                    self.logger.exception("添加提醒失败")

            # do_nothing / 未知工具：忽略
