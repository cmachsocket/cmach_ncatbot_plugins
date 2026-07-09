"""classhelper — 班级小助手（ncatbot 版）

迁移自 08helper/helper.py：
- 监听源群 → LLM 决策 → 转发 / 加提醒 / 忽略
- 直接用 api.ai.chat(event.message, tools=..., tool_choice="auto")
- 所有规则集中在 SYSTEM_PROMPT
"""

from __future__ import annotations

import asyncio
import datetime
import json
from typing import Any, Dict, List

from ncatbot.plugin import NcatBotPlugin
from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.core.registry.hook import Hook, HookAction, HookContext, HookStage
from ncatbot.types import MessageArray, PlainText


# ============== 配置 ==============
SOURCE_GROUP_ID = 1019963716       # 监听源群
TARGET_GROUP_ID = 1042964394       # 转发目标群
MANAGER_USER_ID = 3077906125       # 异常上报
AI_MODEL = "MiniMax-M3"


# ============== System Prompt（所有规则集中在这）==============
SYSTEM_PROMPT = """你是班级小助手，负责审核源群消息。

收到用户消息后，判断需要做什么，调用合适的 tool：
- forward_message：消息对班级同学有用（通知/活动/重要信息）
- add_timer_message：消息提到未来某个时间要做某事
- do_nothing：闲聊/无关/广告/两者都不需要

可以一次调用多个 tool。提醒时间必须晚于当前时间至少 1 分钟。

当前时间：{now}"""


# ============== Tools Schema（暴露给 LLM 的工具定义）==============
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "forward_message",
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
        if gid is None:
            return HookAction.SKIP
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            return HookAction.SKIP
        if gid_int == self.group_from:
            return HookAction.CONTINUE
        return HookAction.SKIP


# ============== Plugin ==============

class HelperPlugin(NcatBotPlugin):

    name = "classhelper"
    version = "0.6.0"
    author = "cmach_socket"
    description = "班级小助手：function calling 决策 forward/reminder/nothing"

    async def on_load(self) -> None:
        global SOURCE_GROUP_ID, TARGET_GROUP_ID, MANAGER_USER_ID, AI_MODEL
        self.init_defaults({"SOURCE_GROUP_ID": 1019963716,
                            "TARGET_GROUP_ID": 1042964394,
                            "MANAGER_USER_ID": 3077906125,
                            "AI_MODEL": "MiniMax-M3"
                           })
        SOURCE_GROUP_ID = self.get_config("SOURCE_GROUP_ID")
        TARGET_GROUP_ID = self.get_config("TARGET_GROUP_ID")
        MANAGER_USER_ID = self.get_config("MANAGER_USER_ID")
        AI_MODEL = self.get_config("AI_MODEL")
        self.logger.info(f"{self.name} 已加载")

    async def on_close(self) -> None:
        self.logger.info(f"{self.name} 已卸载")

    # ---------- 辅助 ----------

    async def _notify_manager(self, text: str) -> None:
        """异常时给管理员发私聊；自身失败静默吞掉。"""
        try:
            await self.api.qq.post_private_msg(
                user_id=MANAGER_USER_ID,
                text=text,
            )
        except Exception:
            self.logger.exception("上报 manager 失败")

    # ---------- 群消息入口 ----------
  

    @registrar.on_group_message()
    @group_filter_hook(group_from=SOURCE_GROUP_ID)
    async def on_group_message(self, event: GroupMessageEvent) -> None:

        # 纯图片 / 表情 / CQ 码等无纯文本的消息，LLM 无可决策依据，跳过
        text = event.raw_message
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

        # 并发派发 tool_calls（LLM 可能一次返回多个）

        async def do_forward() -> None:
            try:
                await self.api.qq.messaging.forward_group_single_msg(
                    group_id=TARGET_GROUP_ID,
                    message_id=event.message_id,
                )
                self.logger.info(f"已转发 message_id={event.message_id}")
            except Exception as e:
                self.logger.exception("转发失败")
                await self._notify_manager(
                    f"转发失败：{e!r}\n原消息：{text[:200]}"
                )

        async def do_add_timer(args: Dict[str, Any]) -> None:
            # 先校验参数（这一步错通常是 LLM 给的字段不对）
            try:
                t = datetime.datetime(
                    args["year"], args["month"], args["day"],
                    args["hour"], args["minute"],
                )
            except (KeyError, TypeError, ValueError) as e:
                self.logger.warning(f"add_timer_message 参数无效：{args!r} ({e!r})")
                return

            content = args.get("content", "").strip()
            if not content:
                self.logger.warning(f"add_timer_message content 为空：{args!r}")
                return

            if t <= datetime.datetime.now():
                self.logger.warning(f"忽略过期提醒：{t:%Y-%m-%d %H:%M}")
                return

            # 注册定时任务（用 event.message_id 保证名字唯一）
            async def timer_callback():
                try:
                    await self.api.qq.send_group_text(TARGET_GROUP_ID, content)
                except Exception:
                    self.logger.exception("定时消息发送失败")

            try:
                self.add_scheduled_task(
                    f"timer_task_{event.message_id}",
                    t.strftime("%Y-%m-%d %H:%M:%S"),
                    callback=timer_callback,
                )
                self.logger.info(f"已添加提醒：{t:%Y-%m-%d %H:%M} - {content}")
            except Exception:
                self.logger.exception("注册定时任务失败")

        coros = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            name = tc.function.name
            if name == "forward_message":
                coros.append(do_forward())
            elif name == "add_timer_message":
                coros.append(do_add_timer(args))
            # do_nothing / 未知工具：忽略

        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
