from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent
from ncatbot.plugin import NcatBotPlugin
import hindsight_litellm
from .time_controller import (
    TemperatureController,
)
import asyncio
import concurrent.futures
import random
import time
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Optional, Sequence, Union
from typing_extensions import NotRequired
from uuid import uuid4
from ncatbot.utils import get_log

from ncatbot.types import MessageArray,Reply,PlainText,At,Image

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import InjectedState
from langchain_litellm import ChatLiteLLM
from ncatbot.utils import get_config_manager
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
    trim_messages,
)
from langchain_core.tools import tool

LOG = get_log("AIPlugin")


class ChatAgentState(AgentState):
    """在 agent 默认 state 上挂三个运行期参数。

    send_message_tool 需要拿到 plugin / event / user_msg，但这些既不该暴露给
    LLM（InjectedState 会自动从 tool schema 里剔除），也不适合走
    config["configurable"] —— InjectedToolArg 本身没有注入来源，只有
    InjectedState / InjectedStore / ToolRuntime 才会被 langgraph 真正填充。
    所以放进 state，由 InjectedState("字段名") 取。

    messages 不覆盖：AgentState 里已经是 Annotated[..., add_messages]，
    继承即可，langgraph 会按消息 id 合并，ToolMessage 也能正常并入历史。
    """

    plugin: NotRequired[Any]
    event: NotRequired[Any]
    user_msg: NotRequired[str]


# middleware 的泛型参数填 ChatAgentState，这样它们和 state_schema 一致，
# 否则 ToolCallLimitMiddleware 会去要自己的 ToolCallLimitState。
global_limiter = ToolCallLimitMiddleware(run_limit=1)


@tool("send_message",return_direct=True)
async def send_message_tool(
    content: str,
    reply: bool = False,
    # 下面三个参数 LLM 看不到，由 langgraph 从 agent state 注入
    plugin: Annotated[Any, InjectedState("plugin")] = None,
    event: Annotated[Any, InjectedState("event")] = None,
    user_msg: Annotated[str, InjectedState("user_msg")] = "",
) -> str:
    """向当前群聊发送一条消息。reply=true 表示回复当前用户；false 表示选择不回复。"""
    if plugin is None:
        # 没注入到就说明 state 没带上，后面只用 plugin，直接早退更清楚
        LOG.error("send_message_tool 未拿到 plugin，state 可能缺少 plugin 字段")
        return "error: missing plugin"

    if not reply or not content.strip():
        LOG.info("模型选择不回复")
        plugin.add_context(bot_content="", message=user_msg)
        return "skipped"

    if event is not None:
        await plugin._send_to_group(event, content)
    else:
        # 主动说话场景没有 event
        await plugin.api.qq.send_group_text(plugin.target_group_id, content)

    plugin.add_context(bot_content=content, message=user_msg)
    return "sent"


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
对于一条消息，<quote>...</quote> 里面的内容是引用的消息，通常是你之前发过的消息或者其他人的消息，你需要根据这些引用来判断当前消息的语境。user_name : message 是用户发给你的消息, 不同的user_name代表不同的用户,你需要根据这些消息来判断每个用户的意图和情感。
重要！！！发送消息时，**必须**调用 send_message 工具；任何直接输出的文字都会被**完全丢弃**，不会作为消息内容发送!!!
你可以自行决定是否调用 send_message 工具，或者直接忽略用户消息。不需要每一条都回复，像人一样选择性回复就行。
不要输出markdown、代码块、表格、列表等格式化内容，直接输出纯文本即可。
"""




class AIPlugin(NcatBotPlugin):
    """AI 适配器基础用法示例"""
    name = "hello_world_ai"
    hindsight: Any = None
    hindsight_port: int = 7071
    target_group_id: int = 1093424135
    bot_id: Optional[str] = None
    # 历史里只有 HumanMessage / AIMessage 两种，不含 ToolMessage 等
    assistent_messages: List[Union[HumanMessage, AIMessage]] = []
    max_k: int = 60  # 历史消息条数缓存上限（内存层面，token 由 token_base × 倍率 限制）
    # 上下文 token 预算的基准值。实际预算 = 4096 × clamp(温度, 0.5, 1.5)
    # 即 2048 ~ 6144 之间浮动，由 TemperatureController 计算。
    context_budget: int = 4096
    _bg_task: Optional[asyncio.Task[None]] = None
    # 上次收到真实用户消息的 monotonic 时间，用于主动说话沉默计时
    _last_user_msg_at: float = 0.0
    model : str
    api_key : str
    base_url : str
    chat_llm : Optional[ChatLiteLLM] = None
    agent: Any = None
    # hindsight 记忆库：群消息按 user_id 分库，主动说话固定用 self
    PROACTIVE_BANK_ID: str = "self"
    # 共用同一个 chat_llm，切 bank_id 时要串行，避免并发调用互相污染
    _bank_lock: asyncio.Lock = asyncio.Lock()
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
        self.temperature_controller = TemperatureController(
            token_base=self.get_config("CONTEXT_TOKEN_BUDGET", 65536),
        )
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
        manager = get_config_manager()
        ai_entry = manager.get_adapter_config("ai")
        if ai_entry :
            self.model = ai_entry.config.get("completion_model", "")
            self.api_key = ai_entry.config.get("api_key", "")
            self.base_url = ai_entry.config.get("base_url", "")
            self.chat_llm = ChatLiteLLM(
                model=self.model,
                api_key=self.api_key,
                api_base=self.base_url,
                # hindsight_litellm.enable() 已 patch 了 litellm.acompletion，
                # ChatLiteLLM 底层走的就是它，所以记忆注入/落库自动生效。
                # hindsight_bank_id 从 model_kwargs 进，每次调用前由
                # _set_hindsight_bank 改写，区分不同用户的记忆库。
                model_kwargs={"hindsight_bank_id": self.PROACTIVE_BANK_ID},
            )
        if self.chat_llm:
            # middleware 参数的 type: ignore 是必要的：langchain 1.4.2 把
            # ToolCallLimitMiddleware 的 StateT 硬绑成它自己的
            # ToolCallLimitState，与 AgentState 子类在类型上无法共存
            # （不变量冲突）。运行时 create_agent 只读 state_schema，
            # 不校验 middleware 的 StateT，所以行为是正确的。
            self.agent = create_agent(
                self.chat_llm,
                tools=[send_message_tool],
                middleware=[
                    global_limiter,  # type: ignore[arg-type]
                    # 历史接近预算时让 LLM 压缩成摘要，而不是硬截断丢掉。
                    # keep=('messages', 24) 表示压缩后保留最近 24 条原文。
                    SummarizationMiddleware(
                        self.chat_llm,
                        trigger=("tokens", self.context_budget),
                        keep=("messages", 24),
                        token_counter=self._count_tokens,
                    ),
                ],
                # 带上自定义 state，send_message_tool 靠 InjectedState
                # 从这里取 plugin / event / user_msg
                state_schema=ChatAgentState,
            )
    def get_now_time(self) -> SystemMessage:
        """获取当前时间，返回 SystemMessage 形式，供 LLM 使用"""
        now = time.localtime()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", now)
        return SystemMessage(content=f"[当前时间] {now_str}")

    def _count_tokens(self, messages: Iterable[Any]) -> int:
        """委托 ChatLiteLLM 使用其 LangChain token 计数实现。"""
        if self.chat_llm is None:
            raise RuntimeError("chat_llm 未初始化，请检查 ai 适配器配置")
        return self.chat_llm.get_num_tokens_from_messages(list(messages))

    @registrar.qq.on_group_message()
    async def ai_chat(self, event: GroupMessageEvent) -> None:
        """AI 对话：LLM 通过 send_message 工具自行决定『要不要回/说什么』。

        本轮消息如果是多模态（带图片），把图片作为 image_url part 一起发给 LLM。
        但写入历史 (assistent_messages) 的只有纯文本版本，图片用 [图片] 占位，
        避免历史越攒越大、过期 URL 被反复引用。
        """
        if not self.is_target_group(event.group_id):
            return
        raw_text = event.message.text.strip()
        if not raw_text and not event.message.filter_image():
            return
    
        uid = str(event.user_id)
        if uid == self.bot_id:
            # 机器人自己发的消息不处理，避免死循环
            return
        now = time.monotonic()
        # 记录真实用户消息时间，供主动说话心跳判断『沉默多久』
        self._last_user_msg_at = now

        at_list = event.message.filter_at()
        is_at_me = any(str(at.user_id) == self.bot_id for at in at_list)

        user_name = await self.get_user_name(event.user_id)
        # 同时拿到：① 纯文本（用于历史）② 多模态 parts（用于本轮 LLM 调用）
        user_text, mm_parts = await self.resolve_message_multimodal(event.message)
        if not user_text.strip():
            return
        prefixed_text = f"{user_name}: {user_text}"

        # 本轮要发给 LLM 的 user content：有图片用多模态列表，纯文本用 str
        # HumanMessage.content 声明的是 list[str | dict]，Sequence 协变后兼容
        if mm_parts:
            # 把前置名字加到第一个 text part；后续 At/图片已经填好
            mm_parts = [
                {"type": "text", "text": f"{user_name}: "},
                *mm_parts,
            ]
            user_chat_content: str | Sequence[Dict[str, Any]] = mm_parts
        else:
            user_chat_content = prefixed_text

        temperature = self.temperature_controller.current_temperature(now)
        if self.temperature_controller.should_reheat(
            prefixed_text, temperature, now
        ):
            temperature = self.temperature_controller.reheat(now)
        # 温度 → token 预算，再从历史尾部取能塞进预算的连续片段
        token_budget = self.temperature_controller.temperature_to_token_budget(
            temperature
        )
        if is_at_me:
            is_at_chat = SystemMessage(content="你被 @ 了，必须回复。")
        else:
            is_at_chat = SystemMessage(content="你没有被 @，可以选择不回复。")
        system_chat = SystemMessage(content=SYSTEM_PROMPT)
        user_chat = HumanMessage(content=user_chat_content)

        # system prompt + 工具定义 + 本轮用户消息是固定开销，先扣掉，
        # 剩下的预算才分给历史，否则总长度会顶穿 token 上限
        fixed_tokens = self._count_tokens([system_chat, user_chat])
        history = trim_messages(
            self.assistent_messages,
            max_tokens=max(1, token_budget - fixed_tokens),
            token_counter=self._count_tokens,
            strategy="last",
            start_on="human",
        )
        # self.add_context(
        #     bot_content="",  
        #     message=prefixed_text,
        # ) 以后修改逻辑，send_message 工具里不再 add_context，避免重复 add
        # 每个群成员一个记忆库，和旧的 hindsight_bank_id=uid 行为一致
        LOG.info(f"ai_chat: {event.message_id}")
        await self._run_agent(
            [system_chat, is_at_chat, self.get_now_time(), *history, user_chat],
            bank_id=uid,
            event=event,
            user_msg=prefixed_text,   # 当前这轮用户消息，供 add_context 用
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
                # 让模型自创一句。主动说话只需要一点近期上下文，
                # 用 token 预算而不是固定条数。
                # 主动说话不按温度调节，用基准值（倍率 1.0）即可。
                proactive_system = SystemMessage(
                    content=SYSTEM_PROMPT
                    + "\n[模式] 主动发起话题。不一定与记忆相关，也不一定与当前群聊的最新消息相关。可以是一个问题、一个建议、一个有趣的想法、一个冷知识、一个笑话等。"
                )
                reserve = self._count_tokens([proactive_system])
                proactive_budget = (
                    self.temperature_controller.temperature_to_token_budget(1.0)
                )
                history = trim_messages(
                    self.assistent_messages,
                    max_tokens=max(1, proactive_budget - reserve),
                    token_counter=self._count_tokens,
                    strategy="last",
                    start_on="human",
                )
                # 主动说话的记忆单独存在 self 库里，不跟群成员混。
                # event 传 None，工具里走 target_group_id 分支；
                # user_msg 留空，主动说话没有用户消息。
                await self._run_agent(
                    [proactive_system, self.get_now_time(), *history],
                    bank_id=self.PROACTIVE_BANK_ID,
                    event=None,
                    user_msg="",
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning("proactive loop error: %s", e)

    def _set_hindsight_bank(self, bank_id: str) -> None:
        """把下一次 LLM 调用的 hindsight 记忆库切到 bank_id。

        chat_llm 是全局共用的，hindsight_bank_id 挂在它的 model_kwargs 上，
        每次调用 litellm 时才会被读走。所以这里只改值，真正的生效点在
        下面的 _run_agent 调用（必须紧挨着 ainvoke，中间不能有别的调用）。
        """
        if self.chat_llm is None:
            return
        kwargs = dict(self.chat_llm.model_kwargs or {})
        kwargs["hindsight_bank_id"] = bank_id
        self.chat_llm.model_kwargs = kwargs

    async def _run_agent(
        self,
        messages: Sequence[BaseMessage],
        bank_id: str,
        event: Any = None,
        user_msg: str = "",
    ) -> Any:
        """在指定 hindsight 记忆库下跑一次 agent。

        event / user_msg 放进 state（不是 config["configurable"]），这样
        send_message_tool 能通过 InjectedState 拿到它们。

        切 bank 和 ainvoke 必须成对加锁：bank_id 挂在共享的 chat_llm 上，
        若中途被别的协程改掉，这次调用的记忆就会记到别人账上。
        """
        if self.agent is None:
            raise RuntimeError("agent 未初始化，请检查 ai 适配器配置")
        # convert_to_messages 是 langchain 内置的规范化函数：原样保留
        # BaseMessage，把裸 dict / 元组转成对应消息对象。
        # 下面 messages 上的 type: ignore 同理：AgentState.messages 标注为
        # list[AnyMessage]，但所有 langchain 构造函数（含 convert_to_messages）
        # 都返回 list[BaseMessage]，而 AnyMessage 联合不含 BaseMessage 本身。
        state = ChatAgentState(
            messages=convert_to_messages(messages),  # type: ignore[arg-type]
            plugin=self,
            event=event,
            user_msg=user_msg,
        )
        LOG.info("agent run: bank_id=%s", bank_id)
        async with self._bank_lock:
            self._set_hindsight_bank(bank_id)
            return await self.agent.ainvoke(state, verbose=True)

    async def _send_to_group(
        self, event: GroupMessageEvent, content: str
    ) -> None:
        """向当前群聊发送一条消息。"""
        if not content.strip():
            return
        messages = content.splitlines()
        for msg in messages:
            if not msg.strip():
                continue
            await self.api.qq.send_group_text(event.group_id, msg)
            await asyncio.sleep(1)  # 避免发太快
    def is_target_group(self, group_id: int | str) -> bool:
        """检查消息是否来自目标群聊。"""
        return str(group_id) == str(self.target_group_id)

    def add_context(self, bot_content: str, message: str) -> None:
        """记录一轮对话到上下文历史。

        message 是这一轮用户的发言（已是 "{name}: {text}" 形式）。
        bot_content 是机器人的回复；空字符串表示机器人选择沉默，
        此时插入统一占位符 "...（已读未回）"，让对话轮次保持完整，
        避免 LLM 下一轮重复尝试回复同一条消息。

        历史里存的是 LangChain 消息对象（HumanMessage / AIMessage），
        图片不进历史，只留纯文本。

        追加走 langgraph 的 add_messages reducer 而不是 list.append：
        它会按消息 id 合并，重复调用同一条消息不会产生重复条目。
        """
        bot_text = bot_content if bot_content else "...（已读未回）"
        # 保留 add_messages 的核心语义：按消息 id 合并，重复的 id 覆盖旧值
        # 而不是追加一份。这里自己实现是因为 langgraph 1.4.2 的 Messages
        # 类型别名里只写了 BaseMessage，没写 HumanMessage/AIMessage 子类，
        # 传具体子类列表过不了类型检查（运行时是正常的）。
        merged: Dict[str, Union[HumanMessage, AIMessage]] = {
            m.id: m for m in self.assistent_messages if m.id is not None
        }
        for msg in (
            HumanMessage(content=message, id=f"u-{uuid4().hex}"),
            AIMessage(content=bot_text, id=f"a-{uuid4().hex}"),
        ):
            assert msg.id is not None  # 上面刚显式给了 id
            merged[msg.id] = msg
        history = list(merged.values())
        # 滑动窗口：丢掉最旧的消息
        if len(history) > self.max_k:
            history = history[-self.max_k :]
        self.assistent_messages = history
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

    async def resolve_message_multimodal(
        self, messages: MessageArray
    ) -> tuple[str, Sequence[Dict[str, Any]]]:
        """把 MessageArray 拆成 (纯文本, 多模态 parts)。
        """
        text_buf: list[str] = []

        # 处理 Reply：作为纯文本 quote 块塞到消息开头
        reply_ids = messages.filter(Reply)
        if reply_ids:
            quote_lines = ["<quote>"]
            for reply in reply_ids:
                message_data = await self.api.qq.query.get_msg(reply.id)
                if message_data is None:
                    continue
                if (
                    message_data.sender is not None
                    and message_data.sender.user_id is not None
                ):
                    name = await self.get_user_name(message_data.sender.user_id)
                else:
                    name = "未知用户"
                quoted = MessageArray.from_list(
                    message_data.message or []
                ).text
                quote_lines.append(f"{name}：{quoted}")
            quote_lines.append("</quote>")
            text_buf.append("\n".join(quote_lines) + "\n")

        multimodal_parts: List[Dict[str, Any]] = []
        for seg in messages:
            if isinstance(seg, PlainText):
                text_buf.append(seg.text)
                # 文本段也加入多模态 parts，保证最终排版与纯文本版本一致
                # （如果消息里没有任何图片，下方会整体走 str 分支，不会用到这些 parts）
                multimodal_parts.append({"type": "text", "text": seg.text})
            elif isinstance(seg, At):
                name = await self.get_user_name(seg.user_id)
                # 历史里用真实昵称；多模态里也用昵称（与历史保持一致）
                mention = f"@{name} "
                text_buf.append(mention)
                multimodal_parts.append({"type": "text", "text": mention})
            elif isinstance(seg, Image):
                url = seg.url or seg.file or ""
                if not url:
                    continue
                # base64:// 前缀按 ncatbot 约定转 data URI
                if url.startswith("base64://"):
                    url = f"data:image/png;base64,{url[9:]}"
                # 历史里只放占位
                text_buf.append("[图片] ")
                multimodal_parts.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
            elif isinstance(seg, Reply):
                # 已在外层 quote 块里处理过，这里跳过避免重复
                continue
            else:
                LOG.info("resolve_message_multimodal: 跳过段 %s", type(seg).__name__)

        text_only = "".join(text_buf)
        # 若没有任何 image_url 段，多模态 parts 没意义，调用方会走纯文本分支
        has_image_part = any(
            p.get("type") == "image_url" for p in multimodal_parts
        )
        if not has_image_part:
            multimodal_parts = []
        return text_only, multimodal_parts