from jmcomic import JmOption, download_album, Feature
import jmcomic
import asyncio
import os

from ncatbot.core import registrar
from ncatbot.event import HasSender
from ncatbot.plugin import NcatBotPlugin
from ncatbot.event.qq import GroupMessageEvent

#LOG = get_log("ConfigAndData")

async def comic_download(id: int) -> str:
    if os.path.exists(f"down/{id}.pdf"):
        return f"down/{id}.pdf" 
    await jmcomic.download_album_async(id, extra=Feature.export_pdf(pdf_dir="down",delete_original_file=True,filename_rule="{Aid}"))
    if os.path.exists(f"down/{id}.pdf"):
        return f"down/{id}.pdf"
    else:
        return "failed"
async def comic_detail(id: int) -> str:
    op = JmOption.default()
    async with op.new_jm_async_client() as cl:
        detail = await cl.get_album_detail(id)

    def join_list(items):
        return "、".join(str(x) for x in items) if items else "无"

    lines = [
        f"ID: {detail.album_id}",
        f"名称: {detail.name}",
        f"作者: {join_list(detail.authors)}",
        f"角色: {join_list(detail.actors)}",
        f"标签: {join_list(detail.tags)}",
        f"章节: {join_list(detail.episode_list)}",
        f"页数: {detail.page_count}",
        f"发布日期: {detail.pub_date}",
        f"更新日期: {detail.update_date}",
        f"喜欢: {detail.likes}    浏览: {detail.views}    评论: {detail.comment_count}",
        f"描述: {detail.description or '无'}",
    ]
    return "\n".join(lines)

class JmPlugin(NcatBotPlugin):
    async def on_load(self):
        self.logger.info(f"{self.name} 已加载")

    async def on_close(self):
        self.logger.info(f"{self.name} 已卸载")
    @registrar.qq.on_group_command("hello", ignore_case=True)
    async def on_group_hello(self, event: GroupMessageEvent):
        await event.reply(text="hi")
    @registrar.on_group_command("/jm",ignore_case=True)
    async def on_gourp_jm_download(self,event: GroupMessageEvent) -> str:
        parts=event.message.text.split(" ")
        if len(parts) < 2:
           return "Syntax Error"
        try:
            id=int(parts[1])        
        except ValueError:
            return "ID must be an integer"
        
        result = await comic_download(id)
        if result=="failed":
            return result
        dir_name, file = result.split("/")
        await self.api.qq.send_group_file(event.group_id,result,name=file)
        return "ok"
    @registrar.on_group_command("/jmd",ignore_case=True)
    async def on_gourp_jm_detail(self,event:GroupMessageEvent) -> str :
        parts=event.message.text.split(" ")
        if len(parts) < 2:
           return "Syntax Error"
        try:
            id=int(parts[1])        
        except ValueError:
            return "ID must be an integer"
        result = await comic_detail(id)
        await event.reply(text=result)
        return "ok"


if __name__ == "__main__":
    asyncio.run(comic_detail(350234))
