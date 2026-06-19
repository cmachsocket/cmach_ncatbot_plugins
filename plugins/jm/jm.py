from jmcomic import JmOption, download_album, Feature,JmModuleConfig
import jmcomic
import asyncio
import hashlib
import os

from jm.text2img import render
from ncatbot.core import registrar
from ncatbot.event import HasSender
from ncatbot.plugin import NcatBotPlugin
from ncatbot.event.qq import GroupMessageEvent

async def get_domain() -> None:
    domain_list = JmModuleConfig.get_html_domain_all()
    JmModuleConfig.DOMAIN_HTML_LIST = domain_list # type: ignore

async def txt2img(text: str) -> str:
    try:
        md5 = hashlib.md5()
        md5.update(text.encode('utf-8'))
        md5_digest = md5.hexdigest()
        out_path = f"/tmp/{md5_digest}.png"
        if os.path.exists(out_path):
            return out_path
        await asyncio.to_thread(render, text, theme="dark", out=out_path)
        return out_path
    except Exception as e:
        return ""

async def comic_download(id: int) -> tuple[bool, str]:
    await get_domain()
    if os.path.exists(f"down/{id}.pdf"):
        return ( True, f"down/{id}.pdf")
    try:
        await jmcomic.download_album_async(id, extra=Feature.export_pdf(pdf_dir="down",delete_original_file=True,filename_rule="{Aid}"))
    except jmcomic.MissingAlbumPhotoException as e:
        return (False, f"呜喵…{id} 没找到喵～")
    except jmcomic.PartialDownloadFailedException as e:
        return (False, (f"呜喵…下载失败了呢喵～"))
    except jmcomic.JmcomicException as e:
        return (False, (f"呜喵…JM 出错了喵～"))
    except Exception as e:
        return (False, (f"呜喵…奇怪的错误发生了喵～"))
    if os.path.exists(f"down/{id}.pdf"):
        return (True, f"down/{id}.pdf")
    else:
        return (False, "呜喵…保存失败了喵…")
async def comic_detail(id: int) -> tuple[bool, str]:
    await get_domain()
    op = JmOption.default()
    async with op.new_jm_async_client() as cl:
        try:
            detail = await cl.get_album_detail(id)
        except jmcomic.MissingAlbumPhotoException as e:
            return (False, f"呜喵…{id} 没找到喵～")
        except jmcomic.JsonResolveFailException as e:
            return (False, f"呜喵…JSON 解析炸了喵")
        except jmcomic.RequestRetryAllFailException as e:
            return (False, f"呜喵…请求都失败了喵～")
        except jmcomic.JmcomicException as e:
            return (False, f"呜喵…详情抓取出错了喵")
        except Exception as e:
            return (False, f"呜喵…奇怪的错误发生了喵～")

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
    return (True, "\n".join(lines))

class JmPlugin(NcatBotPlugin):
    async def on_load(self):
        self.logger.info(f"{self.name} 已加载")

    async def on_close(self):
        self.logger.info(f"{self.name} 已卸载")
    @registrar.qq.on_group_command("hello", ignore_case=True)
    async def on_group_hello(self, event: GroupMessageEvent):
        await event.reply(text="hi 喵～")
    @registrar.on_group_command("/jm",ignore_case=True)
    async def on_gourp_jm_download(self,event: GroupMessageEvent) -> None:
        parts=event.message.text.split(" ")
        if len(parts) < 2:
           await event.reply(text="呜喵…语法用错了啦～")
           return
        try:
            id=int(parts[1])        
        except ValueError:
            await event.reply(text="ID 要整数的喵～")
            return

        try:
            async with asyncio.timeout(300):
                result = await comic_download(id)
        except asyncio.TimeoutError:
            await event.reply(text="呜喵…下载太慢了啦，等了好久都没好喵～")
        if not result[0]:
            await event.reply(text=result[1])
            return
        _ , file = result[1].split("/")
        try:
            await self.api.qq.send_group_file(event.group_id,result[1],name=file)
        except asyncio.TimeoutError as e:
            print("WARN : 呜喵…文件发送超时了喵，快看看是不是已经发出去过了喵～")
        except Exception as e:
            await self.api.qq.send_group_plain_text(event.group_id,f"呜喵…ncatbot 发送出错了喵～")
    @registrar.on_group_command("/jmd",ignore_case=True)
    async def on_gourp_jm_detail(self,event:GroupMessageEvent) -> None :
        parts=event.message.text.split(" ")
        if len(parts) < 2:
           await event.reply(text="呜喵…语法用错了啦～")
           return
        try:
            id=int(parts[1])        
        except ValueError:
            await event.reply(text="ID 要整数的喵～")
            return
        try:
            async with asyncio.timeout(300):
                result = await comic_detail(id)
        except asyncio.TimeoutError:
            await event.reply(text="呜喵…详情拉取太慢了喵～")
        if not result[0]:
            await event.reply(text=result[1])
        else:
            if len(parts) > 2:
                raise Exception("进入纯文本输出模式了喵～")
            try:
                img_path = await txt2img(result[1])
                if not img_path:
                    raise Exception("文本转图片失败了喵～")
                await self.api.qq.send_group_image(event.group_id,img_path)
            except Exception as e:
                await event.reply(text=result[1])