from ncatbot.core import registrar
from ncatbot.event import HasSender
from ncatbot.plugin import NcatBotPlugin
from ncatbot.event.qq import PrivateMessageEvent
from wakeonlan import send_magic_packet

mac_enp4s0="b0:25:aa:85:01:d3"
mac_wlp2s0="f8:3d:c6:1a:b9:a6"

class WolPlugin(NcatBotPlugin):
    async def on_load(self):
        self.logger.info(f"{self.name} 已加载")

    async def on_close(self):
        self.logger.info(f"{self.name} 已卸载")
    @registrar.qq.on_private_command("hello", ignore_case=True)
    async def on_private_hello(self, event: PrivateMessageEvent):
        await event.reply(text="hi")
    @registrar.qq.on_private_command("wol", ignore_case=True)
    async def on_private_wol(self,event:PrivateMessageEvent):
        await event.reply(text="正在发送唤醒包...")
        send_magic_packet(mac_enp4s0)
        send_magic_packet(mac_wlp2s0)
        await event.reply(text="唤醒包已发送")