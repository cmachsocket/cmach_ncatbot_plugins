from ncatbot.plugin import NcatBotPlugin
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pathlib import Path
import rssfetch

#: rssfetch 的配置文件固定放在插件目录下，不受运行时工作目录影响
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

class RssSender(NcatBotPlugin):
    target_id :int
    scheduler : AsyncIOScheduler
    async def on_load(self):
        self.logger.info(f"{self.name} 已加载")
        self.target_id = self.config.get("target_id",0)
        if self.target_id == 0:
            self.logger.error("未配置 target_id，请在配置文件中添加 target_id")
            return
        self.scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
        self.scheduler.add_job(self.send_rss_summary,
        "cron",
        hour=6,
        minute=0)
        self.scheduler.start()
        
    async def on_close(self):
        self.logger.info(f"{self.name} 已卸载")
        self.scheduler.shutdown()
    async def send_rss_summary(self):
        config = rssfetch.load_config(CONFIG_PATH)
        report = await rssfetch.run(config)
        # save_path 是相对路径时固定到插件目录，避免输出随启动目录漂移
        if not Path(config.save_path).is_absolute():
            config.save_path = str(CONFIG_PATH.parent / config.save_path)
        if report.summary_path is None:      # LLM 失败/未配齐/无条目时都是 None
            self.logger.error("RSS 摘要生成失败")
            return
        await self.api.qq.send_private_file(self.target_id, str(report.summary_path))
