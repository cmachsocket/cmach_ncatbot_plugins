"""手动跑一次 RSS 摘要发送流程。

用法（在插件目录下）：
    uv run test/test.py
"""

import asyncio
import sys
from pathlib import Path

# ncatbot.plugin 与 ncatbot.core 互相 import，直接先 import ncatbot.plugin
# 会抛 "partially initialized module" 循环导入错误，这里先加载 core 打破循环。
import ncatbot.core  # noqa: F401

# 以脚本方式运行时没有父包，``from ..rss import`` 无法解析，
# 改为把插件目录加进 sys.path 后按普通模块导入。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rss import RssSender  # noqa: E402


async def test_send_rss_summary():
    rss_sender = RssSender()
    await rss_sender.send_rss_summary()


if __name__ == "__main__":
    asyncio.run(test_send_rss_summary())
