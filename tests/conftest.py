"""测试公共路径设置。

插件入口按 ncatbot 的加载方式以插件目录名作为包名导入，
因此需要把仓库根目录和 ``plugins/`` 都加入 ``sys.path``。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"

for path in (ROOT, PLUGINS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
