"""字体探测与字符级字体选择 —— 多语言混排的关键。

设计目标：
- 零硬编码路径：探测 ``fc-list``，失败则 glob ``/system/fonts`` + ``/usr/share/fonts``
- 按 Unicode 脚本分类字体：CJK / 阿拉伯 / 泰文 / 希伯来 / 西里尔 / emoji
- 每个字符按其脚本选最佳字体（不是全文用一种字体）
- lru_cache 缓存字体加载，Termux ARM 上不能慢

Termux 上需要的字体包（提前装好）::

    pkg install noto-fonts        # Noto Sans/Serif CJK + 拉丁
    pkg install noto-fonts-emoji  # Noto Color Emoji（黑白，但能识别）
    pkg install fontconfig-utils   # 提供 fc-list（可选，没装也能 glob 兜底）
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from PIL import ImageFont


# ---------------------------------------------------------------------------
# Unicode 脚本分类
# ---------------------------------------------------------------------------

# 顺序很关键：先查小范围（如日文假名），再查大范围（CJK 统一表意）
UNICODE_RANGES: tuple[tuple[str, int, int], ...] = (
    # 表情符号
    ("emoji",       0x1F300, 0x1FAFF),  # emoji 大类
    ("emoji_misc",  0x2600, 0x27BF),    # 杂项符号（含 ⚡ ★ 等）
    ("emoji_misc",  0x2300, 0x23FF),    # 杂类技术符号
    # 日文假名（要在 CJK 之前匹配，否则会被当成 CJK）
    ("cjk_jp",      0x3040, 0x309F),    # 平假名
    ("cjk_jp",      0x30A0, 0x30FF),    # 片假名
    ("cjk_jp",      0x31F0, 0x31FF),    # 片假名语音扩展
    # 韩文
    ("cjk_kr",      0xAC00, 0xD7AF),    # 谚文音节
    ("cjk_kr",      0x1100, 0x11FF),    # 谚文字母
    ("cjk_kr",      0x3130, 0x318F),    # 谚文兼容字母
    # CJK 统一表意（中文为主，但日韩也用）
    ("cjk",         0x4E00, 0x9FFF),
    ("cjk",         0x3400, 0x4DBF),    # 扩展 A
    ("cjk",         0x20000, 0x2A6DF),  # 扩展 B（用 surrogate pair）
    ("cjk",         0x2A700, 0x2EBEF),  # 扩展 C-F
    ("cjk",         0xF900, 0xFAFF),    # 兼容表意
    # CJK 标点 / 全角
    ("cjk_punct",   0x3000, 0x303F),
    ("cjk_punct",   0xFF00, 0xFFEF),
    # 阿拉伯文
    ("arabic",      0x0600, 0x06FF),
    ("arabic",      0x0750, 0x077F),
    ("arabic",      0xFB50, 0xFDFF),
    ("arabic",      0xFE70, 0xFEFF),
    # 希伯来文
    ("hebrew",      0x0590, 0x05FF),
    # 泰文
    ("thai",        0x0E00, 0x0E7F),
    # 天城文（印地语等）
    ("devanagari",  0x0900, 0x097F),
    # 西里尔
    ("cyrillic",    0x0400, 0x04FF),
    ("cyrillic",    0x0500, 0x052F),
    # 希腊
    ("greek",       0x0370, 0x03FF),
    # 拉丁扩展（含越南文声调）
    ("latin_ext",   0x0100, 0x024F),
    ("latin_ext",   0x1E00, 0x1EFF),
    ("latin_ext",   0x1EA0, 0x1EF9),
    # 拉丁基本
    ("latin",       0x0020, 0x007E),
)


def char_script(ch: str) -> str:
    """给一个字符打脚本标签，用于字体选择。"""
    if not ch:
        return "latin"
    cp = ord(ch[0])
    # 高位 surrogate pair（CJK 扩展 B+）
    if 0xD800 <= cp <= 0xDBFF and len(ch) >= 2:
        cp = 0x10000 + ((cp - 0xD800) << 10) + (ord(ch[1]) - 0xDC00)
    for label, start, end in UNICODE_RANGES:
        if start <= cp <= end:
            return label
    return "latin"


# ---------------------------------------------------------------------------
# 字体集合
# ---------------------------------------------------------------------------

@dataclass
class FontSet:
    """一套字体，按 Unicode 脚本分类。空字符串表示未找到（自动降级）。"""
    sans: str = ""
    serif: str = ""
    mono: str = ""
    cjk: str = ""
    cjk_sc: str = ""
    cjk_jp: str = ""
    cjk_kr: str = ""
    arabic: str = ""
    hebrew: str = ""
    thai: str = ""
    cyrillic: str = ""
    emoji: str = ""

    def font_for(self, label: str) -> str:
        """按脚本标签返回最佳字体路径（找不到则降级）。"""
        if label.startswith("emoji"):
            return self.emoji or self.sans
        if label.startswith("cjk_jp"):
            return self.cjk_jp or self.cjk or self.sans
        if label.startswith("cjk_kr"):
            return self.cjk_kr or self.cjk or self.sans
        if label in ("cjk", "cjk_punct"):
            return self.cjk_sc or self.cjk or self.sans
        if label == "arabic":
            return self.arabic or self.sans
        if label == "hebrew":
            return self.hebrew or self.sans
        if label == "thai":
            return self.thai or self.sans
        if label == "cyrillic":
            return self.cyrillic or self.sans
        if label in ("greek", "latin_ext", "latin", "devanagari"):
            return self.sans
        return self.sans


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------

def _fc_list(query: str) -> list[str]:
    """``fc-list :lang=zh file`` 风格的字体路径探测。失败返回空列表。"""
    try:
        out = subprocess.check_output(
            ["fc-list", query, "file"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        paths = []
        for line in out.splitlines():
            if ":" in line:
                paths.append(line.split(":")[0].strip())
        return paths
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, PermissionError):
        return []


def _glob_system_fonts() -> list[str]:
    """Termux 上 ``/system/fonts``，Linux 上 ``/usr/share/fonts``，Android ``/data/fonts``。"""
    candidates = []
    roots = [
        "/system/fonts",
        "/data/fonts",
        "/product/fonts",
        "/vendor/fonts",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/Library/Fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]
    for root in roots:
        if os.path.isdir(root):
            for ext in ("ttf", "otf", "ttc"):
                candidates.extend(glob.glob(os.path.join(root, f"**/*.{ext}"),
                                            recursive=True))
    return candidates


def _pick_first(paths: list[str], keywords: list[str]) -> Optional[str]:
    """从路径列表里按关键词优先级挑一个。空列表返回 None。"""
    if not paths:
        return None
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    for kw in keywords:
        kw_l = kw.lower()
        for p in unique:
            if kw_l in os.path.basename(p).lower():
                return p
    return unique[0]


def _classify(paths: list[str]) -> dict[str, list[str]]:
    """把字体路径按家族分类。"""
    classes: dict[str, list[str]] = {
        "cjk": [], "cjk_sc": [], "cjk_jp": [], "cjk_kr": [],
        "sans": [], "serif": [], "mono": [],
        "arabic": [], "hebrew": [], "thai": [], "cyrillic": [],
        "emoji": [],
    }
    for p in paths:
        name = os.path.basename(p).lower()
        # emoji 优先（避免被 CJK 抢走）
        if "emoji" in name:
            classes["emoji"].append(p)
            continue
        if any(kw in name for kw in ("cjk", "han", "pingfang", "yahei", "simsun",
                                     "songti", "wenquanyi", "sourcehan", "notosanscjk")):
            classes["cjk"].append(p)
            if any(kw in name for kw in ("sc", "cn", "chs")):
                classes["cjk_sc"].append(p)
            elif "jp" in name:
                classes["cjk_jp"].append(p)
            elif "kr" in name:
                classes["cjk_kr"].append(p)
            # 如果只有 cjk 没标区域，先归 cjk_sc
            elif not classes["cjk_sc"]:
                classes["cjk_sc"].append(p)
        elif "serif" in name:
            classes["serif"].append(p)
        elif any(kw in name for kw in ("mono", "code", "console")):
            classes["mono"].append(p)
        elif "sans" in name:
            classes["sans"].append(p)
        elif "naskh" in name or "arabic" in name:
            classes["arabic"].append(p)
        elif "hebrew" in name:
            classes["hebrew"].append(p)
        elif "thai" in name:
            classes["thai"].append(p)
        elif "cyrillic" in name:
            classes["cyrillic"].append(p)
        else:
            classes["sans"].append(p)  # 默认归 sans
    return classes


def detect_fontset() -> FontSet:
    """探测系统，返回一个 FontSet。

    探测顺序：
    1. ``fc-list :lang=zh`` 拿到 CJK 字体
    2. ``fc-list``（全量）拿到其他字体
    3. 失败时 glob ``/system/fonts`` + ``/usr/share/fonts``
    """
    fs = FontSet()

    # 1) fc-list：分语言拿更精确
    cjk_paths = _fc_list(":lang=zh")
    if not cjk_paths:
        cjk_paths = _fc_list(":lang=ja")  # 至少拿个 CJK 兜底

    arabic_paths = _fc_list(":lang=ar")
    hebrew_paths = _fc_list(":lang=he")
    thai_paths = _fc_list(":lang=th")
    cyrillic_paths = _fc_list(":lang=ru")
    emoji_paths = _fc_list(":lang=und-zsye")  # 通用 emoji 集合

    all_paths = _fc_list(":")

    # 2) 兜底 glob（如果 fc-list 没拿到 CJK）
    if not cjk_paths:
        globbed = _glob_system_fonts()
        all_paths = all_paths or globbed
        cjk_paths = globbed  # 让后续分类处理

    # 3) 分类
    classes = _classify(all_paths)
    # 合并手动拿到的特定语种
    classes["cjk"].extend(cjk_paths)
    classes["arabic"].extend(arabic_paths)
    classes["hebrew"].extend(hebrew_paths)
    classes["thai"].extend(thai_paths)
    classes["cyrillic"].extend(cyrillic_paths)
    classes["emoji"].extend(emoji_paths)

    # 4) 填充 FontSet
    fs.cjk = _pick_first(classes["cjk"], ["NotoSansCJK-Regular", "NotoSansCJK",
                                          "WenQuanYi", "SourceHan", "DroidSansFallback"])
    fs.cjk_sc = (
        _pick_first(classes["cjk_sc"], ["SC", "CN", "NotoSansCJK", "SourceHanSansCN"])
        or _pick_first(classes["cjk"], ["SC", "CN"])
        or fs.cjk
    )
    fs.cjk_jp = (
        _pick_first(classes["cjk_jp"], ["JP"])
        or _pick_first(classes["cjk"], ["JP"])
        or fs.cjk
    )
    fs.cjk_kr = (
        _pick_first(classes["cjk_kr"], ["KR"])
        or _pick_first(classes["cjk"], ["KR"])
        or fs.cjk
    )
    fs.sans = (
        _pick_first(classes["sans"], ["NotoSans-Regular", "NotoSans", "DejaVuSans"])
        or fs.cjk_sc or fs.cjk
    )
    fs.serif = (
        _pick_first(classes["serif"], ["NotoSerif-Regular", "NotoSerif", "DejaVuSerif"])
        or fs.sans
    )
    fs.mono = (
        _pick_first(classes["mono"], ["JetBrainsMapleMono", "JetBrainsMono", "DejaVuSansMono", "DroidSansMono"])
        or fs.sans
    )
    fs.arabic = _pick_first(classes["arabic"], ["NotoSansArabic", "Arabic"]) or fs.sans
    fs.hebrew = _pick_first(classes["hebrew"], ["NotoSansHebrew", "Hebrew"]) or fs.sans
    fs.thai = _pick_first(classes["thai"], ["NotoSansThai", "Thai"]) or fs.sans
    fs.cyrillic = _pick_first(classes["cyrillic"], ["NotoSans", "Cyrillic"]) or fs.sans
    fs.emoji = _pick_first(classes["emoji"], ["ColorEmoji", "Emoji"]) or ""

    return fs


@lru_cache(maxsize=1)
def cached_fontset() -> FontSet:
    return detect_fontset()


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _load(path: str, size: int) -> ImageFont.FreeTypeFont:
    """加载字体。路径为空则用 PIL 默认字体（很丑但至少不报错）。"""
    if not path:
        return ImageFont.load_default(size=size)
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        # TTC 多字面体：可能需要 index，但 truetype 默认通常能处理
        try:
            return ImageFont.truetype(path, size=size, index=0)
        except OSError:
            return ImageFont.load_default(size=size)


@lru_cache(maxsize=512)
def _font_for_char(ch: str, fs_id: int, size: int) -> ImageFont.FreeTypeFont:
    """按字符选字体并加载。``fs_id`` 是 FontSet 对象的 id()，lru_cache 需要可哈希。"""
    fs = _FS_BY_ID.get(fs_id)
    if fs is None:
        return _load("", size)
    label = char_script(ch)
    path = fs.font_for(label)
    return _load(path, size)


# 用 id(fs) → fs 的映射，避开把 FontSet 放进 lru_cache（不可哈希）
_FS_BY_ID: dict[int, FontSet] = {}


def register_fontset(fs: FontSet) -> int:
    """把 FontSet 注册成可缓存的形式，返回它的 id。"""
    fid = id(fs)
    _FS_BY_ID[fid] = fs
    return fid


def font_for_char(ch: str, fs: FontSet, size: int) -> ImageFont.FreeTypeFont:
    fid = register_fontset(fs)
    return _font_for_char(ch, fid, size)


# ---------------------------------------------------------------------------
# 测量 & 换行
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+"           # ASCII 单词
    r"|\s"                     # 空白
    r"|[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"  # 单个 CJK 字符
    r"|."                      # 其他单个字符
)


def measure(text: str, fs: FontSet, size: int) -> int:
    """返回文本渲染宽度（px）。会按字符选字体。"""
    w = 0
    for ch in text:
        f = font_for_char(ch, fs, size)
        w += f.getlength(ch)
    return int(round(w))


def wrap(text: str, fs: FontSet, size: int, max_width: int) -> list[str]:
    """按像素宽度贪心换行。

    - CJK 单字可断
    - ASCII 单词不拆
    - ``\\n`` 是硬换行
    - 超长 CJK 串强制逐字拆
    """
    lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        cur = ""
        cur_w = 0
        for tok in _TOKEN_RE.findall(para):
            if not tok.strip():
                # 空白：保留但避免开头
                if cur and not cur.endswith(" "):
                    cur += tok
                continue
            tok_w = measure(tok, fs, size)
            if cur_w + tok_w <= max_width:
                cur += tok
                cur_w += tok_w
            else:
                if cur.rstrip():
                    lines.append(cur.rstrip())
                if tok_w > max_width:
                    buf = ""
                    buf_w = 0
                    for ch in tok:
                        cw = measure(ch, fs, size)
                        if buf_w + cw > max_width and buf:
                            lines.append(buf)
                            buf = ch
                            buf_w = cw
                        else:
                            buf += ch
                            buf_w += cw
                    cur = buf
                    cur_w = buf_w
                else:
                    cur = tok
                    cur_w = tok_w
        if cur.rstrip():
            lines.append(cur.rstrip())
    return lines


# ---------------------------------------------------------------------------
# 调试
# ---------------------------------------------------------------------------

def describe(fs: FontSet) -> str:
    """格式化输出 FontSet 探测结果，方便调试。"""
    lines = ["FontSet:"]
    for field in fs.__dataclass_fields__:
        val = getattr(fs, field)
        marker = "✓" if val else "✗"
        lines.append(f"  {marker} {field:10s} {val or '(fallback)'}")
    return "\n".join(lines)