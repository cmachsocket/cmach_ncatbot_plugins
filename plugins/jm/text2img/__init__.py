"""text2img — 纯文本 → 多语言 PNG（纯 Pillow，Termux 友好）。

Quick start::

    from text2img import render

    render("你好世界\\nHello World\\nこんにちは", theme="dark", out="out.png")
"""

from .engine import render
from .fonts import FontSet, describe, detect_fontset, measure, wrap
from .styles import Style, get_style, list_styles

__all__ = [
    "FontSet",
    "Style",
    "describe",
    "detect_fontset",
    "get_style",
    "list_styles",
    "measure",
    "render",
    "wrap",
]
__version__ = "0.1.0"
