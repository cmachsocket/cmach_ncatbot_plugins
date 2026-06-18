"""渲染引擎：纯文本 → PNG（纯 Pillow）。

要点：
- 按字符级字体切换（run-based drawing）
- 自动换行
- 文本框宽度 = ``width`` - 2 * padding_x
- 画布高度根据内容自适应
- 可选 2x 超采样保证清晰度
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

from .fonts import FontSet, cached_fontset, font_for_char, measure, wrap
from .styles import Style, get_style


# ---------------------------------------------------------------------------
# 渐变背景
# ---------------------------------------------------------------------------

def _vertical_gradient(size: tuple[int, int], top: tuple, bottom: tuple) -> Image.Image:
    """线性垂直渐变。``size = (width, height)``。"""
    w, h = size
    if h <= 0:
        return Image.new("RGB", (w, 1), top)
    img = Image.new("RGB", (w, h), top)
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        # 每行填充
        for x in range(w):
            px[x, y] = (r, g, b)
    return img


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _draw_run(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fs: FontSet,
    size: int,
    color,
):
    """把 ``text`` 按字体分段绘制到 ``(x, y)``，返回绘制后的 ``x`` 终点。"""
    cur_x = x
    prev_font = None
    run = ""
    for ch in text:
        if ch == "\n":
            if run and prev_font is not None:
                draw.text((cur_x, y), run, font=prev_font, fill=color)
                cur_x += int(round(prev_font.getlength(run)))
            run = ""
            prev_font = None
            cur_x = x
            y += int(size * 1.5)
            continue
        f = font_for_char(ch, fs, size)
        if f is prev_font:
            run += ch
        else:
            if run and prev_font is not None:
                draw.text((cur_x, y), run, font=prev_font, fill=color)
                cur_x += int(round(prev_font.getlength(run)))
            run = ch
            prev_font = f
    if run and prev_font is not None:
        draw.text((cur_x, y), run, font=prev_font, fill=color)
        cur_x += int(round(prev_font.getlength(run)))
    return cur_x, y


def _text_block_height(text: str, fs: FontSet, style: Style,
                       text_width: int) -> int:
    """计算一段文本渲染后的高度（行数 * 行高）。"""
    lines = wrap(text, fs, style.font_size, text_width)
    line_h = int(style.font_size * style.line_height)
    return len(lines) * line_h


def render(
    text: str,
    *,
    theme: str | Style = "modern",
    width: int = 900,
    font_size: int | None = None,
    line_height: float | None = None,
    out: str | Path,
    fontset: FontSet | None = None,
    scale: int = 2,
) -> Path:
    """把纯文本渲染成 PNG。

    Parameters
    ----------
    text : str
        任意纯文本，含 ``\\n`` 硬换行。多语言自动按字符选字体。
    theme : str | Style
        主题名或自定义 ``Style``。
    width : int
        画布逻辑宽度（px）。实际 PNG 宽度 = ``width * scale``。
    font_size, line_height : 可选，覆盖主题默认值。
    out : str | Path
        输出 PNG 路径。
    fontset : FontSet
        可选：注入已探测好的字体集合（用于 Termux 等环境调优）。
    scale : int
        超采样倍率。2 = 2 倍像素密度（推荐，Retina 清晰度）。
    """
    if isinstance(theme, str):
        style = get_style(theme)
    else:
        style = replace(theme)
    if font_size is not None:
        style.font_size = font_size
    if line_height is not None:
        style.line_height = line_height

    fs = fontset or cached_fontset()

    # 1) 文本框可用宽度
    inner_w = width - 2 * style.padding_x
    if inner_w <= 0:
        raise ValueError(f"画布太窄：width={width} 容不下 padding_x={style.padding_x}")

    # 2) 换行 + 算高度
    lines = wrap(text, fs, style.font_size, inner_w)
    line_h = int(style.font_size * style.line_height)
    text_h = len(lines) * line_h

    # 3) 画布总高
    canvas_h = text_h + 2 * style.padding_y
    if style.show_rule:
        canvas_h += line_h // 2  # 横线 + 间距

    # 4) 创建画布
    bg = style.background
    if len(bg) == 4:
        img = Image.new("RGBA", (width, canvas_h), bg)
    else:
        img = Image.new("RGB", (width, canvas_h), bg)

    # 渐变背景（如果有）
    # 这里支持简单的水平/垂直渐变：检测主题里是否声明了 gradient
    grad = getattr(style, "gradient", None)
    if grad:
        if len(grad[0]) == 4:
            top = grad[0][:3]
            bot = grad[1][:3]
        else:
            top = grad[0]
            bot = grad[1]
        img = _vertical_gradient((width, canvas_h), top, bot)
        img = img.convert("RGBA")

    draw = ImageDraw.Draw(img)

    # 5) 绘制文本
    text_x = style.padding_x
    text_y = style.padding_y
    text_color = style.text if len(style.text) == 4 else style.text + (255,)

    # 把所有行连起来用 _draw_run（它会处理 \\n，但我们已经按 \\n 分好行了）
    for i, line in enumerate(lines):
        y = text_y + i * line_h
        # 垂直对齐：每行 baseline 大致在 size 下方 ~80% 处
        _draw_run(draw, text_x, y, line, fs, style.font_size, text_color)

    # 6) 装饰横线
    if style.show_rule:
        rule_y = text_y + text_h + line_h // 4
        rule_color = style.rule if len(style.rule) == 4 else style.rule + (200,)
        draw.line(
            [(style.padding_x, rule_y),
             (width - style.padding_x, rule_y)],
            fill=rule_color, width=2,
        )

    # 7) 超采样放大
    if scale != 1:
        img = img.resize((width * scale, canvas_h * scale), Image.LANCZOS)

    # 8) 保存
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 彩色转 RGB（A 通道如果不是透明就丢了；这里通常是有底色所以 RGBA→RGB 也行）
    if img.mode == "RGBA":
        # 用背景色合成
        bg_flat = style.background[:3]
        flat = Image.new("RGB", img.size, bg_flat)
        flat.paste(img, mask=img.split()[3])
        img = flat
    img.save(out_path, "PNG", optimize=True)
    return out_path