"""预设主题（和 text2png 风格一致）。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple

Color = Tuple[int, int, int] | Tuple[int, int, int, int]


@dataclass
class Style:
    name: str

    background: Color = (250, 248, 245)
    text: Color = (28, 28, 30)
    muted: Color = (140, 130, 120)
    accent: Color = (180, 90, 60)
    rule: Color = (220, 215, 210)

    font_size: int = 26
    line_height: float = 1.75
    letter_spacing: float = 0

    padding_x: int = 72
    padding_y: int = 80

    title_font: str = "serif"   # 文本样式对应字体角色（"serif"/"sans"/"mono"）
    body_font: str = "serif"
    mono_font: str = "mono"

    show_rule: bool = False


THEMES: dict[str, Style] = {
    "modern": Style(
        name="modern",
        background=(255, 255, 255),
        text=(26, 26, 26),
        muted=(150, 150, 150),
        accent=(26, 26, 26),
        rule=(225, 225, 225),
        font_size=28, line_height=1.75,
        body_font="sans",
    ),
    "literary": Style(
        name="literary",
        background=(252, 250, 245),
        text=(35, 35, 35),
        muted=(124, 112, 102),
        accent=(124, 60, 30),
        rule=(200, 190, 175),
        font_size=28, line_height=1.85,
        letter_spacing=0.2,
        show_rule=True,
        body_font="serif", title_font="serif",
    ),
    "dark": Style(
        name="dark",
        background=(15, 17, 26),
        text=(235, 238, 245),
        muted=(140, 156, 180),
        accent=(100, 200, 255),
        rule=(50, 60, 80),
        font_size=28, line_height=1.75,
        body_font="sans",
    ),
    "paper": Style(
        name="paper",
        background=(245, 235, 220),
        text=(60, 40, 30),
        muted=(150, 120, 95),
        accent=(170, 70, 40),
        rule=(210, 180, 150),
        font_size=28, line_height=1.85,
        letter_spacing=0.1,
        body_font="serif",
    ),
    "memo": Style(
        name="memo",
        background=(255, 245, 180),
        text=(60, 50, 30),
        muted=(130, 110, 70),
        accent=(220, 80, 60),
        rule=(220, 195, 120),
        font_size=30, line_height=1.7,
        padding_x=64, padding_y=72,
        body_font="sans",
    ),
    "terminal": Style(
        name="terminal",
        background=(12, 14, 16),
        text=(200, 211, 168),
        muted=(122, 138, 106),
        accent=(255, 175, 95),
        rule=(42, 46, 38),
        font_size=22, line_height=1.55,
        body_font="mono", title_font="mono",
    ),
    "ocean": Style(
        name="ocean",
        background=(12, 74, 110),
        text=(240, 249, 255),
        muted=(186, 230, 253),
        accent=(253, 230, 138),
        rule=(255, 255, 255, 50),
        font_size=28, line_height=1.85,
        body_font="serif",
    ),
}


def get_style(name: str) -> Style:
    if name not in THEMES:
        raise KeyError(f"未知主题 {name!r}。可选：{', '.join(THEMES)}")
    return replace(THEMES[name])


def list_styles() -> list[str]:
    return list(THEMES)