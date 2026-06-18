"""``text2png`` 命令行入口（text2img 的 CLI）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import render
from .fonts import cached_fontset, describe
from .styles import list_styles


def _read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file and args.file != "-":
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("错误：需要提供文本、文件，或通过 stdin 管道。", file=sys.stderr)
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="text2img",
        description="纯文本 → 多语言 PNG（纯 Pillow，Termux 友好）",
    )
    parser.add_argument("text", nargs="?", help="直接给的文本（也可用 -f 或 stdin）")
    parser.add_argument("-f", "--file", help="从文件读取（- 表示 stdin）")
    parser.add_argument("-o", "--out", required=False, help="输出 PNG 路径")
    parser.add_argument("-t", "--theme", default="modern",
                        choices=list_styles(),
                        help="主题（默认 modern）")
    parser.add_argument("-w", "--width", type=int, default=900,
                        help="画布宽度 px（默认 900）")
    parser.add_argument("--font-size", type=int, default=None,
                        help="字号 px（覆盖主题）")
    parser.add_argument("--line-height", type=float, default=None,
                        help="行高倍数（覆盖主题）")
    parser.add_argument("--scale", type=int, default=2,
                        help="超采样倍率，默认 2")
    parser.add_argument("--list-themes", action="store_true",
                        help="列出所有可用主题")
    parser.add_argument("--show-fonts", action="store_true",
                        help="打印探测到的字体")
    args = parser.parse_args(argv)

    if args.list_themes:
        print("\n".join(list_styles()))
        return 0

    if args.show_fonts:
        print(describe(cached_fontset()))
        return 0

    if not args.out:
        print("错误：需要 -o/--out，或者使用 --list-themes / --show-fonts。", file=sys.stderr)
        sys.exit(2)

    text = _read_text(args)
    out = render(
        text,
        theme=args.theme,
        width=args.width,
        font_size=args.font_size,
        line_height=args.line_height,
        out=args.out,
        scale=args.scale,
    )
    print(f"✓ {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())