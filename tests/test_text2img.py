"""text2img 的字体、换行、渲染引擎和 CLI 测试。"""

import argparse
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jm.text2img import cli, engine, fonts, styles
from jm.text2img.fonts import FontSet
from jm.text2img.styles import Style
from PIL import Image

# ---------- styles ----------


def test_get_style_returns_independent_copy() -> None:
    first = styles.get_style("dark")
    second = styles.get_style("dark")

    first.font_size = 99

    assert first is not second
    assert second.font_size == styles.THEMES["dark"].font_size
    assert styles.THEMES["dark"].font_size != 99


def test_get_style_rejects_unknown_theme() -> None:
    with pytest.raises(KeyError, match="未知主题"):
        styles.get_style("missing")


def test_list_styles_contains_presets() -> None:
    names = styles.list_styles()

    assert "modern" in names
    assert "dark" in names
    assert names == list(styles.THEMES)


# ---------- fonts ----------


@pytest.mark.parametrize(
    ("char", "script"),
    [
        ("a", "latin"),
        ("中", "cjk"),
        ("あ", "cjk_jp"),
        ("ア", "cjk_jp"),
        ("한", "cjk_kr"),
        ("،", "arabic"),
        ("א", "hebrew"),
        ("ก", "thai"),
        ("Б", "cyrillic"),
        ("Ω", "greek"),
        ("Ā", "latin_ext"),
        ("、", "cjk_punct"),
        ("，", "cjk_punct"),
        ("😀", "emoji"),
        ("⚡", "emoji_misc"),
        ("☀", "emoji_misc"),
    ],
)
def test_char_script_classification(char: str, script: str) -> None:
    assert fonts.char_script(char) == script


def test_char_script_handles_empty_string() -> None:
    assert fonts.char_script("") == "latin"


def test_char_script_handles_cjk_extension_b() -> None:
    assert fonts.char_script(chr(0x20000)) == "cjk"


def test_fontset_font_for_prefers_region_specific_fonts() -> None:
    fontset = FontSet(
        sans="sans.ttf",
        cjk="cjk.ttf",
        cjk_sc="sc.ttf",
        cjk_jp="jp.ttf",
        cjk_kr="kr.ttf",
        emoji="emoji.ttf",
        arabic="arabic.ttf",
        hebrew="hebrew.ttf",
        thai="thai.ttf",
        cyrillic="cyrillic.ttf",
    )

    assert fontset.font_for("cjk") == "sc.ttf"
    assert fontset.font_for("cjk_punct") == "sc.ttf"
    assert fontset.font_for("cjk_jp") == "jp.ttf"
    assert fontset.font_for("cjk_kr") == "kr.ttf"
    assert fontset.font_for("emoji") == "emoji.ttf"
    assert fontset.font_for("emoji_misc") == "emoji.ttf"
    assert fontset.font_for("arabic") == "arabic.ttf"
    assert fontset.font_for("hebrew") == "hebrew.ttf"
    assert fontset.font_for("thai") == "thai.ttf"
    assert fontset.font_for("cyrillic") == "cyrillic.ttf"
    assert fontset.font_for("latin") == "sans.ttf"


def test_fontset_font_for_falls_back_to_generic_fonts() -> None:
    fontset = FontSet(sans="sans.ttf", cjk="cjk.ttf")

    assert fontset.font_for("cjk_jp") == "cjk.ttf"
    assert fontset.font_for("cjk_kr") == "cjk.ttf"
    assert fontset.font_for("emoji") == "sans.ttf"
    assert fontset.font_for("unknown") == "sans.ttf"


@pytest.mark.parametrize(
    ("paths", "keywords", "fallback", "expected"),
    [
        ([], ["Sans"], False, None),
        (
            ["/f/DejaVuSans.ttf", "/f/NotoSans.ttf"],
            ["Noto", "DejaVu"],
            False,
            "/f/NotoSans.ttf",
        ),
        (["/f/DejaVuSans.ttf"], ["Missing"], False, None),
        (["/f/DejaVuSans.ttf"], ["Missing"], True, "/f/DejaVuSans.ttf"),
        (["/f/a.ttf", "/f/a.ttf"], ["A"], False, "/f/a.ttf"),
        (["/f/MONO.ttf"], ["mono"], False, "/f/MONO.ttf"),
    ],
)
def test_pick_first(paths, keywords, fallback, expected) -> None:
    assert fonts._pick_first(paths, keywords, fallback) == expected


def test_classify_groups_font_families() -> None:
    classes = fonts._classify(
        [
            "/f/NotoCJK-SC-Regular.otf",
            "/f/NotoCJK-JP-Regular.otf",
            "/f/NotoCJK-KR-Regular.otf",
            "/f/NotoSerif-Regular.ttf",
            "/f/JetBrainsMono-Regular.ttf",
            "/f/NotoSans-Regular.ttf",
            "/f/NotoArabic-Regular.ttf",
            "/f/NotoHebrew-Regular.ttf",
            "/f/NotoThai-Regular.ttf",
            "/f/NotoCyrillic-Regular.ttf",
            "/f/NotoColorEmoji.ttf",
            "/f/Mystery.ttf",
        ]
    )

    assert "/f/NotoCJK-SC-Regular.otf" in classes["cjk_sc"]
    assert "/f/NotoCJK-JP-Regular.otf" in classes["cjk_jp"]
    assert "/f/NotoCJK-KR-Regular.otf" in classes["cjk_kr"]
    assert "/f/NotoSerif-Regular.ttf" in classes["serif"]
    assert "/f/JetBrainsMono-Regular.ttf" in classes["mono"]
    assert "/f/NotoSans-Regular.ttf" in classes["sans"]
    assert "/f/NotoArabic-Regular.ttf" in classes["arabic"]
    assert "/f/NotoHebrew-Regular.ttf" in classes["hebrew"]
    assert "/f/NotoThai-Regular.ttf" in classes["thai"]
    assert "/f/NotoCyrillic-Regular.ttf" in classes["cyrillic"]
    assert "/f/NotoColorEmoji.ttf" in classes["emoji"]
    assert "/f/Mystery.ttf" in classes["sans"]


def test_describe_reports_present_and_missing_fonts() -> None:
    text = fonts.describe(FontSet(sans="sans.ttf"))

    assert "FontSet:" in text
    assert "✓ sans" in text
    assert "✗ cjk" in text


def test_load_empty_path_uses_default_font() -> None:
    font = fonts._load("", 16)

    assert font.getlength("abc") > 0


def test_load_invalid_path_uses_default_font() -> None:
    font = fonts._load("/definitely/not/a/font.ttf", 16)

    assert font.getlength("abc") > 0


def test_measure_returns_integer_width_for_ascii() -> None:
    width = fonts.measure("hello", FontSet(), 16)

    assert isinstance(width, int)
    assert width > 0


def test_wrap_splits_cjk_per_character(monkeypatch) -> None:
    monkeypatch.setattr(fonts, "measure", lambda text, fs, size: len(text))

    assert fonts.wrap("你好世界", FontSet(), 10, 2) == ["你好", "世界"]


def test_wrap_keeps_ascii_words_together(monkeypatch) -> None:
    monkeypatch.setattr(fonts, "measure", lambda text, fs, size: len(text))

    assert fonts.wrap("hello world", FontSet(), 10, 5) == ["hello", "world"]


def test_wrap_force_splits_word_longer_than_width(monkeypatch) -> None:
    monkeypatch.setattr(fonts, "measure", lambda text, fs, size: len(text))

    assert fonts.wrap("abcdef", FontSet(), 10, 3) == ["abc", "def"]


def test_wrap_preserves_hard_newlines(monkeypatch) -> None:
    monkeypatch.setattr(fonts, "measure", lambda text, fs, size: len(text))

    assert fonts.wrap("a\n\nb", FontSet(), 10, 100) == ["a", "", "b"]


def test_wrap_trims_trailing_spaces(monkeypatch) -> None:
    monkeypatch.setattr(fonts, "measure", lambda text, fs, size: len(text))

    assert fonts.wrap("hello   ", FontSet(), 10, 100) == ["hello"]


# ---------- engine ----------


def test_vertical_gradient_interpolates_rows() -> None:
    image = engine._vertical_gradient((3, 5), (0, 0, 0), (255, 255, 255))

    assert image.size == (3, 5)
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((0, 4)) == (255, 255, 255)
    middle = image.getpixel((0, 2))
    assert isinstance(middle, tuple)
    red = middle[0]
    assert isinstance(red, int)
    assert 0 < red < 255


def test_vertical_gradient_handles_zero_height() -> None:
    image = engine._vertical_gradient((4, 0), (1, 2, 3), (9, 9, 9))

    assert image.size == (4, 1)
    assert image.getpixel((0, 0)) == (1, 2, 3)


def test_render_creates_png_with_expected_size(tmp_path) -> None:
    style = Style(
        name="test",
        font_size=20,
        line_height=1.5,
        padding_x=10,
        padding_y=10,
    )
    out = tmp_path / "nested" / "out.png"

    result = engine.render(
        "hello\nworld",
        theme=style,
        width=200,
        out=out,
        fontset=FontSet(),
        scale=1,
    )

    assert result == out
    assert out.exists()
    with Image.open(out) as image:
        assert image.size == (200, 2 * 30 + 20)
        assert image.mode == "RGB"


def test_render_scale_doubles_pixels(tmp_path) -> None:
    out = tmp_path / "scaled.png"

    engine.render(
        "hello",
        theme=Style(
            name="test", font_size=10, line_height=1.0, padding_x=5, padding_y=5
        ),
        width=100,
        out=out,
        fontset=FontSet(),
        scale=2,
    )

    with Image.open(out) as image:
        assert image.size == (200, 2 * (10 + 10))


def test_render_draws_rule_and_flattens_rgba(tmp_path) -> None:
    style = Style(
        name="rgba",
        background=(0, 0, 0, 0),
        text=(255, 255, 255, 255),
        rule=(255, 0, 0, 255),
        font_size=10,
        line_height=1.0,
        padding_x=5,
        padding_y=5,
        show_rule=True,
    )
    out = tmp_path / "rgba.png"

    engine.render("abc", theme=style, width=100, out=out, fontset=FontSet(), scale=1)

    with Image.open(out) as image:
        assert image.mode == "RGB"
        assert image.size == (100, 10 + 10 + 5)


def test_render_applies_font_size_and_line_height_overrides(tmp_path) -> None:
    base = tmp_path / "base.png"
    overridden = tmp_path / "overridden.png"
    style = Style(name="test", font_size=10, line_height=1.0, padding_x=5, padding_y=5)

    engine.render("a", theme=style, width=100, out=base, fontset=FontSet(), scale=1)
    engine.render(
        "a",
        theme=style,
        width=100,
        font_size=20,
        line_height=2.0,
        out=overridden,
        fontset=FontSet(),
        scale=1,
    )

    with Image.open(base) as base_image, Image.open(overridden) as over_image:
        assert base_image.height == 20
        assert over_image.height == 20 * 2 + 10


def test_render_rejects_canvas_narrower_than_padding(tmp_path) -> None:
    with pytest.raises(ValueError, match="画布太窄"):
        engine.render(
            "hello",
            theme=Style(name="test", padding_x=100),
            width=50,
            out=tmp_path / "bad.png",
            fontset=FontSet(),
        )


def test_render_empty_text_produces_image(tmp_path) -> None:
    out = tmp_path / "empty.png"

    engine.render(
        "",
        theme=Style(
            name="test", font_size=10, line_height=1.0, padding_x=5, padding_y=5
        ),
        width=100,
        out=out,
        fontset=FontSet(),
        scale=1,
    )

    assert out.exists()


# ---------- CLI ----------


def _args(**kwargs: Any) -> argparse.Namespace:
    defaults = {"text": None, "file": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_read_text_prefers_positional_text() -> None:
    assert cli._read_text(_args(text="hello", file="ignored.txt")) == "hello"


def test_read_text_reads_file(tmp_path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("from file", encoding="utf-8")

    assert cli._read_text(_args(file=str(path))) == "from file"


def test_read_text_reads_stdin(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("from stdin"))

    assert cli._read_text(_args()) == "from stdin"


def test_read_text_exits_without_input(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    with pytest.raises(SystemExit) as exc:
        cli._read_text(_args())

    assert exc.value.code == 2
    assert "需要提供文本" in capsys.readouterr().err


def test_cli_lists_themes(capsys) -> None:
    assert cli.main(["--list-themes"]) == 0
    assert "modern" in capsys.readouterr().out


def test_cli_shows_fonts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "cached_fontset", lambda: FontSet(sans="sans.ttf"))

    assert cli.main(["--show-fonts"]) == 0
    output = capsys.readouterr().out
    assert "FontSet:" in output
    assert "sans.ttf" in output


def test_cli_requires_output_path(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code == 2
    assert "需要 -o/--out" in capsys.readouterr().err


def test_cli_renders_text(monkeypatch, tmp_path, capsys) -> None:
    out = tmp_path / "cli.png"
    calls: list[dict[str, Any]] = []

    def fake_render(text: str, **kwargs: Any) -> Path:
        calls.append({"text": text, **kwargs})
        out.write_bytes(b"png")
        return out

    monkeypatch.setattr(cli, "render", fake_render)

    code = cli.main(
        [
            "hello",
            "-o",
            str(out),
            "-t",
            "modern",
            "-w",
            "300",
            "--font-size",
            "20",
            "--line-height",
            "1.5",
            "--scale",
            "1",
        ]
    )

    assert code == 0
    assert calls == [
        {
            "text": "hello",
            "theme": "modern",
            "width": 300,
            "font_size": 20,
            "line_height": 1.5,
            "out": str(out),
            "scale": 1,
        }
    ]
    assert "✓" in capsys.readouterr().out


def test_cli_reads_text_from_file(monkeypatch, tmp_path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("from file", encoding="utf-8")
    out = tmp_path / "out.png"

    def fake_render(text: str, **kwargs: Any) -> Path:
        assert text == "from file"
        out.write_bytes(b"png")
        return out

    monkeypatch.setattr(cli, "render", fake_render)

    assert cli.main(["-f", str(source), "-o", str(out)]) == 0
