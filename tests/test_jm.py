"""JM 漫画插件的下载、详情、文转图与命令测试。"""

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any, Self

import jmcomic
import pytest
from jm import jm as jm_module


class _Event:
    def __init__(self, text: str, group_id: int = 123) -> None:
        self.message = SimpleNamespace(text=text)
        self.group_id = group_id
        self.replies: list[str] = []

    async def reply(self, *, text: str) -> None:
        self.replies.append(text)


class _QQ:
    def __init__(self) -> None:
        self.files: list[tuple[int, str, str]] = []
        self.images: list[tuple[int, str]] = []
        self.file_error: Exception | None = None

    async def send_group_file(self, group_id: int, path: str, *, name: str) -> None:
        if self.file_error is not None:
            raise self.file_error
        self.files.append((group_id, path, name))

    async def send_group_image(self, group_id: int, path: str) -> None:
        self.images.append((group_id, path))


class _Api:
    def __init__(self) -> None:
        self.qq = _QQ()


def _plugin() -> jm_module.JmPlugin:
    plugin = object.__new__(jm_module.JmPlugin)
    plugin.name = "jm-test"
    plugin.api = _Api()  # type: ignore[assignment]
    return plugin


# ---------- 工具函数 ----------


def test_get_domain_updates_module_domain_list(monkeypatch) -> None:
    monkeypatch.setattr(
        jm_module.JmModuleConfig,
        "get_html_domain_all",
        staticmethod(lambda: ["a.example", "b.example"]),
    )
    monkeypatch.setattr(jm_module.JmModuleConfig, "DOMAIN_HTML_LIST", [])

    asyncio.run(jm_module.get_domain())

    assert jm_module.JmModuleConfig.DOMAIN_HTML_LIST == [  # type: ignore[attr-defined]
        "a.example",
        "b.example",
    ]


def test_txt2img_returns_cached_path_without_rendering(monkeypatch) -> None:
    text = "缓存测试"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    rendered: list[Any] = []
    monkeypatch.setattr(jm_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(jm_module, "render", lambda *a, **kw: rendered.append((a, kw)))

    path = asyncio.run(jm_module.txt2img(text))

    assert path == f"/tmp/{digest}.png"
    assert rendered == []


def test_txt2img_renders_dark_theme_on_cache_miss(monkeypatch) -> None:
    text = "渲染测试"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(jm_module.os.path, "exists", lambda path: False)

    def fake_render(value: str, **kwargs: Any) -> None:
        calls.append((value, kwargs))

    monkeypatch.setattr(jm_module, "render", fake_render)

    path = asyncio.run(jm_module.txt2img(text))

    assert path == f"/tmp/{digest}.png"
    assert calls == [
        (text, {"theme": "dark", "out": f"/tmp/{digest}.png"}),
    ]


@pytest.mark.parametrize("error", [OSError("x"), ValueError("x"), TypeError("x")])
def test_txt2img_returns_empty_string_when_render_fails(monkeypatch, error) -> None:
    monkeypatch.setattr(jm_module.os.path, "exists", lambda path: False)

    def fake_render(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(jm_module, "render", fake_render)

    assert asyncio.run(jm_module.txt2img("boom")) == ""


# ---------- 下载 ----------


def test_comic_download_returns_existing_pdf(monkeypatch) -> None:
    monkeypatch.setattr(jm_module, "get_domain", _async_noop)
    monkeypatch.setattr(jm_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        jm_module.jmcomic,
        "download_album_async",
        pytest.fail,
    )

    assert asyncio.run(jm_module.comic_download(123)) == (True, "down/123.pdf")


def test_comic_download_downloads_and_returns_pdf(monkeypatch) -> None:
    calls: list[tuple[int, Any]] = []
    exists = {"value": False}
    monkeypatch.setattr(jm_module, "get_domain", _async_noop)
    monkeypatch.setattr(jm_module.os.path, "exists", lambda path: exists["value"])

    async def fake_download(album_id: int, **kwargs: Any) -> None:
        calls.append((album_id, kwargs))
        exists["value"] = True

    monkeypatch.setattr(jm_module.jmcomic, "download_album_async", fake_download)

    assert asyncio.run(jm_module.comic_download(123)) == (True, "down/123.pdf")
    assert calls[0][0] == 123
    extra = calls[0][1]["extra"]
    assert extra.plugin_key == "img2pdf"
    assert extra.kwargs == {
        "pdf_dir": "down",
        "delete_original_file": True,
        "filename_rule": "{Aid}",
    }


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (jmcomic.MissingAlbumPhotoException("x", {}), "呜喵…123 没找到喵～"),
        (jmcomic.PartialDownloadFailedException("x", {}), "呜喵…下载失败了呢喵～"),
        (jmcomic.JmcomicException("x", {}), "呜喵…JM 出错了喵～"),
    ],
)
def test_comic_download_maps_exceptions(monkeypatch, error, message) -> None:
    monkeypatch.setattr(jm_module, "get_domain", _async_noop)

    async def fake_download(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(jm_module.jmcomic, "download_album_async", fake_download)

    assert asyncio.run(jm_module.comic_download(123)) == (False, message)


def test_comic_download_reports_missing_output(monkeypatch) -> None:
    monkeypatch.setattr(jm_module, "get_domain", _async_noop)
    monkeypatch.setattr(jm_module.os.path, "exists", lambda path: False)

    async def fake_download(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(jm_module.jmcomic, "download_album_async", fake_download)

    assert asyncio.run(jm_module.comic_download(123)) == (
        False,
        "呜喵…保存失败了喵…",
    )


# ---------- 详情 ----------


def _detail() -> SimpleNamespace:
    return SimpleNamespace(
        album_id=123,
        name="测试漫画",
        authors=["作者A", "作者B"],
        actors=[],
        tags=["标签"],
        episode_list=["第1话", "第2话"],
        page_count=20,
        pub_date="2024-01-01",
        update_date="2024-02-01",
        likes=7,
        views=99,
        comment_count=3,
        description=None,
    )


class _Client:
    def __init__(self, detail: Any = None, error: Exception | None = None) -> None:
        self.detail = detail
        self.error = error

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def get_album_detail(self, album_id: int) -> Any:
        assert album_id == 123
        if self.error is not None:
            raise self.error
        return self.detail


class _Option:
    def __init__(self, client: _Client) -> None:
        self.client = client

    def new_jm_async_client(self) -> _Client:
        return self.client


def _patch_detail(monkeypatch, client: _Client) -> None:
    monkeypatch.setattr(jm_module, "get_domain", _async_noop)
    monkeypatch.setattr(
        jm_module.JmOption,
        "default",
        lambda: _Option(client),
    )


def test_comic_detail_formats_all_fields(monkeypatch) -> None:
    _patch_detail(monkeypatch, _Client(_detail()))

    ok, text = asyncio.run(jm_module.comic_detail(123))

    assert ok is True
    assert text.splitlines() == [
        "ID: 123",
        "名称: 测试漫画",
        "作者: 作者A、作者B",
        "角色: 无",
        "标签: 标签",
        "章节: 第1话、第2话",
        "页数: 20",
        "发布日期: 2024-01-01",
        "更新日期: 2024-02-01",
        "喜欢: 7    浏览: 99    评论: 3",
        "描述: 无",
    ]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (jmcomic.MissingAlbumPhotoException("x", {}), "呜喵…123 没找到喵～"),
        (jmcomic.JsonResolveFailException("x", {}), "呜喵…JSON 解析炸了喵"),
        (jmcomic.RequestRetryAllFailException("x", {}), "呜喵…请求都失败了喵～"),
        (jmcomic.JmcomicException("x", {}), "呜喵…详情抓取出错了喵"),
    ],
)
def test_comic_detail_maps_exceptions(monkeypatch, error, message) -> None:
    _patch_detail(monkeypatch, _Client(error=error))

    assert asyncio.run(jm_module.comic_detail(123)) == (False, message)


# ---------- 命令 ----------


def test_group_hello_replies() -> None:
    event = _Event("/hello")

    asyncio.run(_plugin().on_group_hello(event))  # type: ignore[arg-type]

    assert event.replies == ["hi 喵～"]


@pytest.mark.parametrize(
    ("handler_name", "text", "message"),
    [
        ("on_gourp_jm_download", "/jm", "呜喵…语法用错了啦～"),
        ("on_gourp_jm_download", "/jm abc", "ID 要整数的喵～"),
        ("on_gourp_jm_detail", "/jmd", "呜喵…语法用错了啦～"),
        ("on_gourp_jm_detail", "/jmd abc", "ID 要整数的喵～"),
    ],
)
def test_jm_commands_validate_arguments(handler_name, text, message) -> None:
    plugin = _plugin()
    event = _Event(text)

    asyncio.run(getattr(plugin, handler_name)(event))

    assert event.replies == [message]


def test_jm_download_replies_failure(monkeypatch) -> None:
    async def fake_download(album_id: int) -> tuple[bool, str]:
        return False, "下载失败"

    monkeypatch.setattr(jm_module, "comic_download", fake_download)
    plugin = _plugin()
    event = _Event("/jm 123")

    asyncio.run(plugin.on_gourp_jm_download(event))  # type: ignore[arg-type]

    assert event.replies == ["下载失败"]
    assert plugin.api.qq.files == []  # type: ignore[union-attr]


def test_jm_download_sends_file(monkeypatch) -> None:
    async def fake_download(album_id: int) -> tuple[bool, str]:
        return True, "down/123.pdf"

    monkeypatch.setattr(jm_module, "comic_download", fake_download)
    plugin = _plugin()
    event = _Event("/jm 123")

    asyncio.run(plugin.on_gourp_jm_download(event))  # type: ignore[arg-type]

    assert plugin.api.qq.files == [(123, "down/123.pdf", "123.pdf")]  # type: ignore[union-attr]
    assert event.replies == []


def test_jm_download_timeout_replies(monkeypatch) -> None:
    async def fake_download(album_id: int) -> tuple[bool, str]:
        raise TimeoutError

    monkeypatch.setattr(jm_module, "comic_download", fake_download)
    plugin = _plugin()
    event = _Event("/jm 123")

    asyncio.run(plugin.on_gourp_jm_download(event))  # type: ignore[arg-type]

    assert event.replies == ["呜喵…下载太慢了啦，等了好久都没好喵～"]


def test_jm_download_ignores_file_send_timeout(monkeypatch, capsys) -> None:
    async def fake_download(album_id: int) -> tuple[bool, str]:
        return True, "down/123.pdf"

    monkeypatch.setattr(jm_module, "comic_download", fake_download)
    plugin = _plugin()
    plugin.api.qq.file_error = TimeoutError()  # type: ignore[union-attr]
    event = _Event("/jm 123")

    asyncio.run(plugin.on_gourp_jm_download(event))  # type: ignore[arg-type]

    assert "文件发送超时" in capsys.readouterr().out
    assert event.replies == []


def test_jmd_detail_replies_text_for_extra_argument(monkeypatch) -> None:
    async def fake_detail(album_id: int) -> tuple[bool, str]:
        return True, "详情文本"

    monkeypatch.setattr(jm_module, "comic_detail", fake_detail)
    plugin = _plugin()
    event = _Event("/jmd 123 text")

    asyncio.run(plugin.on_gourp_jm_detail(event))  # type: ignore[arg-type]

    assert event.replies == ["详情文本"]
    assert plugin.api.qq.images == []  # type: ignore[union-attr]


def test_jmd_detail_sends_rendered_image(monkeypatch) -> None:
    async def fake_detail(album_id: int) -> tuple[bool, str]:
        return True, "详情文本"

    async def fake_txt2img(text: str) -> str:
        assert text == "详情文本"
        return "/tmp/detail.png"

    monkeypatch.setattr(jm_module, "comic_detail", fake_detail)
    monkeypatch.setattr(jm_module, "txt2img", fake_txt2img)
    plugin = _plugin()
    event = _Event("/jmd 123")

    asyncio.run(plugin.on_gourp_jm_detail(event))  # type: ignore[arg-type]

    assert plugin.api.qq.images == [(123, "/tmp/detail.png")]  # type: ignore[union-attr]
    assert event.replies == []


def test_jmd_detail_falls_back_to_text_when_render_fails(monkeypatch) -> None:
    async def fake_detail(album_id: int) -> tuple[bool, str]:
        return True, "详情文本"

    async def fake_txt2img(text: str) -> str:
        return ""

    monkeypatch.setattr(jm_module, "comic_detail", fake_detail)
    monkeypatch.setattr(jm_module, "txt2img", fake_txt2img)
    plugin = _plugin()
    event = _Event("/jmd 123")

    asyncio.run(plugin.on_gourp_jm_detail(event))  # type: ignore[arg-type]

    assert event.replies == ["详情文本"]


def test_jmd_detail_falls_back_to_text_when_send_raises(monkeypatch) -> None:
    async def fake_detail(album_id: int) -> tuple[bool, str]:
        return True, "详情文本"

    async def fake_txt2img(text: str) -> str:
        return "/tmp/detail.png"

    monkeypatch.setattr(jm_module, "comic_detail", fake_detail)
    monkeypatch.setattr(jm_module, "txt2img", fake_txt2img)
    plugin = _plugin()

    async def broken_image(group_id: int, path: str) -> None:
        raise OSError("send failed")

    plugin.api.qq.send_group_image = broken_image  # type: ignore[union-attr,method-assign]
    event = _Event("/jmd 123")

    asyncio.run(plugin.on_gourp_jm_detail(event))  # type: ignore[arg-type]

    assert event.replies == ["详情文本"]


def test_jmd_detail_replies_failure(monkeypatch) -> None:
    async def fake_detail(album_id: int) -> tuple[bool, str]:
        return False, "详情失败"

    monkeypatch.setattr(jm_module, "comic_detail", fake_detail)
    plugin = _plugin()
    event = _Event("/jmd 123")

    asyncio.run(plugin.on_gourp_jm_detail(event))  # type: ignore[arg-type]

    assert event.replies == ["详情失败"]


def test_jmd_detail_timeout_replies(monkeypatch) -> None:
    async def fake_detail(album_id: int) -> tuple[bool, str]:
        raise TimeoutError

    monkeypatch.setattr(jm_module, "comic_detail", fake_detail)
    plugin = _plugin()
    event = _Event("/jmd 123")

    asyncio.run(plugin.on_gourp_jm_detail(event))  # type: ignore[arg-type]

    assert event.replies == ["呜喵…详情拉取太慢了喵～"]


async def _async_noop(*args: Any, **kwargs: Any) -> None:
    return None
