# text2img — 纯文本 → 多语言 PNG（纯 Pillow，Termux 友好）

把任意纯文本（多语言混排）转成 PNG 图片。**零浏览器依赖**，**零外部 CLI 依赖**，只有 Pillow。

适用场景：
- Termux / Android 上的笔记 → 图片分享
- 服务器无 GUI 环境批量生成图片
- 不想装 Chromium / wkhtmltopdf 等大块头

## 安装

```bash
pip install Pillow
```

把 `text2img/` 目录扔到你的项目里，或者 `pip install -e .` 本地装。

## Termux 上的字体准备

**必须装**，否则中文/日文/阿拉伯文会显示成方框：

```bash
pkg update
pkg install noto-fonts          # Noto Sans/Serif CJK + 拉丁/西里尔
pkg install fontconfig-utils    # 提供 fc-list（可选，没装也能 glob 兜底）
```

可选字体包（覆盖更多语言）：
```bash
pkg install noto-fonts-arabic-hebrew-thai   # 阿拉伯/希伯来/泰文（具体包名以 pkg 实际为准）
```

## 快速使用

### 命令行

```bash
# 直接给文本
python -m text2img "你好世界" -o out.png

# 从文件
python -m text2img -f note.txt -t dark -o out.png

# 从 stdin
echo "Hello 你好 こんにちは" | python -m text2img -t paper -o out.png

# 调宽度/字号
python -m text2img -f note.txt -t modern -w 1080 --font-size 24 -o out.png

# 看主题列表
python -m text2img --list-themes

# 看探测到的字体
python -m text2img --show-fonts
```

### Python API

```python
from text2img import render

render("你好世界", theme="modern", out="hi.png")

# 调字号 / 行高 / 宽度
render(text, theme="dark", width=1080, font_size=24, line_height=1.8, out="out.png")
```

## 主题

| 名称 | 风格 | 字体 |
|------|------|------|
| `modern` | 白底黑字，干净 | sans |
| `literary` | 米色衬线，文学感 | serif |
| `dark` | 深色科技感 | sans |
| `paper` | 暖色复古纸张 | serif |
| `memo` | 黄色便利贴 | sans |
| `terminal` | 终端绿字黑底 | mono |
| `ocean` | 深海蓝渐变 | serif |

## 多语言支持

每个字符按 Unicode 脚本选字体：

| 脚本 | 字体（默认 Termux `pkg install noto-fonts`） |
|------|-------|
| 中文（简繁）/ 日文 / 韩文 | Noto Sans CJK |
| 拉丁 / 西里尔 / 希腊 | Noto Sans |
| 阿拉伯 | Noto Sans Arabic |
| 希伯来 | Noto Sans Hebrew |
| 泰文 | Noto Sans Thai |
| 越南文（声调） | Noto Sans Latin Extended |
| 天城文（印地语等） | ❌ 默认不装，需 `noto-fonts-devanagari` |

## 不支持的内容

按设计**主动不处理**（避免给纯文本场景增加复杂度）：

- ❌ **Emoji** —— 跳过。Noto Color Emoji 在 PIL 上需要固定 size=109 + embedded_color=True 的特殊路径，且渲染效果远不如 Chromium。如果要 emoji，请改用 `text2png`（基于 Chromium）。
- ❌ **数学符号** —— 不优化。如需要 ∑ ∫ ₀ ² ∞ √ 等符号，请装 Noto Sans Math + Noto Sans Symbols 后改代码。
- ❌ **Markdown 语法** —— 按纯文本处理，`*` `_` `#` 等都是字面字符。要 Markdown 渲染请改用 `text2png`。

## 性能

桌面 x86 上单张 900×700 图约 **0.3-0.5 秒**。Termux (ARM64) 上预计 **2-5 倍** 慢，渲染一张 5-15 秒，能用但不实时。

## 架构

```
text2img/
├── __init__.py    # 公开 API
├── __main__.py    # python -m text2img 入口
├── cli.py         # 命令行
├── engine.py      # 渲染核心（wrap / layout / draw）
├── fonts.py       # 字体探测 + 字符级字体选择
└── styles.py      # 主题定义
```

字体探测策略：
1. `fc-list :lang=xx file` 拿路径（按语言拿更精确）
2. 失败时 glob `/system/fonts` + `/usr/share/fonts`
3. 按文件名分类（CJK / sans / serif / mono / arabic / hebrew / thai / cyrillic / emoji）
4. 缓存到 `_FS_BY_ID` 避免重复探测

## 已知限制 / 坑

- **字体 fallback**：如果某个字符找不到专门字体，会落到 `sans`（Noto Sans）—— Noto Sans 不含 CJK / 阿拉伯 / 希伯来 / 泰文，所以**一定要装对应字体包**，否则那些字符会显示成方框。
- **emoji 方框**：见上。
- **天城文（印地等）方框**：Termux 默认 `noto-fonts` 不含天城文字体，需手动装或忽略。
- **CJK 标点（如 「」 ￥）**：会自动走 CJK 字体，OK。
- **超长 CJK 串**：会逐字强制换行（不会拆英文单词）。
- **换行算法**：贪心 + 单词级不拆，CJK 字符级可拆。

## 调试

```bash
# 看探测到的字体（确认 CJK/Arabic/Thai 等都有）
python -m text2img --show-fonts

# 应该看到类似输出：
# FontSet:
#   ✓ sans       /usr/share/fonts/noto/NotoSans-Regular.ttf
#   ✓ cjk        /usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc
#   ✓ cjk_sc     ...
#   ✓ arabic     ...
```

如果某项是 `✗` 就是字体没装到。