# -*- coding: utf-8 -*-
"""
把 Markdown 报告导成 PDF（给「结论_一页.md」用）

用法：
    python tools/md_to_pdf.py eval/结论_一页.md
    python tools/md_to_pdf.py eval/结论_一页.md --out eval/结论_一页.pdf

为什么不用 weasyprint / wkhtmltopdf：
  两个在 Windows 上都要另外装系统组件（GTK / 独立 exe），评审机器上大概率没有。
  浏览器是每台机器都有的，直接借 Playwright 驱动**系统自带的 Edge**（channel=msedge）
  打印成 PDF —— 不下载 Chromium，也不用管理员权限，中文和表格由 Edge 自己排版，
  不会出现「中文变方块」这种缺字体的坑。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A4 + 中文正文样式。字体栈按 Windows / macOS / Linux 依次回退。
CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
               "Source Han Sans SC", "SimSun", sans-serif;
  font-size: 10.5pt; line-height: 1.65; color: #1a1a1a; margin: 0;
}
h1 { font-size: 18pt; margin: 0 0 4px; padding-bottom: 8px;
     border-bottom: 2px solid #1a1a1a; }
h2 { font-size: 13pt; margin: 18px 0 8px; padding-left: 8px;
     border-left: 4px solid #c0392b; }
h3 { font-size: 11.5pt; margin: 14px 0 6px; color: #333; }
p, li { margin: 5px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 4px 7px; text-align: left; }
th { background: #f2f2f2; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt;
       background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 8px 10px; border-radius: 4px;
      font-size: 8.5pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { margin: 8px 0; padding: 6px 12px; color: #555;
             border-left: 3px solid #ccc; background: #fafafa; }
hr { border: none; border-top: 1px solid #ddd; margin: 14px 0; }
h2, h3, table, pre { page-break-inside: avoid; }
"""


def md_to_html(md_text: str, title: str) -> str:
    try:
        import markdown
    except ImportError:
        sys.exit("缺少 markdown 包：pip install markdown")
    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>{title}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("缺少 playwright：pip install playwright")
    with sync_playwright() as p:
        # 用系统自带 Edge，不下载 Chromium（国内下 Chromium 常卡）
        try:
            browser = p.chromium.launch(channel="msedge", headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="load")
        page.pdf(path=str(pdf_path), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm",
                         "left": "14mm", "right": "14mm"})
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown → PDF（借系统 Edge 排版）")
    ap.add_argument("src", help="输入 .md")
    ap.add_argument("--out", default=None, help="输出 .pdf（默认同目录同名）")
    ap.add_argument("--keep-html", action="store_true", help="保留中间 HTML")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        sys.exit(f"找不到输入文件：{src}")

    pdf_path = Path(args.out) if args.out else src.with_suffix(".pdf")
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_html = pdf_path.with_suffix(".html")
    tmp_html.write_text(
        md_to_html(src.read_text(encoding="utf-8"), src.stem), encoding="utf-8")
    html_to_pdf(tmp_html, pdf_path)
    if not args.keep_html:
        tmp_html.unlink(missing_ok=True)
    print(f"[完成] {src.name} → {pdf_path}  ({pdf_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
