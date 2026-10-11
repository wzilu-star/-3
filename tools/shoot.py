# -*- coding: utf-8 -*-
"""给 Streamlit 问答页面自动截图（交付物②：页面截图）。

为什么要写成脚本，而不是手动截图：
    1) Streamlit 是 JS 单页应用，答案要等「检索 + 生成」跑完才出现。
       命令行 `msedge --headless --screenshot` 在页面 load 事件时就按快门，
       拍到的永远是「正在加载索引…」。
    2) `generate.provider=local` 时一道全景题要跑几分钟，无头浏览器默认早超时了。
    所以这里用 Playwright：一直等到页面上真的出现「引用来源」再拍。

为什么不用 `playwright install`：
    直接用系统已装的 Edge（channel="msedge"），不额外下载一份 Chromium。

用法：
    先起服务：streamlit run app/streamlit_app.py --server.headless true
    再截图：  python tools/shoot.py
             python tools/shoot.py --set 2          # 只拍第 2 组
             python tools/shoot.py --q "自定义问题" --out docs/screenshots/x.png

产物：docs/screenshots/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent

# (输出文件名, 问题, 是否展开「检索过程」)
SHOTS = [
    ("01_单公司_营业收入与同比.png",
     "贵州茅台2024年实现营业收入是多少？同比增速是多少？", False),
    ("02_跨公司全景_毛利率最高.png",
     "2024年这12家白酒公司中，哪一家的毛利率最高？它的毛利率是多少？", True),
    ("03_跨公司全景_营收下滑.png",
     "2024年哪些白酒公司的营业收入出现了同比下滑？", False),
    ("04_表格多指标_茅台分产品.png",
     "贵州茅台2024年分产品（茅台酒、系列酒）的营业收入和毛利率分别是多少？", False),
]


def shoot(url_base: str, question: str, out: Path, expand: bool,
          wait_ms: int, channel: str) -> None:
    from playwright.sync_api import sync_playwright

    out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        ctx = browser.new_context(
            viewport={"width": 1500, "height": 1100},
            device_scale_factor=2,          # 2 倍图，页面上的小字截图后仍清晰
            locale="zh-CN",
        )
        page = ctx.new_page()
        page.set_default_timeout(wait_ms)

        url = f"{url_base}/?q={quote(question)}"
        print(f"[打开] {url}")
        page.goto(url, wait_until="domcontentloaded")

        print(f"[等待] 等「引用来源」出现（local 生成较慢，最长等 {wait_ms // 1000}s）...")
        page.wait_for_selector("text=引用来源", timeout=wait_ms)
        # 生成结束后 spinner 会消失；再等一拍，让 Markdown 表格渲染完
        try:
            page.wait_for_selector("[data-testid='stSpinner']", state="detached", timeout=20_000)
        except Exception:
            pass
        page.wait_for_timeout(2500)

        if expand:
            print("[展开] 检索过程")
            try:
                page.click("text=检索过程（向量 / BM25 / 融合）")
                page.wait_for_timeout(1200)
            except Exception as e:
                print(f"       （展开失败，跳过：{e}）")

        page.screenshot(path=str(out), full_page=True)
        print(f"[完成] {out}")
        ctx.close()
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge", help="msedge / chrome")
    ap.add_argument("--wait", type=int, default=900_000, help="单题最长等待毫秒数")
    ap.add_argument("--set", type=int, help="只拍第 N 组（从 1 开始）")
    ap.add_argument("--q", help="自定义问题（与 --out 配合）")
    ap.add_argument("--out", help="自定义输出路径")
    args = ap.parse_args()

    if args.q:
        if not args.out:
            sys.exit("--q 需要同时给 --out")
        shoot(args.url, args.q, ROOT / args.out, False, args.wait, args.channel)
        return

    todo = SHOTS
    if args.set:
        todo = [SHOTS[args.set - 1]]

    for name, q, expand in todo:
        print(f"\n=== {name} ===")
        try:
            shoot(args.url, q, ROOT / "docs" / "screenshots" / name, expand,
                  args.wait, args.channel)
        except Exception as e:
            print(f"[失败] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
