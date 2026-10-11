# -*- coding: utf-8 -*-
"""
02 · 抽取文字与表格（表格按行列还原）

用法：
    python src/02_extract.py
    python src/02_extract.py --only 贵州茅台

产出：data/extracted/blocks.jsonl
每条记录 = 一个「块」，已携带完整元数据：
    company / code / report_period / report_type / section / page /
    block_type(text|table) / content / unit / table_title / source_file

关键实现点：
  1) 用 pdfplumber 逐页抽文字 + extract_tables()，表格还原为 Markdown（行列不丢）
  2) 正则识别「第X节」标题，作为最近章节标记 section
  3) 剔除跨页重复的页眉页脚
  4) 跨页表格自动补回表头
  5) 识别「单位：万元/元」写入 unit，并在块文本里显式标明
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pdfplumber
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (detect_unit, drop_repeated_lines, ensure_dir, load_config,  # noqa: E402
                   read_jsonl, resolve, table_to_markdown, write_jsonl)

TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 8,
}


def extract_one_pdf(pdf_path: Path, meta: dict, cfg: dict) -> list[dict]:
    section_re = re.compile(cfg["extract"]["section_regex"])
    blocks: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        # ---- 第一遍：拿每页文字，用于识别并剔除重复页眉页脚 ----
        page_texts, page_tables = [], []
        for page in pdf.pages:
            txt = page.extract_text() or ""
            page_texts.append(txt)
            # 表格抽取比文字慢，但年报表格是主体信息，必须做
            try:
                tbls = page.extract_tables(table_settings=TABLE_SETTINGS) or []
            except Exception:
                tbls = []
            page_tables.append(tbls)

        if cfg["extract"]["drop_repeated_lines"]:
            page_texts = drop_repeated_lines(page_texts, cfg["extract"]["repeated_ratio"])

        current_section = ""
        prev_table_header: list[str] | None = None

        for i in range(n_pages):
            page_no = i + 1
            raw_text = page_texts[i]

            # ---- 章节识别：取本页最后出现的「第X节」作为当前章节 ----
            sections = section_re.findall(raw_text)
            if sections:
                current_section = sections[-1].strip()

            # 去掉表格区域占用的文字行不做严格处理（保持简单：文字与表格都存）
            text_body = raw_text.strip()
            # 单位识别（作用于本页上下文）
            unit = detect_unit(text_body[:2000]) if text_body else "元"

            # ---- 1) 文本块：按段落给，切块留给 03 步 ----
            if text_body:
                blocks.append({
                    **meta,
                    "section": current_section,
                    "page": page_no,
                    "block_type": "text",
                    "content": text_body,
                    "unit": unit,
                })

            # ---- 2) 表格块：还原为 Markdown，整表不切 ----
            for t_idx, table in enumerate(page_tables[i] or []):
                if not table or len(table) < 2:
                    continue
                rows = [[c for c in row] for row in table]
                header = [str(c).strip() if c else "" for c in rows[0]]

                # 跨页续表：首行不含数字、且与上页表头相似 → 认为是续表，补上页表头
                first_row_text = " ".join(header)
                looks_like_continuation = (
                    prev_table_header is not None
                    and not re.search(r"\d", first_row_text)
                    and sum(1 for h in header if h and h in prev_table_header) >= max(1, len(header) - 1)
                )
                if looks_like_continuation:
                    md_header = prev_table_header
                else:
                    md_header = header
                    prev_table_header = header

                md = table_to_markdown(rows, header=md_header)
                if not md:
                    continue

                # 表格标题：向上找最近的短行作为表名（如「主营业务分行业情况」）
                title = ""
                for ln in reversed(text_body.splitlines()[-25:]):
                    ln = ln.strip()
                    if 2 < len(ln) < 30 and not re.search(r"\d", ln) and "单位" not in ln:
                        title = ln
                        break

                content = md
                if unit != "元":
                    content = f"(本表金额单位：{unit})\n" + md
                else:
                    content = "(本表金额单位：元)\n" + md

                blocks.append({
                    **meta,
                    "section": current_section,
                    "page": page_no,
                    "block_type": "table",
                    "table_title": title,
                    "table_index": t_idx,
                    "content": content,
                    "unit": unit,
                    "rows": len(rows),
                })

    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    cfg = load_config()
    period = str(cfg["project"]["period"])
    rtype = cfg["project"]["report_type"]
    raw_dir = resolve(cfg["paths"]["raw_dir"])

    companies = cfg["companies"]
    if args.only:
        companies = [c for c in companies if args.only in c["name"]]

    all_blocks: list[dict] = []
    stats: list[tuple[str, int, int, int]] = []

    for c in tqdm(companies, desc="抽取"):
        pdf_path = raw_dir / f"{c['name']}_{c['code']}" / f"{period}_{rtype}.pdf"
        if not pdf_path.exists():
            print(f"  [跳过] {c['name']}：未找到 {pdf_path}")
            continue

        meta = {
            "company": c["name"], "code": c["code"],
            "report_period": period, "report_type": rtype,
            "source_file": str(pdf_path).replace("\\", "/"),
        }
        blocks = extract_one_pdf(pdf_path, meta, cfg)
        all_blocks.extend(blocks)

        n_text = sum(1 for b in blocks if b["block_type"] == "text")
        n_table = sum(1 for b in blocks if b["block_type"] == "table")
        stats.append((c["name"], len(blocks), n_text, n_table))

    out = resolve(cfg["paths"]["extracted_dir"]) / "blocks.jsonl"
    n = write_jsonl(out, all_blocks)

    print("\n=== 抽取自检 ===")
    print(f"{'公司':<10}{'总块数':>8}{'文本块':>8}{'表格块':>8}")
    for name, total, nt, nb in stats:
        flag = "  ⚠ 表格为0，请检查" if nb == 0 else ""
        print(f"{name:<10}{total:>8}{nt:>8}{nb:>8}{flag}")
    print(f"\n合计 {n} 块 → {out}")


if __name__ == "__main__":
    main()
