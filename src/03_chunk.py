# -*- coding: utf-8 -*-
"""
03 · 切块：文本按语义边界切 + 块首注入上下文头；表格整表成块

用法：
    python src/03_chunk.py

产出：data/chunks/chunks.jsonl
每个 chunk 都带：
    chunk_id / company / code / report_period / report_type /
    section / page / block_type / content / unit / text_for_index

text_for_index = 上下文头 + 正文（向量与 BM25 都索引这个字段）
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import context_header, load_config, read_jsonl, resolve, write_jsonl  # noqa: E402

# 语义边界：章节小标题、段落末尾标点、列表项开头
SENT_END = re.compile(r"(?<=[。；！？])")
SUB_HEAD = re.compile(r"^\s*(?:[（(][一二三四五六七八九十]+[)）]|[一二三四五六七八九十]+、|\d+[、.])\s*\S")


def split_text_semantic(text: str, min_chars: int, max_chars: int, overlap: int) -> list[str]:
    """句子级聚合切块：先切句，再按长度聚合成块，块间保留整句 overlap。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    units: list[str] = []
    buf = ""
    for ln in lines:
        if SUB_HEAD.match(ln) and buf:      # 遇到小标题就把当前缓冲收掉
            units.append(buf)
            buf = ln
        else:
            buf = (buf + ln) if buf else ln
    if buf:
        units.append(buf)

    # 把超长 unit 再按句末切开
    sentences: list[str] = []
    for u in units:
        if len(u) <= max_chars:
            sentences.append(u)
            continue
        sentences.extend(p for p in SENT_END.split(u) if p.strip())

    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) <= max_chars:
            cur += s
        else:
            if len(cur) >= min_chars or len(cur) + len(s) > max_chars * 1.5:
                chunks.append(cur)
                # overlap：带上当前块末尾若干字，保证跨块语义连续
                tail = cur[-overlap:] if overlap > 0 else ""
                cur = tail + s
            else:
                cur += s
    if cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if len(c.strip()) >= 20]


def split_table(content: str, max_rows: int) -> list[str]:
    """表格：不超过 max_rows 行整表成块；超过则按行拆，且每片重复表头。"""
    lines = content.splitlines()
    header_lines, body_lines = [], []
    for ln in lines:
        if ln.startswith("|") and not body_lines and (set(ln) <= set("|-: ")):
            header_lines.append(ln)
        elif ln.startswith("|") and len(header_lines) < 2:
            header_lines.append(ln)
        else:
            body_lines.append(ln)

    if len(body_lines) <= max_rows:
        return [content]

    head = "\n".join(header_lines)
    out, buf = [], []
    for ln in body_lines:
        buf.append(ln)
        if len(buf) >= max_rows:
            out.append(head + "\n" + "\n".join(buf))
            buf = []
    if buf:
        out.append(head + "\n" + "\n".join(buf))
    return out


def main() -> None:
    cfg = load_config()
    ck = cfg["chunk"]
    blocks_path = resolve(cfg["paths"]["extracted_dir"]) / "blocks.jsonl"
    if not blocks_path.exists():
        sys.exit(f"未找到 {blocks_path}，请先运行 02_extract.py")

    chunks: list[dict] = []
    counter: dict[str, int] = defaultdict(int)
    stat = {"text_chunks": 0, "table_chunks": 0}

    for blk in tqdm(read_jsonl(blocks_path), desc="切块"):
        company_slug = f"{blk['company']}{blk['code']}"
        pieces: list[tuple[str, str]] = []   # (block_type, content)

        if blk["block_type"] == "table":
            for p in split_table(blk["content"], ck["table_max_rows"]):
                pieces.append(("table", p))
        else:
            for p in split_text_semantic(blk["content"], ck["text_min_chars"],
                                         ck["text_max_chars"], ck["overlap_chars"]):
                pieces.append(("text", p))

        for btype, content in pieces:
            counter[company_slug] += 1
            chunk_id = f"{company_slug}-P{blk['page']}-{counter[company_slug]:04d}"
            head = context_header({**blk, "block_type": btype}) if ck["inject_context_header"] else ""
            chunks.append({
                "chunk_id": chunk_id,
                "company": blk["company"],
                "code": blk["code"],
                "report_period": blk["report_period"],
                "report_type": blk["report_type"],
                "section": blk.get("section", ""),
                "page": blk["page"],
                "block_type": btype,
                "table_title": blk.get("table_title", ""),
                "unit": blk.get("unit", "元"),
                "has_number": bool(re.search(r"[\d.]+%|[\d,]{4,}|同比|增长|下降", content)),
                "content": content,
                "text_for_index": (head + "\n" + content).strip(),
                "source_file": blk.get("source_file", ""),
            })
            stat[f"{btype}_chunks"] += 1

    out = resolve(cfg["paths"]["chunks_path"])
    n = write_jsonl(out, chunks)

    print(f"\n=== 切块结果 ===")
    print(f"文本块 {stat['text_chunks']} | 表格块 {stat['table_chunks']} | 合计 {n}")
    print(f"覆盖公司 {len(counter)} 家")
    print("\n各公司块数：")
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"  {k:<18}{v:>6}")
    avg = sum(len(c["text_for_index"]) for c in chunks) / max(1, n)
    print(f"\n平均块长度 {avg:.0f} 字 → {out}")


if __name__ == "__main__":
    main()
