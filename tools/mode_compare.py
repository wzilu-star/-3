# -*- coding: utf-8 -*-
"""三种检索模式的召回对比（《结论_一页》第六节用）。

为什么单独做这个：把 10 道题在「仅向量 / 仅 BM25 / 混合」下各跑一遍、
再逐题人工判对错，要 3×10 次生成，CPU 上得跑一个多小时。
但第六节真正想说明的是**召回环节**的差异，跟生成模型无关。
所以这里只跑检索、不跑生成，用两个可自动计算的口径打分：

    ① 覆盖公司数 —— 全景题尤其关键：召回里出现了几家公司的块
    ② 命中数值   —— 召回的块里是否真的出现了金标准数字（字符串匹配）
                    数字是年报原文写法（如 170,899,152,276.34），
                    只要有一块包含它，就说明「答案所需证据已经进了上下文」

口径②比「答案对不对」严格得多也宽松得多，读的时候注意：
    命中 ≠ 答对（模型可能算错/忽略），未命中 ⇒ 一定答不对（证据根本没进来）。

用法：
    python tools/mode_compare.py
产物：
    eval/mode_compare.md
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, resolve  # noqa: E402
from importlib import import_module  # noqa: E402

Rag = import_module("05_rag").Retriever

# 金标准数值串：直接取自年报原文（data/extracted/blocks.jsonl），
# 多个写法则任意命中一个即算命中。
GOLD = {
    "Q01": ["170,899,152,276.34", "1708.99"],
    "Q02": ["134.73", "13,472"],
    "Q03": ["12,172,323,337.21", "121.72"],
    "Q04": ["145,928,075,955.31", "1,459.28", "94.06", "24,683,762,096.71"],
    "Q05": ["分地区", "省外", "省内"],
    "Q06": ["89,175", "891.75", "7.09"],
    "Q07": ["销售费用", "3,726", "销售费用率"],
    "Q08": ["94.06", "92.01", "91.9"],
    "Q09": ["-12.83", "-24.41", "12.83", "24.41"],
    "Q10": ["170,899,152,276.34", "89,175", "36,010,992,321.46"],
}

MODES = [
    ("仅向量", {"bm25_top_k": 0}),
    ("仅 BM25", {"vector_top_k": 0}),
    ("混合（RRF）", {}),
]


def main() -> None:
    cfg0 = load_config()
    qpath = resolve(cfg0["paths"]["eval_dir"] + "/questions.yaml")
    questions = yaml.safe_load(qpath.read_text(encoding="utf-8"))["questions"]

    lines = ["# 三种检索模式的召回对比", "",
             "> 只跑检索、不跑生成；口径见 `tools/mode_compare.py` 文件头注释。",
             "> 「命中数值」= 送进模型的片段里出现了年报原文的关键数字。", ""]
    table = ["| 题号 | 类型 | 仅向量 覆盖/命中 | 仅 BM25 覆盖/命中 | 混合 覆盖/命中 |",
             "|---|---|---|---|---|"]

    rags = {}
    for mname, override in MODES:
        cfg = copy.deepcopy(cfg0)
        cfg["retrieve"].update(override)
        cfg["generate"]["provider"] = "none"      # 不生成，只要检索结果
        rags[mname] = Rag(cfg)
        print(f"[就绪] {mname}")

    for q in questions:
        qid = q["id"]
        cells = []
        for mname, _ in MODES:
            out = rags[mname].search(q["question"])
            hits = out["hits"]
            comps = {h["company"] for h in hits}
            blob = "\n".join(h["content"] for h in hits)
            hit = any(g in blob for g in GOLD.get(qid, []))
            n_comp = len(comps)
            mark = "✅" if hit else "❌"
            cells.append(f"{n_comp} 家 / {mark}")
            print(f"  {qid} {mname:<10} 块={len(hits):<3} 覆盖={n_comp:<3} 命中={hit}")
        table.append(f"| {qid} | {q['type']} | " + " | ".join(cells) + " |")

    lines += table
    lines += ["", "## 逐题召回明细", ""]
    for q in questions:
        lines += [f"### {q['id']} 【{q['type']}】{q['question']}", ""]
        for mname, _ in MODES:
            out = rags[mname].search(q["question"])
            hits = out["hits"]
            comps = sorted({h["company"] for h in hits})
            blob = "\n".join(h["content"] for h in hits)
            hit = any(g in blob for g in GOLD.get(qid, []))
            recalled = "；".join(f"{h['chunk_id']}" for h in hits) or "（无召回）"
            lines += [f"- **{mname}**：命中数值 {'✅' if hit else '❌'}；"
                      f"覆盖 {len(comps)} 家（{'、'.join(comps)}）",
                      f"  - 召回：{recalled}"]
        lines.append("")

    out_path = resolve(cfg0["paths"]["eval_dir"]) / "mode_compare.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
