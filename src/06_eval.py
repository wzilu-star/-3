# -*- coding: utf-8 -*-
"""
06 · 批量评测：跑 eval/questions.yaml 里的题目，输出结果表

用法：
    python src/06_eval.py                       # 用问题里的默认检索模式
    python src/06_eval.py --questions eval/questions.yaml

产出：
    eval/results_<时间戳>.md   每题：召回块 / 答案 / 耗时，并留出人工填写「对错 / 错因」的列

设计说明：
  · 「对错」与「错因」不自动判定 —— 财报数字题的自动判分不可靠，
    交给人工填，但格式固定（错因从枚举里选），便于最终统计。
  · 结果表里完整列出召回的 chunk_id 与「公司·页码」，而不是只写"召回3条"。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, resolve  # noqa: E402
from importlib import import_module  # noqa: E402

Rag = import_module("05_rag").Retriever

ERROR_TYPES = ["未召回", "召回但被忽略", "表格切碎", "单位口径混淆",
               "跨公司未聚合", "检索被单公司霸榜", "数字算错", "—"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    qpath = resolve(args.questions or (cfg["paths"]["eval_dir"] + "/questions.yaml"))
    questions = yaml.safe_load(qpath.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        questions = questions[:args.limit]

    print(f"载入 {len(questions)} 道题，开始检索 ...")
    rag = Rag(cfg)

    rows = []
    for q in questions:
        t0 = time.time()
        out = rag.answer(q["question"], args.top_k)
        cost = time.time() - t0

        recalled = "；".join(
            f"{h['chunk_id']}({h['company']}·P{h['page']})" for h in out["hits"]
        ) or "（无召回）"

        rows.append({
            "id": q.get("id", ""),
            "type": q.get("type", ""),
            "question": q["question"],
            "strategy": out["strategy"],
            "recalled": recalled,
            "answer": out["answer"].replace("\n", " ").strip(),
            "cost": f"{cost:.1f}s",
            "correct": q.get("expected_verdict", ""),
            "error_type": "",
            "note": "",
        })
        print(f"  ✔ {q.get('id','')} {q['question'][:32]}…  {cost:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = resolve(cfg["paths"]["eval_dir"]) / f"results_{ts}.md"

    lines = [
        f"# 评测结果 {ts}", "",
        f"- 题目数：{len(rows)}",
        f"- 检索模式：`{rows[0]['strategy'] if rows else '-'}`（可在页面侧边栏切换后重跑）",
        "",
        "> **人工填写栏**：`答案对错` 从 `✅正确 / ⚠️数字对但口径错 / ❌错 / 🚫拒答` 中选；",
        f"> `错误归因` 从 `{' / '.join(ERROR_TYPES)}` 中选。", "",
        "---", "",
    ]
    for r in rows:
        lines += [
            f"## {r['id']} 【{r['type']}】{r['question']}", "",
            f"- **检索策略**：{r['strategy']}",
            f"- **召回块**：{r['recalled']}",
            f"- **答案**：{r['answer']}",
            f"- **耗时**：{r['cost']}",
            f"- **答案对错**：{r['correct'] or '____'}",
            f"- **错误归因**：{r['error_type'] or '____'}",
            f"- **备注**：{r['note']}", "",
        ]

    # 附一张汇总表，方便直接统计
    lines += ["---", "", "## 汇总表", "",
              "| 题号 | 类型 | 问题 | 召回块数 | 耗时 | 对错 | 错因 |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        n = len([x for x in r["recalled"].split("；") if x and x != "（无召回）"])
        lines.append(f"| {r['id']} | {r['type']} | {r['question'][:40]} | {n} | "
                     f"{r['cost']} | {r['correct'] or ''} | {r['error_type'] or ''} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已写入 → {out_path}")
    print("请人工核对每题「答案对错」与「错误归因」，最后据此写「结论_一页.md」。")


if __name__ == "__main__":
    main()
