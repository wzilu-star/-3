# -*- coding: utf-8 -*-
"""
生成仿真示例数据（不需要下载任何 PDF）

用途：
    没装 faiss / sentence-transformers，或者还没下载年报时，
    先造一批形态与真实抽取结果完全一致的 chunks.jsonl，
    把「建索引 → 检索 → 页面」整条链路跑通、看清效果。

用法：
    python tools/make_demo_data.py                # 12 家 × 2023+2024
    python tools/make_demo_data.py --companies 3  # 只造 3 家（更快）
    python tools/make_demo_data.py --out demo/chunks/chunks.jsonl

生成的数据特点（与真实数据同构）：
    · 字段完全一致：chunk_id / company / code / section / page / block_type
                    / unit / has_number / content / text_for_index
    · 文本块首部带【公司·代码·报告期·章节·页码】上下文头
    · 表格块为 Markdown 表格，含营收/毛利率/分产品/分地区/现金流量表
    · 数字之间保持逻辑自洽（分产品之和 = 总营收、毛利率倒算毛利额等）

⚠️ 数据为**模拟生成**，仅用于演示与联调，不得用于任何真实结论。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import context_header, ensure_dir, resolve  # noqa: E402

# ---------------------------------------------------------------
# 12 家白酒公司的仿真基础数据（营收单位：亿元）
# 数值参考行业量级构造，仅用于演示
# ---------------------------------------------------------------
COMPANIES = [
    # name,      code,   2023营收, 2024营收, 毛利率,  产品结构[(名称, 占比)]
    ("贵州茅台", "600519", 1476.9, 1708.9, 91.9, [("茅台酒", 0.855), ("系列酒", 0.145)]),
    ("五粮液",   "000858",  832.7,  891.8, 75.4, [("五粮液产品", 0.78), ("其他酒产品", 0.22)]),
    ("泸州老窖", "000568",  302.3,  311.9, 88.2, [("中高档酒类", 0.89), ("其他酒类", 0.11)]),
    ("山西汾酒", "600809",  319.3,  360.1, 76.2, [("汾酒", 0.74), ("系列酒", 0.26)]),
    ("洋河股份", "002304",  331.3,  288.4, 73.1, [("中高档酒", 0.86), ("普通酒", 0.14)]),
    ("古井贡酒", "000596",  202.5,  235.8, 79.9, [("年份原浆", 0.79), ("古井贡酒", 0.21)]),
    ("今世缘",   "603369",  100.9,  115.4, 75.6, [("特A类", 0.80), ("A类及以下", 0.20)]),
    ("迎驾贡酒", "603198",   67.2,   73.4, 72.7, [("中高档白酒", 0.83), ("普通白酒", 0.17)]),
    ("口子窖",   "603589",   59.6,   60.9, 74.2, [("高档白酒", 0.84), ("中低档白酒", 0.16)]),
    ("舍得酒业", "600702",   70.8,   53.6, 66.4, [("中高档酒", 0.88), ("普通酒", 0.12)]),
    ("水井坊",   "600779",   49.5,   52.2, 82.8, [("高档", 0.90), ("中档", 0.10)]),
    ("老白干酒", "600559",   52.7,   55.9, 66.9, [("高档酒", 0.62), ("中档酒", 0.28)]),
]

REGIONS = [
    ("省内", 0.55), ("省外", 0.45),
]


def pct(a: float, b: float) -> float:
    """增长率（b 相对 a）。"""
    return (b - a) / a * 100 if a else 0.0


def fmt_yi(x: float) -> str:
    return f"{x:,.2f}"


def yi_to_yuan_str(x_yi: float) -> str:
    """亿元 -> 元，带千分位。"""
    return f"{x_yi * 1e8:,.0f}"


def build_chunks(years: list[int], limit: int | None) -> list[dict]:
    chunks: list[dict] = []
    seq: dict[str, int] = {}
    rnd = random.Random(20241010)

    companies = COMPANIES[:limit] if limit else COMPANIES

    for name, code, rev23, rev24, gm24, products in companies:
        rev_map = {2023: rev23, 2024: rev24}
        for year in years:
            rev = rev_map.get(year)
            if rev is None:                      # 2023 用 2022 反推
                rev = rev23 / (1 + rnd.uniform(0.06, 0.18))
            prev = rev / (1 + rnd.uniform(0.03, 0.20)) if year == 2023 else rev_map.get(2023, rev / 1.15)
            growth = pct(prev, rev)
            gm = gm24 + rnd.uniform(-1.2, 1.2) if year == 2024 else gm24 + rnd.uniform(-1.5, 1.0)
            gp = rev * gm / 100
            cost = rev - gp
            net = rev * rnd.uniform(0.28, 0.42)
            net_margin = net / rev * 100
            ocf = net * rnd.uniform(0.85, 1.35)
            sell_exp = rev * rnd.uniform(0.06, 0.18)
            sell_ratio = sell_exp / rev * 100
            total_assets = rev * rnd.uniform(1.6, 3.2)
            debt_ratio = rnd.uniform(18, 40)
            inventory = rev * rnd.uniform(0.35, 0.75)
            dealers = int(rnd.uniform(1200, 4500))
            roe = net / (total_assets * (1 - debt_ratio / 100)) * 100

            slug = f"{name}{code}"

            def add(section: str, page: int, block_type: str, content: str,
                    unit: str = "元", table_title: str = "") -> dict:
                seq[slug] = seq.get(slug, 0) + 1
                rec = {
                    "chunk_id": f"{slug}-P{page}-{seq[slug]:04d}",
                    "company": name, "code": code,
                    "report_period": str(year), "report_type": "年报",
                    "section": section, "page": page,
                    "block_type": block_type,
                    "table_title": table_title,
                    "unit": unit,
                    "has_number": bool(any(ch.isdigit() for ch in content)),
                    "content": content,
                    "text_for_index": "",
                    "source_file": f"data/raw/{slug}/{year}_年报.pdf",
                }
                rec["text_for_index"] = (context_header(rec) + "\n" + content).strip()
                chunks.append(rec)
                return rec

            # ---- 1) 主要会计数据（文本）----
            add("第二节 公司简介和主要财务指标", 6, "text",
                f"报告期内，公司实现营业收入{fmt_yi(rev)}亿元，同比增长{growth:.2f}%；"
                f"实现归属于上市公司股东的净利润{fmt_yi(net)}亿元，净利率{net_margin:.2f}%；"
                f"基本每股收益{rnd.uniform(0.8, 5.5):.2f}元。经营活动产生的现金流量净额为"
                f"{fmt_yi(ocf)}亿元，同比下降{rnd.uniform(0, 12):.2f}%。"
                f"截至报告期末，公司总资产{fmt_yi(total_assets)}亿元，资产负债率{debt_ratio:.2f}%。")

            # ---- 2) 主营业务分析（文本）----
            add("第三节 管理层讨论与分析", 12, "text",
                f"报告期内，公司围绕主业持续推进产品结构优化。"
                f"全年实现营业总收入{fmt_yi(rev)}亿元，营业成本{fmt_yi(cost)}亿元，"
                f"综合毛利率{gm:.2f}%，较上年同期{'提升' if gm > gm24 else '下降'}"
                f"{abs(gm - gm24):.2f}个百分点。"
                f"实现归属于上市公司股东的净利润{fmt_yi(net)}亿元，同比增长{pct(prev * net_margin / 100, net):.2f}%。"
                f"公司加权平均净资产收益率{roe:.2f}%，存货账面价值{fmt_yi(inventory)}亿元。"
                f"报告期末经销商数量{dealers}家。")

            # ---- 3) 分产品营收与毛利率（表格）----
            rows = ["| 分产品 | 营业收入(元) | 营业成本(元) | 毛利率(%) | 营收占比(%) |",
                    "|---|---|---|---|---|"]
            for pname, share in products:
                p_rev = rev * share
                p_gm = gm + rnd.uniform(-6, 4)
                p_cost = p_rev * (1 - p_gm / 100)
                rows.append(f"| {pname} | {yi_to_yuan_str(p_rev)} | {yi_to_yuan_str(p_cost)} "
                            f"| {p_gm:.2f} | {share * 100:.2f} |")
            rows.append(f"| 合计 | {yi_to_yuan_str(rev)} | {yi_to_yuan_str(cost)} "
                        f"| {gm:.2f} | 100.00 |")
            add("第三节 管理层讨论与分析", 16, "table",
                "(本表金额单位：元)\n" + "\n".join(rows),
                table_title="主营业务分产品情况")

            # ---- 4) 分地区（表格）----
            region_rows = ["| 分地区 | 营业收入(元) | 毛利率(%) |",
                           "|---|---|---|"]
            for rname, share in REGIONS:
                r_rev = rev * share
                r_gm = gm + rnd.uniform(-3, 3)
                region_rows.append(f"| {rname} | {yi_to_yuan_str(r_rev)} | {r_gm:.2f} |")
            add("第三节 管理层讨论与分析", 17, "table",
                "(本表金额单位：元)\n" + "\n".join(region_rows),
                table_title="主营业务分地区情况")

            # ---- 5) 现金流量表摘要（表格）----
            cash_rows = [
                "| 项目 | 本期金额(元) | 上期金额(元) |",
                "|---|---|---|",
                f"| 经营活动产生的现金流量净额 | {yi_to_yuan_str(ocf)} | {yi_to_yuan_str(ocf * rnd.uniform(0.9, 1.2))} |",
                f"| 投资活动产生的现金流量净额 | {yi_to_yuan_str(-rev * rnd.uniform(0.02, 0.12))} | {yi_to_yuan_str(-rev * rnd.uniform(0.02, 0.10))} |",
                f"| 筹资活动产生的现金流量净额 | {yi_to_yuan_str(-net * rnd.uniform(0.3, 0.8))} | {yi_to_yuan_str(-net * rnd.uniform(0.2, 0.7))} |",
                f"| 期末现金及现金等价物余额 | {yi_to_yuan_str(rev * rnd.uniform(0.3, 0.9))} | {yi_to_yuan_str(rev * rnd.uniform(0.25, 0.8))} |",
            ]
            add("第十节 财务报告", 78, "table",
                "(本表金额单位：元)\n" + "\n".join(cash_rows),
                table_title="现金流量表主要项目")

            # ---- 6) 费用与研发（文本）----
            add("第三节 管理层讨论与分析", 21, "text",
                f"报告期内，公司销售费用{fmt_yi(sell_exp)}亿元，销售费用率{sell_ratio:.2f}%；"
                f"管理费用{fmt_yi(rev * rnd.uniform(0.04, 0.09))}亿元；"
                f"研发投入{fmt_yi(rev * rnd.uniform(0.003, 0.02))}亿元。"
                f"公司合同负债（预收款项）{fmt_yi(rev * rnd.uniform(0.05, 0.25))}亿元，"
                f"反映渠道打款意愿{'较强' if rnd.random() > 0.4 else '平稳'}。")

            # ---- 7) 分红方案（文本）----
            add("第四节 公司治理", 34, "text",
                f"公司拟以总股本为基数，向全体股东每10股派发现金红利{rnd.uniform(1.5, 30):.2f}元（含税），"
                f"合计派发现金红利{fmt_yi(net * rnd.uniform(0.3, 0.65))}亿元，占本年度归母净利润的"
                f"{rnd.uniform(30, 65):.2f}%。")

            # ---- 8) 行业情况（文本，无数字，用于测试"非数值题"）----
            add("第三节 管理层讨论与分析", 10, "text",
                "报告期内，白酒行业整体呈现结构性分化态势。名酒企业凭借品牌力与渠道优势，"
                "在高端及次高端价格带保持较强韧性；区域酒企则以省内基本盘为依托，"
                "稳步推进产品结构升级。行业竞争焦点从单纯的渠道扩张，转向品牌、品质与消费者培育的综合比拼。")

    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/chunks/chunks.jsonl")
    ap.add_argument("--companies", type=int, default=None, help="只生成前 N 家")
    ap.add_argument("--years", default="2023,2024")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",")]
    chunks = build_chunks(years, args.companies)

    out = ensure_dir(Path(args.out).parent) / Path(args.out).name
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    names = sorted({c["company"] for c in chunks})
    n_text = sum(1 for c in chunks if c["block_type"] == "text")
    n_table = sum(1 for c in chunks if c["block_type"] == "table")

    print("=== 仿真数据生成完成 ===")
    print(f"公司 {len(names)} 家：{'、'.join(names)}")
    print(f"年份 {years}")
    print(f"块数 {len(chunks)}（文本 {n_text} / 表格 {n_table}）")
    print(f"输出 → {out}")
    print("\n⚠️  数据为模拟生成，仅用于演示与流程联调，不可用于任何真实分析。")


if __name__ == "__main__":
    main()
