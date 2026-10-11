# -*- coding: utf-8 -*-
"""
一键演示：不需要下载 PDF、不需要装 faiss/大模型，直接把整条链路跑给你看

用法：
    python tools/run_demo.py              # 控制台输出 + 生成页面预览 HTML
    python tools/run_demo.py --open       # 顺便尝试用浏览器打开预览

它做三件事：
    1) 确保仿真数据与轻量索引存在（没有就自动生成）
    2) 对 7 个代表性问题跑一遍检索，控制台打印召回详情
    3) 渲染 docs/demo_preview.html —— 问答页面的静态快照（真实检索结果）

⚠️ 仿真数据为模拟生成，仅用于演示；页面效果是真实的，数字不是。
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, resolve  # noqa: E402
from importlib import import_module  # noqa: E402

DEMO_QUESTIONS = [
    ("单公司 · 单指标", "贵州茅台2024年实现营业收入是多少？"),
    ("单公司 · 表格多指标", "贵州茅台2024年分产品的营业收入和毛利率分别是多少？"),
    ("单公司 · 跨年度对比", "五粮液2024年营业收入相比2023年增长了多少？"),
    ("跨公司 · 全景比较", "2024年这12家白酒公司中，哪一家的毛利率最高？"),
    ("跨公司 · 全景比较", "2024年哪些白酒公司的营业收入出现了同比下滑？"),
    ("跨公司 · 全景聚合", "这12家白酒公司2024年的营业收入合计大约是多少？"),
    ("非数值题 · 行业描述", "报告期内白酒行业的竞争格局是怎样的？"),
]


def ensure_inputs(rebuild: bool = False) -> None:
    cfg = load_config()
    chunks = resolve(cfg["paths"].get("demo_chunks_path", "demo/chunks/chunks.jsonl"))
    if not chunks.exists():
        print("[准备] 生成仿真数据 ...")
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo_data.py")],
                       cwd=ROOT, check=True)
    light = resolve(cfg["index"].get("light_path", "index/light_index.pkl"))
    if rebuild or not light.exists():
        print("[准备] 构建轻量索引 ...")
        subprocess.run([sys.executable, str(ROOT / "src" / "04_index.py"),
                        "--backend", "light", "--rebuild"], cwd=ROOT, check=True)


# ---------------------------------------------------------------
# HTML 渲染
# ---------------------------------------------------------------
CSS = """
:root{--ink:#1a1a1a;--sub:#6b6b6b;--line:#e4e4e4;--accent:#8B1E3F;
      --ok:#0f7b3f;--bg:#fafaf9;--chip:#f4f1f2;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.7;
     font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;}
.layout{max-width:1180px;margin:0 auto;padding:28px 20px 80px;display:grid;
        grid-template-columns:250px 1fr;gap:26px;}
@media(max-width:900px){.layout{grid-template-columns:1fr;}}
.sidebar{background:#fff;border:1px solid var(--line);border-radius:12px;
         padding:18px;height:fit-content;position:sticky;top:20px;}
.sidebar h3{font-size:.92rem;margin:0 0 12px;color:var(--accent);}
.sidebar label{display:block;font-size:.82rem;color:var(--sub);margin:12px 0 4px;}
.sidebar .radio{font-size:.85rem;padding:5px 8px;border-radius:6px;background:#f7f7f6;margin:3px 0;}
.sidebar .radio.sel{background:var(--accent);color:#fff;}
.sidebar code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:.78rem;}
.qbtn{display:block;width:100%;text-align:left;font:inherit;font-size:.8rem;
      background:#fff;border:1px solid var(--line);border-radius:7px;
      padding:7px 9px;margin:5px 0;cursor:default;color:#333;line-height:1.45;}
main h1{font-size:1.65rem;margin:0 0 4px;}
main .lead{color:var(--sub);font-size:.88rem;margin:0 0 22px;}
.warn{background:#fff8e6;border:1px solid #f0d48a;border-radius:9px;
      padding:11px 15px;font-size:.83rem;color:#7a5a00;margin-bottom:20px;}
.qbox{background:#fff;border:1px solid var(--line);border-radius:11px;padding:16px 18px;margin-bottom:18px;}
.qbox .q{font-size:1.08rem;font-weight:600;}
.qbox .meta{font-size:.78rem;color:var(--sub);margin-top:6px;}
.ans{background:#f2faf5;border:1px solid #bfe3cd;border-left:4px solid var(--ok);
     border-radius:9px;padding:16px 18px;margin-bottom:14px;}
.ans h4{margin:0 0 10px;font-size:.9rem;color:var(--ok);letter-spacing:.4px;}
.ans p{margin:8px 0;font-size:.9rem;}
.ans blockquote{margin:10px 0;padding:8px 14px;border-left:3px solid #cfd8d3;
                background:#fff;color:#333;font-size:.84rem;}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
         gap:10px;margin:14px 0 6px;}
.metric{background:#fff;border:1px solid var(--line);border-radius:9px;padding:10px 12px;}
.metric b{display:block;font-size:.72rem;color:var(--sub);font-weight:500;}
.metric span{font-size:.95rem;font-weight:600;}
h4.sec{font-size:1rem;margin:26px 0 10px;padding-left:10px;border-left:4px solid var(--accent);}
.cite{background:#fff;border:1px solid var(--line);border-radius:10px;
      margin:9px 0;overflow:hidden;}
.cite .head{padding:10px 14px;background:#fcbcbc;background:#fbf9f9;border-bottom:1px solid var(--line);
            font-size:.83rem;display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
.cite .head .co{font-weight:600;}
.badge{font-size:.7rem;padding:1px 7px;border-radius:999px;background:var(--chip);color:#555;}
.badge.tbl{background:#eef4ff;color:#2b4f9e;}
.badge.txt{background:#f0f7f1;color:#2c6b41;}
.cite pre{margin:0;padding:12px 14px;background:#fff;font-size:.79rem;
          overflow:auto;white-space:pre-wrap;color:#333;
          font-family:ui-monospace,Menlo,Consolas,monospace;}
.cite .txtbody{margin:0;padding:12px 14px;font-size:.83rem;color:#333;background:#fff;}
.tnote{padding:7px 14px;font-size:.75rem;color:#888;background:#fcfcfb;
       border-bottom:1px solid #f0f0ee;}
table.md{width:100%;border-collapse:collapse;font-size:.79rem;background:#fff;}
table.md th,table.md td{border-bottom:1px solid #ececeb;border-right:1px solid #f4f4f3;
                        padding:7px 12px;text-align:left;}
table.md th{background:#f7f5f5;font-weight:600;color:#444;white-space:nowrap;}
table.md td.n{text-align:right;font-variant-numeric:tabular-nums;}
table.md tr.total td{font-weight:600;background:#fcfaf9;}
table.md tr:last-child td{border-bottom:none;}
.sumwrap{background:#fff;border:1px solid #e0d9cd;border-radius:10px;
         margin:12px 0 16px;overflow:hidden;}
.sumhead{padding:9px 14px;background:#faf6ef;border-bottom:1px solid #efe8dc;
         font-size:.84rem;font-weight:600;color:#7a5a00;}
.sumconcl{padding:9px 14px;background:#f5fbf7;border-top:1px solid #dcece2;
          font-size:.83rem;color:#1f5c3a;}
details.fold{margin:6px 0;padding:7px 13px;}
details.fold summary{font-size:.82rem;font-weight:500;color:#555;}
details{background:#fff;border:1px solid var(--line);border-radius:10px;
        padding:10px 14px;margin-top:14px;}
summary{cursor:pointer;font-size:.87rem;font-weight:600;}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px;}
@media(max-width:700px){.two{grid-template-columns:1fr;}}
.two h5{margin:0 0 6px;font-size:.8rem;color:var(--sub);}
.two ol{margin:0;padding-left:20px;font-size:.78rem;color:#444;}
.foot{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);
      color:var(--sub);font-size:.78rem;}
"""


def esc(s: str) -> str:
    return html.escape(str(s))


def md_table_to_html(md: str) -> str:
    """把 Markdown 表格转成 HTML 表格，让预览里的表格真正成表。

    非表格行（如「(本表金额单位：元)」）作为说明放到表格上方。
    """
    lines = [ln for ln in md.splitlines() if ln.strip()]
    notes, rows = [], []
    for ln in lines:
        if ln.lstrip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):   # 分隔行
                continue
            rows.append(cells)
        else:
            notes.append(ln.strip())

    if not rows:
        return f'<pre>{esc(md)}</pre>'

    head, body = rows[0], rows[1:]
    out = []
    if notes:
        out.append(f'<div class="tnote">{esc(" ".join(notes))}</div>')
    out.append('<table class="md"><thead><tr>')
    out += [f"<th>{esc(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        r = r + [""] * (len(head) - len(r))
        cls = ' class="total"' if any(k in r[0] for k in ("合计", "总计", "小计")) else ""
        out.append(f"<tr{cls}>" + "".join(
            f'<td class="n">{esc(c)}</td>' if i else f"<td>{esc(c)}</td>"
            for i, c in enumerate(r[:len(head)])) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def parse_md_table(md: str) -> tuple[list[str], list[list[str]]]:
    """解析 Markdown 表格 -> (表头, 数据行)。"""
    head: list[str] = []
    rows: list[list[str]] = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):    # 分隔行
            continue
        if not head:
            head = cells
        else:
            rows.append(cells)
    return head, rows


def col_index(head: list[str], *keywords: str) -> int | None:
    """按表头关键词定位列。优先匹配靠右的同名列，避开“占比”这类干扰列。"""
    cands = []
    for i, h in enumerate(head):
        if any(k in h for k in keywords):
            cands.append(i)
    if not cands:
        return None
    # 「营收占比(%)」也含"营收"，但它不是我们要的金额列：
    # 金额列的表头不含“占比/比例/比重”
    for i in cands:
        if not any(x in head[i] for x in ("占比", "比例", "比重")):
            return i
    return cands[0]


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_metrics(hits: list[dict]) -> list[dict]:
    """从召回片段里抽出每家公司的关键指标，拼成一张横向可比较的表。

    这正是方案里强调的「先抽数值、再比较」层：
    不让 LLM 自己在长文里找数字，而是先把命中片段里的指标结构化出来，
    再交给模型比较——跨公司题靠「模型自己读长文」几乎必错。

    ⚠️ 关键实现细节：取毛利率必须**按表头定位列**，
    不能“取行里最大的百分数”——分产品表同时存在“毛利率(%)”和
    “营收占比(%)”两列，后者合计恒为 100.00，按数值筛会把毛利率全错成 100%。
    """
    import re
    by_company: dict[str, dict] = {}

    for h in hits:
        co = h["company"]
        row = by_company.setdefault(co, {
            "company": co, "rev": None, "gm": None, "net": None,
            "period": h["report_period"], "page": h["page"],
        })
        text = h["content"]

        # ---- 文本块：从叙述句里取数 ----
        if row["rev"] is None:
            m = re.search(r"营业(?:总)?收入\s*([\d,\.]+)\s*亿元", text)
            if m:
                row["rev"] = _to_float(m.group(1))
        if row["net"] is None:
            m = (re.search(r"归属于上市公司股东的净利润\s*([\d,\.]+)\s*亿元", text)
                 or re.search(r"归母净利润\s*([\d,\.]+)\s*亿元", text)
                 or re.search(r"净利润\s*([\d,\.]+)\s*亿元", text))
            if m:
                row["net"] = _to_float(m.group(1))
        if row["gm"] is None:
            m = re.search(r"综合毛利率\s*([\d\.]+)\s*%", text) or \
                re.search(r"毛利率\s*([\d\.]+)\s*%", text)
            if m:
                row["gm"] = _to_float(m.group(1))

        # ---- 表格块：按表头定位列，读「合计」行 ----
        if h["block_type"] != "table":
            continue

        head, rows = parse_md_table(text)
        if not rows:
            continue

        i_gm = col_index(head, "毛利率")
        i_rev = col_index(head, "营业收入", "营业总收入", "收入")
        i_cost = col_index(head, "营业成本")

        total_row = None
        for r in rows:
            if r and any(k in r[0] for k in ("合计", "总计", "小计")):
                total_row = r
                break
        data_rows = [r for r in rows if r is not total_row]

        # 毛利率：优先读合计行（无合计行则取分项中的最大值）
        gm_cands = []
        if i_gm is not None and i_gm < len(total_row or []):
            v = _to_float(total_row[i_gm]) if total_row else None
            if v is not None:
                gm_cands.append(v)
        if i_gm is not None:
            gm_cands += [v for v in (_to_float(r[i_gm]) for r in data_rows
                                     if i_gm < len(r)) if v is not None]
        # 只有落在合理区间才算毛利率（兼顾极端值防御）
        gm_cands = [v for v in gm_cands if 0 < v <= 100]
        if row["gm"] is None and gm_cands:
            row["gm"] = gm_cands[0]

        # 营业收入：全表为元，换算成亿元
        if row["rev"] is None and i_rev is not None and total_row and i_rev < len(total_row):
            v = _to_float(total_row[i_rev])
            if v is not None and v > 10000:
                row["rev"] = v / 1e8
        # 无合计行的表（如分地区）：直接把收入列求和
        if row["rev"] is None and i_rev is not None and not total_row:
            vals = [_to_float(r[i_rev]) for r in data_rows if i_rev < len(r)]
            vals = [v for v in vals if v is not None]
            if vals and sum(vals) > 10000:
                row["rev"] = sum(vals) / 1e8

    return [r for r in by_company.values() if r["rev"] is not None or r["gm"] is not None]


def render_summary_table(rows: list[dict], sort_by: str = "gm") -> str:
    """全景题的结论汇总表。"""
    if not rows:
        return ""

    # 按选定的指标降序；该指标缺失的排到最后
    def key(r: dict):
        v = r.get(sort_by)
        return -1 if v is None else -v * -1

    ranked = sorted(rows, key=lambda r: -(r.get(sort_by) if r.get(sort_by) is not None else -1))
    best = ranked[0]
    metric_name = "毛利率" if sort_by == "gm" else "营业收入"

    trs = []
    for i, r in enumerate(ranked, 1):
        cls = ' class="total"' if i == 1 else ""
        gm = "—" if r["gm"] is None else f"{r['gm']:.2f}"
        net = "—" if r["net"] is None else f"{r['net']:,.2f}"
        rev = "—" if r["rev"] is None else f"{r['rev']:,.2f}"
        trs.append(
            f'<tr{cls}><td class="n">{i}</td><td>{esc(r["company"])}</td>'
            f'<td class="n">{rev}</td><td class="n">{gm}</td>'
            f'<td class="n">{net}</td>'
            f'<td>{r["period"]} P{r["page"]}</td></tr>'
        )

    best_v = best.get(sort_by)
    if best_v is None:
        concl = "片段里的指标不足以排序，需扩大召回或改进抽取。"
    else:
        unit = "%" if sort_by == "gm" else " 亿元"
        concl = (f'按结构化数值排序，<b>{esc(best["company"])}</b> 以 '
                 f'<b>{best_v:.2f}{unit}</b> 的{metric_name}排在首位 '
                 f'（{best["period"]} 年报 第 {best["page"]} 页）。')

    return (f'<div class="sumwrap">'
            f'<div class="sumhead">📊 结构化汇总（系统从召回片段中抽取的关键指标，'
            f'按{metric_name}降序 · {len(ranked)} 家）</div>'
            f'<table class="md"><thead><tr><th>排名</th><th>公司</th>'
            f'<th>营业收入(亿元)</th><th>毛利率(%)</th><th>净利润(亿元)</th>'
            f'<th>出处</th></tr></thead><tbody>'
            + "".join(trs) +
            f'</tbody></table><div class="sumconcl">✅ {concl}'
            f'</div></div>')


def render_cite(i: int, h: dict) -> str:
    cls = "tbl" if h["block_type"] == "table" else "txt"
    label = "表格" if h["block_type"] == "table" else "文本"
    tt = f'<span class="badge">表名：{esc(h["table_title"])}</span>' if h.get("table_title") else ""
    if h["block_type"] == "table":
        body = md_table_to_html(h["content"])
    else:
        body = f'<p class="txtbody">{esc(h["content"])}</p>'
    return f"""<div class="cite">
  <div class="head">
    <span class="co">{i}. 【{esc(h['company'])} {esc(h['report_period'])}{esc(h['report_type'])} ·
      {esc(h['section'] or '—')} · 第 {h['page']} 页】</span>
    <span class="badge {cls}">{label}</span>
    <span class="badge">单位：{esc(h['unit'])}</span>
    <span class="badge">score={h['score']}</span>
    {tt}
  </div>
  {body}
</div>"""


def render_html(results: list[dict], cfg: dict) -> str:
    parts = [f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>问答页面预览 · 白酒财报 RAG 知识库</title><style>{CSS}</style></head><body>
<div class="layout">
<aside class="sidebar">
  <h3>⚙️ 检索设置</h3>
  <label>检索模式</label>
  <div class="radio sel">混合（向量 + BM25 + RRF）</div>
  <div class="radio">仅向量</div>
  <div class="radio">仅 BM25</div>
  <label>送入模型的相关片段数</label>
  <div class="radio">6</div>
  <label>数据与生成</label>
  <div class="radio">provider = <code>{esc(cfg['generate']['provider'])}</code></div>
  <div class="radio">backend = <code>light</code></div>
  <label>试试这些问题</label>
"""]
    for _, q in DEMO_QUESTIONS:
        parts.append(f'  <button class="qbtn">{esc(q)}</button>\n')
    parts.append("""</aside>
<main>
  <h1>🍶 白酒财报问答知识库</h1>
  <p class="lead">12 家上市公司 · 2024 年年报 · 向量检索 + BM25 混合召回的带出处问答</p>
  <div class="warn"><b>这是页面预览快照</b>（静态 HTML）。真实页面由
    <code>streamlit run app/streamlit_app.py</code> 启动，可输入任意问题、切换检索模式。
    当前展示的是用<b>仿真数据</b>跑出的<b>真实检索结果</b>——流程真实，数字为模拟值。</div>
""")

    for label, q in DEMO_QUESTIONS:
        out = next(r for r in results if r["question"] == q)
        comps = {h["company"] for h in out["hits"]}
        parts.append(f"""  <div class="qbox">
    <div class="q">{esc(q)}</div>
    <div class="meta">题型：{esc(label)}</div>
  </div>
  <div class="ans">
    <h4>📌 答案（demo 模式）</h4>
    <p>{esc(out['answer'].splitlines()[0])}</p>
    <blockquote>{esc(out['strategy'])}</blockquote>
    <p style="font-size:.82rem;color:#555">下面是检索到的最相关原文片段；
      接入大模型后，这些片段会连同强制引用出处的 Prompt 一起送出，由模型合成完整答案。
      引用格式示例：<code>【贵州茅台 2024 第16页】</code></p>
  </div>
  <div class="metrics">
    <div class="metric"><b>检索策略</b><span>{'全景/配额' if out['is_panorama'] else '常规'}</span></div>
    <div class="metric"><b>召回片段</b><span>{len(out['hits'])}</span></div>
    <div class="metric"><b>覆盖公司</b><span>{len(comps)} / 12</span></div>
    <div class="metric"><b>识别公司</b><span>{esc('、'.join(out['detected_companies']) or '—')}</span></div>
  </div>
  <h4 class="sec">📎 引用来源</h4>
""")
        # 全景题：先给结论汇总表（已抽数值再比较），再列原文出处
        if out["is_panorama"]:
            rows = extract_metrics(out["hits"])
            if rows:
                parts.append(render_summary_table(rows, sort_by="gm"))

        for i, h in enumerate(out["hits"][:6], 1):
            cite = render_cite(i, h)
            if i > 2:   # 只展开前2条，其余默认折叠，避免页面冗长
                cite = f'<details class="fold"><summary>展开来源 {i}</summary>{cite}</details>'
            parts.append(cite)
        if len(out["hits"]) > 6:
            parts.append(f'  <p style="font-size:.8rem;color:#777">'
                         f'（另有 {len(out["hits"]) - 6} 个片段未在预览中展开）</p>\n')

        parts.append(f"""  <details>
    <summary>🔍 检索过程（向量 / BM25）</summary>
    <div class="two">
      <div><h5>向量召回 Top-5</h5><ol>""")
        for r in out["vector_top"]:
            parts.append(f"<li><code>{esc(r['chunk_id'])}</code> score={r['score']}</li>")
        parts.append("</ol></div><div><h5>BM25 召回 Top-5</h5><ol>")
        for r in out["bm25_top"]:
            parts.append(f"<li><code>{esc(r['chunk_id'])}</code> score={r['score']}</li>")
        parts.append("</ol></div></div></details>\n")

    parts.append(f"""  <div class="foot">
    预览生成时间：{datetime.now():%Y-%m-%d %H:%M} ·
    语料：仿真数据（12 家 × 2023/2024 年报）·
    后端：light（纯标准库 TF-IDF + BM25）·
    正式部署请使用 <code>faiss</code> 后端与真实年报
  </div>
</main></div></body></html>""")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="强制重建仿真数据与索引")
    ap.add_argument("--open", action="store_true", help="生成后用浏览器打开预览")
    args = ap.parse_args()

    ensure_inputs(args.rebuild)

    cfg = load_config()
    Rag = import_module("05_rag").Retriever
    rag = Rag(cfg)

    print("\n" + "=" * 78)
    print("问答演示（仿真数据 · light 后端）")
    print("=" * 78)

    results = []
    for label, q in DEMO_QUESTIONS:
        out = rag.search(q)
        out["question"] = q
        out["label"] = label
        out["answer"] = rag._demo_answer(q, out["hits"])
        results.append(out)

        comps = {h["company"] for h in out["hits"]}
        print(f"\n【{label}】{q}")
        print(f"  策略：{out['strategy']}")
        print(f"  召回 {len(out['hits'])} 块，覆盖 {len(comps)} 家公司")
        for h in out["hits"][:5]:
            tag = "表" if h["block_type"] == "table" else "文"
            print(f"    [{tag}] {h['company']:<8} P{h['page']:<3} {h['section'][:14]:<16} "
                  f"score={h['score']}")
        if len(out["hits"]) > 5:
            print(f"    …… 另有 {len(out['hits']) - 5} 块")

    # 落盘 JSON，便于其他脚本复用
    out_json = resolve("demo") / "demo_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 渲染页面预览
    html_path = resolve("docs") / "demo_preview.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(results, cfg), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"检索结果 JSON → {out_json}")
    print(f"页面预览 HTML → {html_path}")
    print("用浏览器打开上面这个 HTML，就是问答页面的样子。")
    print("=" * 78)

    if args.open:
        import webbrowser
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
