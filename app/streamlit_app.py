# -*- coding: utf-8 -*-
"""
Streamlit 问答页面

本地运行：
    streamlit run app/streamlit_app.py

Colab 运行（配合 cloudflared 一条命令起公网链接）：
    !streamlit run app/streamlit_app.py --server.port 8501 &
    !cloudflared tunnel --url http://localhost:8501

功能：
  · 提问 → 答案 + 出处（公司/章节/页码/原文片段）
  · 侧边栏可切换检索模式（纯向量 / 纯 BM25 / 混合 RRF / 是否跨公司配额）
  · 展开「检索过程」可看到向量 top-5、BM25 top-5、融合后结果
  · 显示耗时统计，便于评测与截图
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, resolve  # noqa: E402
from importlib import import_module  # noqa: E402

Rag = import_module("05_rag").Retriever

st.set_page_config(page_title="白酒财报问答知识库", page_icon="🍶", layout="wide")

DEMO_QUESTIONS = [
    "贵州茅台2024年实现营业收入是多少？同比增速是多少？",
    "泸州老窖2024年归属于上市公司股东的净利润是多少？",
    "贵州茅台2024年分产品的营业收入和毛利率分别是多少？",
    "五粮液2024年营业收入相比2023年增长了多少？",
    "2024年这12家白酒公司中，哪一家的毛利率最高？",
    "2024年哪些白酒公司的营业收入出现了同比下滑？",
    "这12家白酒公司2024年的营业收入合计大约是多少？",
]


@st.cache_resource(show_spinner="正在加载索引与模型（首次较慢）...")
def get_rag():
    return Rag(load_config())


def main() -> None:
    cfg = load_config()

    st.markdown(
        f"<h1 style='margin-bottom:0'>🍶 {cfg['project']['industry']}财报问答知识库</h1>"
        f"<p style='color:#888;margin-top:4px'>"
        f"{len(cfg['companies'])} 家上市公司 · {cfg['project']['period']} 年{cfg['project']['report_type']}"
        f" · 向量检索 + BM25 混合召回的带出处问答</p>",
        unsafe_allow_html=True,
    )

    # ---------------- 侧边栏 ----------------
    with st.sidebar:
        st.header("⚙️ 检索设置")
        mode = st.radio("检索模式", ["混合（向量 + BM25 + RRF）", "仅向量", "仅 BM25"],
                        index=0, help="切换模式后重跑同一批题，即可对比三种模式的准确率——这是「结论页」最好的素材")
        top_k = st.slider("送入模型的相关片段数", 3, 12, cfg["retrieve"]["final_top_k"])
        force_quota = st.checkbox("强制跨公司配额召回", value=False,
                                  help="勾选后即使问题是单公司题，也对每家公司保底召回，用于验证全景题策略")

        st.divider()
        st.caption("数据与生成")
        st.code(f"provider = {cfg['generate']['provider']}", language="text")
        if cfg["generate"]["provider"] == "none":
            st.info("当前为「仅检索」模式，答案区只展示召回片段。")

        st.divider()
        st.caption("试试这些问题：")
        for i, q in enumerate(DEMO_QUESTIONS):
            if st.button(q, key=f"demo_{i}", use_container_width=True):
                st.session_state["question"] = q

    # ---------------- 提问区 ----------------
    # 支持用 URL 参数直接带一个问题进来：?q=2024年哪家白酒公司毛利率最高
    # 用途：① 分享/截图时不用手敲问题；② Colab 隧道地址后面接 ?q=... 就能直接演示某一题。
    param_q = st.query_params.get("q")
    if param_q and not st.session_state.get("question"):
        st.session_state["question"] = param_q

    default_q = st.session_state.get("question", "")
    c1, c2 = st.columns([6, 1])
    with c1:
        question = st.text_input("请输入问题", value=default_q,
                                 placeholder="例：2024年哪家白酒公司毛利率最高？")
    with c2:
        st.write("")
        go = st.button("🔍 提问", type="primary", use_container_width=True)

    if not (go or (question and question != st.session_state.get("_last_q"))):
        st.info("👈 从左侧选一个问题，或直接在输入框提问。")
        return

    st.session_state["_last_q"] = question
    rag = get_rag()

    # 侧边栏模式覆盖：临时改写配置
    if mode == "仅向量":
        cfg["retrieve"]["bm25_top_k"] = 0
    elif mode == "仅 BM25":
        cfg["retrieve"]["vector_top_k"] = 0

    with st.spinner("检索与生成中..."):
        t0 = time.time()
        out = rag.answer(question, top_k)
        cost = time.time() - t0

    # ---------------- 答案 ----------------
    st.markdown("### 📌 答案")
    st.success(out["answer"] if out.get("answer") else "（无生成结果）")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("检索策略", "全景/配额" if out["is_panorama"] else "常规")
    m2.metric("召回片段", len(out["hits"]))
    m3.metric("识别公司", "、".join(out["detected_companies"]) or "—")
    m4.metric("总耗时", f"{cost:.2f}s")
    st.caption(f"策略明细：{out['strategy']}")

    # ---------------- 出处 ----------------
    st.markdown("### 📎 引用来源")
    for i, h in enumerate(out["hits"], 1):
        label = (f"{i}. 【{h['company']} {h['report_period']}{h['report_type']} · "
                 f"{h['section'] or '—'} · P{h['page']}】 "
                 f"[{h['block_type']}｜{h['unit']}｜score={h['score']}]")
        with st.expander(label, expanded=(i <= 2)):
            if h.get("table_title"):
                st.caption(f"表名：{h['table_title']}")
            st.markdown(h["content"] if h["block_type"] == "table"
                        else h["content"], unsafe_allow_html=False)
            st.caption(f"chunk_id = {h['chunk_id']}")

    # ---------------- 检索过程 ----------------
    with st.expander("🔍 检索过程（向量 / BM25 / 融合）"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**向量召回 Top-5**")
            for r in out["vector_top"]:
                st.write(f"- `{r['chunk_id']}`  score={r['score']}")
        with c2:
            st.markdown("**BM25 召回 Top-5**")
            for r in out["bm25_top"]:
                st.write(f"- `{r['chunk_id']}`  score={r['score']}")

    st.caption(f"索引文件：{resolve(cfg['index']['vector_dir'])} · {cfg['paths']['chunks_path']}")


if __name__ == "__main__":
    main()
