# -*- coding: utf-8 -*-
"""
04 · 建索引

两种后端（config.yaml -> index.backend）：
    faiss = 正式方案：向量索引（bge-m3 + FAISS） + BM25（jieba + 财报词典）
    light = 轻量降级：纯标准库 TF-IDF + BM25，零依赖，用于演示与联调
    auto  = 能 import faiss/sentence_transformers 就用 faiss，否则降级 light（默认）

用法：
    python src/04_index.py                     # 按配置自动选后端
    python src/04_index.py --backend light     # 强制轻量后端（秒级完成，无需下模型）
    python src/04_index.py --rebuild           # 强制重建

产出：
    faiss 后端：index/faiss/index.faiss + meta.json、index/bm25.pkl、index/finance_dict.txt
    light 后端：index/light_index.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_dir, load_config, read_jsonl, resolve  # noqa: E402

# 财报专有词典：不加载它，jieba 会把「归母净利润」这类术语切碎
FINANCE_WORDS = [
    "归母净利润", "扣非净利润", "归属于上市公司股东的净利润", "营业收入", "营业成本",
    "毛利率", "净利率", "经营活动产生的现金流量净额", "投资活动产生的现金流量净额",
    "筹资活动产生的现金流量净额", "基本每股收益", "稀释每股收益", "加权平均净资产收益率",
    "资产负债率", "存货周转率", "合同负债", "预收款项", "销售费用", "管理费用",
    "研发费用", "营业收入同比", "高档酒", "中档酒", "普通酒", "系列酒", "基酒",
    "成品酒", "省内", "省外", "经销商", "直销", "批发代理", "分产品", "分地区",
    "营业总收入", "利润总额", "净利润", "研发投入", "在建工程", "货币资金",
]


def build_finance_dict(path: Path) -> None:
    path.write_text("\n".join(FINANCE_WORDS), encoding="utf-8")


def tokenize(text: str) -> list[str]:
    import jieba
    return [t for t in jieba.lcut_for_search(text) if (t.strip() and len(t) > 1) or t.isdigit()]


def pick_chunks_path(cfg: dict, override: str | None) -> str:
    """优先真实抽取结果，其次仿真数据。"""
    if override:
        return override
    real = cfg["paths"]["chunks_path"]
    if resolve(real).exists():
        return real
    demo = cfg["paths"].get("demo_chunks_path")
    if demo and resolve(demo).exists():
        print(f"[提示] 未找到真实语料 {real}，改用仿真数据 {demo}")
        return demo
    sys.exit("找不到语料。请先运行 03_chunk.py，或生成仿真数据：\n"
             "  python tools/make_demo_data.py")


def resolve_backend(want: str) -> str:
    """选择实际后端。与 05_rag.Retriever._resolve_backend 保持同一规则：
    能 import faiss 且索引文件已存在，才用 faiss；否则一律 light。
    """
    want = (want or "auto").lower()
    if want == "light":
        return "light"
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return "faiss"
    except ImportError as e:
        if want == "faiss":
            print(f"[自动降级] 指定 faiss 但缺少依赖 {e.name}，改用 light 后端")
        else:
            print(f"[自动降级] 未安装 {e.name}，改用 light 后端")
            print("           正式检索请执行：pip install faiss-cpu sentence-transformers")
        return "light"


def build_light(cfg: dict, chunks: list[dict], rebuild: bool) -> None:
    from light_index import LightIndex

    path = resolve(cfg["index"].get("light_path", "index/light_index.pkl"))
    if path.exists() and not rebuild:
        print(f"[跳过] 轻量索引已存在：{path}（--rebuild 可强制重建）")
        return
    li = LightIndex()
    li.build(chunks)
    li.save(path)


def build_faiss(cfg: dict, chunks: list[dict], rebuild: bool) -> None:
    import jieba

    idx_cfg = cfg["index"]
    dict_path = ensure_dir(Path(idx_cfg["bm25_dict"]).parent) / "finance_dict.txt"
    build_finance_dict(dict_path)
    jieba.load_userdict(str(dict_path))

    # ---------- 1) 向量索引 ----------
    vec_dir = ensure_dir(idx_cfg["vector_dir"])
    faiss_path, meta_path = vec_dir / "index.faiss", vec_dir / "meta.json"

    if faiss_path.exists() and meta_path.exists() and not rebuild:
        print(f"[跳过] 向量索引已存在：{faiss_path}（--rebuild 可强制重建）")
    else:
        import faiss
        from sentence_transformers import SentenceTransformer

        print(f"[编码] 模型 {idx_cfg['embed_model']}（首次会下载权重，需联网）")
        model = SentenceTransformer(idx_cfg["embed_model"])
        texts = [c["text_for_index"] for c in chunks]

        embs = model.encode(
            texts, batch_size=idx_cfg["embed_batch_size"],
            normalize_embeddings=True, show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(embs.shape[1])   # 归一化后内积 = 余弦相似度
        index.add(embs)
        faiss.write_index(index, str(faiss_path))
        meta_path.write_text(json.dumps(
            {"model": idx_cfg["embed_model"], "count": len(chunks), "dim": int(embs.shape[1])},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[完成] 向量索引 {index.ntotal} 条 → {faiss_path}")

    # ---------- 2) BM25 索引 ----------
    bm25_path = resolve(idx_cfg["bm25_path"])
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    if bm25_path.exists() and not rebuild:
        print(f"[跳过] BM25 索引已存在：{bm25_path}")
        return

    from rank_bm25 import BM25Okapi

    print("[分词] 构建 BM25 ...")
    corpus_tokens = []
    for c in chunks:
        c["tokens"] = tokenize(c["text_for_index"])
        corpus_tokens.append(c["tokens"])

    bm25 = BM25Okapi(corpus_tokens)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks]}, f)
    print(f"[完成] BM25 索引 → {bm25_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--backend", choices=["auto", "faiss", "light"], default=None,
                    help="覆盖 config.yaml 里的 index.backend")
    ap.add_argument("--chunks", default=None, help="指定 chunks.jsonl（默认自动选择）")
    args = ap.parse_args()

    cfg = load_config()
    backend = resolve_backend(args.backend or cfg["index"].get("backend", "auto"))
    chunks_path = pick_chunks_path(cfg, args.chunks)
    chunks = list(read_jsonl(chunks_path))

    print(f"载入 {len(chunks)} 个 chunk（{chunks_path}）")
    print(f"检索后端：{backend}\n")

    if backend == "light":
        build_light(cfg, chunks, args.rebuild)
        print("\n索引构建完毕。下一步：streamlit run app/streamlit_app.py")
        print("提醒：light 是 TF-IDF + BM25，不是语义向量；正式评测请改用 faiss 后端。")
    else:
        build_faiss(cfg, chunks, args.rebuild)
        print("\n索引构建完毕。下一步：streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
