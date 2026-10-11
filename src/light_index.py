# -*- coding: utf-8 -*-
"""
轻量检索后端：纯标准库实现的 TF-IDF 向量检索 + BM25 关键词检索

存在意义：
    正式后端用 FAISS + bge-m3（效果好，但要下 ~2GB 权重、装 faiss）。
    本模块让整条链路在**零第三方依赖**下也能跑起来——
    没装大库、或只想快速看页面效果时，用它。

局限（务必了解）：
    · "向量检索"这里是 TF-IDF 稀疏向量，不是语义向量。
      它能做词形匹配与 TF-IDF 加权，但**不做同义/改写泛化**
      （问"营收"不会自动匹配到"营业总收入"的语义）。
      因此它是演示与联调的降级方案，正式评测请用 faiss 后端。
    · 中文分词用「单字 + 二元组」切分（无需 jieba），对财报这种
      术语密集的文本效果尚可，但不如 jieba + 自定义词典。

接口与正式后端一致：
    LightIndex.build(chunks) / save() / load()
    .search_vec(query, k) -> [(chunk_id, score)]
    .search_bm25(query, k) -> [(chunk_id, score)]
"""
from __future__ import annotations

import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
TOKEN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%")


def tokenize(text: str) -> list[str]:
    """中文：单字 + 相邻二元组；英文/数字：整体。无需 jieba。"""
    tokens: list[str] = []
    tokens.extend(m.group(0).lower() for m in TOKEN.finditer(text))

    chars = [c for c in text if CJK.match(c)]
    tokens.extend(chars)                                    # unigram
    tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))  # bigram

    # 英文词内部也给 bigram/trigram，提升"revenue"类词的部分匹配
    for m in TOKEN.finditer(text):
        w = m.group(0).lower()
        if len(w) > 4 and w.isalpha():
            tokens.extend(w[i:i + 3] for i in range(len(w) - 2))
    return tokens


class LightIndex:
    """TF-IDF 稀疏向量 + BM25。全部用标准库实现。"""

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.doc_terms: list[dict[str, int]] = []       # 每篇的词频
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, float]]] = {}   # 倒排：词 -> [(doc_idx, tfidf权重)]
        self.avg_len: float = 0.0
        self.n_docs: int = 0

    # ---------------- 构建 ----------------
    def build(self, chunks: list[dict]) -> None:
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.n_docs = len(chunks)

        for c in chunks:
            tf: dict[str, int] = defaultdict(int)
            for t in tokenize(c["text_for_index"]):
                tf[t] += 1
            self.doc_terms.append(dict(tf))
            self.doc_len.append(sum(tf.values()))

        self.avg_len = sum(self.doc_len) / max(1, self.n_docs)

        df: dict[str, int] = defaultdict(int)
        for tf in self.doc_terms:
            for t in tf:
                df[t] += 1
        self.df = dict(df)

        # 倒排 + TF-IDF 权重（用对数饱和的 tf，避免长文档压制短文档）
        post: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, tf in enumerate(self.doc_terms):
            for t, f in tf.items():
                idf = math.log((self.n_docs + 1) / (self.df[t] + 1)) + 1
                w = (1 + math.log(f)) * idf
                post[t].append((i, w))
        # 归一化，便于余弦相似度
        norms = [0.0] * self.n_docs
        for t, plist in post.items():
            for i, w in plist:
                norms[i] += w * w
        norms = [math.sqrt(n) or 1.0 for n in norms]
        self.postings = {t: [(i, w / norms[i]) for i, w in plist] for t, plist in post.items()}

        print(f"[light] 索引构建完成：{self.n_docs} 篇，词表 {len(self.postings)} 项")

    # ---------------- 检索 ----------------
    def search_vec(self, query: str, k: int) -> list[tuple[str, float]]:
        """TF-IDF 余弦相似度。"""
        if k <= 0:
            return []
        q_tf: dict[str, int] = defaultdict(int)
        for t in tokenize(query):
            q_tf[t] += 1
        qw: dict[str, float] = {}
        for t, f in q_tf.items():
            if t not in self.postings:
                continue
            idf = math.log((self.n_docs + 1) / (self.df[t] + 1)) + 1
            qw[t] = (1 + math.log(f)) * idf
        qnorm = math.sqrt(sum(v * v for v in qw.values())) or 1.0

        scores: dict[int, float] = defaultdict(float)
        for t, w in qw.items():
            for i, dw in self.postings[t]:
                scores[i] += (w / qnorm) * dw

        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.chunk_ids[i], round(s, 6)) for i, s in top if s > 0]

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        """BM25（Okapi）。"""
        if k <= 0:
            return []
        q_terms = [t for t in tokenize(query) if t in self.postings]
        scores: dict[int, float] = defaultdict(float)
        for t in q_terms:
            idf = math.log(1 + (self.n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, _ in self.postings[t]:
                f = self.doc_terms[i].get(t, 0)
                denom = f + self.K1 * (1 - self.B + self.B * self.doc_len[i] / max(1e-9, self.avg_len))
                scores[i] += idf * f * (self.K1 + 1) / max(1e-9, denom)

        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.chunk_ids[i], round(s, 6)) for i, s in top if s > 0]

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "doc_terms": self.doc_terms,
                "doc_len": self.doc_len,
                "df": self.df,
                "postings": self.postings,
                "avg_len": self.avg_len,
                "n_docs": self.n_docs,
            }, f)
        print(f"[light] 已保存 → {p}")

    @classmethod
    def load(cls, path: str | Path) -> "LightIndex":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        for k, v in d.items():
            setattr(obj, k, v)
        return obj
