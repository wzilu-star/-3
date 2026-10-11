# -*- coding: utf-8 -*-
"""
05 · RAG 核心：混合检索 + RRF 融合 + 跨公司配额召回 + 生成

对外提供：
    Retriever.search(question)  -> list[dict]   检索到的块（含 score / 来源）
    Retriever.answer(question)  -> dict         答案 + 出处 + 检索轨迹

跨公司全景题的处理（本项目的评分重点）：
    1) 识别问题是否为「全景/比较」类（config.utils.is_panorama_question）
    2) 若是：对每家公司分别检索，各取 per_company_quota 块 → 保证每家都有位置
              （普通 top-k 会被单家公司霸榜，全景题必错）
    3) 生成时要求先做数值抽取再比较，而不是让模型自己在长文里找
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (detect_companies, faiss_load, is_panorama_question,  # noqa: E402
                   load_config, pick_device, read_jsonl, resolve, resolve_model)


class Retriever:
    def __init__(self, cfg: dict | None = None, chunks_path: str | None = None):
        self.cfg = cfg or load_config()
        self.chunks_path = chunks_path or self._pick_chunks_path()
        self.chunks = list(read_jsonl(self.chunks_path))
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self._index = self._model = None
        self._bm25 = None
        self._light = None
        self._rrf_k = self.cfg["retrieve"]["rrf_k"]
        self.backend = self._resolve_backend()
        print(f"[Retriever] 语料 {len(self.chunks)} 块 · 后端 = {self.backend}"
              f"{'' if self.backend == 'faiss' else '  (轻量模式：TF-IDF + BM25，非语义向量)'}")

    def _pick_chunks_path(self) -> str:
        """优先用真实抽取结果；没有则用仿真数据，让流程能空跑。"""
        real = self.cfg["paths"]["chunks_path"]
        if resolve(real).exists():
            return real
        demo = self.cfg["paths"].get("demo_chunks_path")
        if demo and resolve(demo).exists():
            print(f"[Retriever] 未找到真实语料 {real}，改用仿真数据 {demo}")
            return demo
        sys.exit(f"既没有真实语料 {real}，也没有仿真数据 {demo}。\n"
                 f"  · 跑真实数据：python src/01_download.py && python src/02_extract.py && python src/03_chunk.py\n"
                 f"  · 或造仿真数据：python tools/make_demo_data.py")

    def _resolve_backend(self) -> str:
        """选择实际使用的检索后端。

        注意：光“能 import faiss”还不够，还必须已经建好 faiss 索引文件，
        否则会警告用户装了依赖但没建索引，然后在检索时抛 FileNotFoundError——
        这类报错对新手极不友好。所以这里一并检查索引是否存在。
        """
        want = str(self.cfg["index"].get("backend", "auto")).lower()
        if want not in ("auto", "faiss", "light"):
            want = "auto"
        if want == "light":
            return "light"

        faiss_index = resolve(self.cfg["index"]["vector_dir"]) / "index.faiss"
        bm25_file = resolve(self.cfg["index"]["bm25_path"])
        has_index = faiss_index.exists() and bm25_file.exists()

        try:
            import faiss  # noqa: F401
            import sentence_transformers  # noqa: F401
            has_deps = True
        except ImportError as e:
            has_deps = False
            if want == "faiss":
                print(f"[Retriever] ⚠ 指定 faiss 后端但依赖缺失（{e.name}），已自动降级为 light。"
                      f"\n           安装：pip install faiss-cpu sentence-transformers")
        if has_deps and has_index:
            return "faiss"

        if has_deps and not has_index:
            print("[Retriever] 检测到 faiss/sentence-transformers 已安装，但没有索引文件：\n"
                  f"           缺 {faiss_index}\n           缺 {bm25_file}\n"
                  "           → 先建索引：python src/04_index.py\n"
                  "           本次先降级用 light 后端（TF-IDF + BM25）。")
        return "light"

    # ---------------- 懒加载 ----------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # 编码器和建索引时用同一个设备，否则查询向量和库里的向量不在同一 dtype/device 上
            self._model = SentenceTransformer(
                resolve_model(self.cfg["index"]["embed_model"]), device=pick_device())
        return self._model

    @property
    def index(self):
        if self._index is None:
            # 用 faiss_load 而不是 faiss.read_index：后者在 Windows 上读不了含中文的路径
            self._index = faiss_load(
                resolve(self.cfg["index"]["vector_dir"]) / "index.faiss")
        return self._index

    @property
    def bm25(self):
        if self._bm25 is None:
            with open(resolve(self.cfg["index"]["bm25_path"]), "rb") as f:
                self._bm25 = pickle.load(f)
        return self._bm25

    @property
    def light(self):
        if self._light is None:
            from light_index import LightIndex
            path = resolve(self.cfg["index"].get("light_path", "index/light_index.pkl"))
            if not path.exists():
                raise FileNotFoundError(
                    f"轻量索引不存在：{path}\n  请先运行：python src/04_index.py")
            self._light = LightIndex.load(path)
        return self._light

    # ---------------- 单路检索 ----------------
    def vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        if self.backend == "light":
            return self.light.search_vec(query, k)

        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idxs = self.index.search(q, k)
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.chunks):
                out.append((self.chunks[i]["chunk_id"], float(s)))
        return out

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        if self.backend == "light":
            return self.light.search_bm25(query, k)

        import jieba
        jieba.load_userdict(str(resolve(self.cfg["index"]["bm25_dict"])))
        tokens = [t for t in jieba.lcut_for_search(query) if t.strip()]
        data = self.bm25
        scores = data["bm25"].get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(data["chunk_ids"][i], float(scores[i])) for i in order if scores[i] > 0]

    # ---------------- RRF 融合 ----------------
    def _rrf(self, *rank_lists: list[tuple[str, float]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for lst in rank_lists:
            for rank, (cid, _) in enumerate(lst, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self._rrf_k + rank)
        return fused

    @staticmethod
    def _detect_year(question: str) -> str | None:
        """从问题里识别报告期年份，如「2024年」-> "2024"。

        只在问题明确点名一个 20xx 年时才返回；提到多个年份（跨年度对比）
        则返回 None，因为两期数据都要召回。
        """
        import re as _re
        years = set(_re.findall(r"(20\d{2})\s*年", question))
        return years.pop() if len(years) == 1 else None

    def _mode_k(self, base: int, which: str) -> int:
        """把「该检索通路是否开启」带进配额召回。

        `vector_top_k` / `bm25_top_k` 置 0 表示关掉那条通路 —— 页面侧边栏的
        「仅向量 / 仅 BM25 / 混合」和 tools/mode_compare.py 都靠这个开关切模式。

        但全景题走的是 _quota_search，它是按「每家公司 quota*N」取块的，
        取块数不是从这两个配置推出来的。如果不在这里把开关带下去，
        切模式对全景题**完全无效** —— 表现为三种模式召回一模一样的块，
        而页面上却显示「已切换到仅 BM25」，属于静默失效。
        """
        return base if self.cfg["retrieve"].get(which, 0) > 0 else 0

    def _quota_search(self, question: str, companies: list[str],
                      year: str | None = None) -> list[list[tuple[str, float]]]:
        """跨公司配额召回：每家公司单独检索，各取 quota 块。

        返回**按公司分组**的结构（而不是拍平），因为后续必须做轮询交错：
        若直接按公司顺序拼接再截断 top-k，结果会只剩前几家公司，
        反而制造出配额本来要解决的「单公司霸榜」问题。

        year 必须在这里就参与过滤，不能等外层再筛：
        每家公司的 quota 块在内部就取完了，同一家公司 2023/2024 的同类章节
        内容高度相似，若不限年份，2023 会先把 2024 挤掉，
        外层再筛就无块可用、只能回退——表现为“问了 2024 却全是 2023”。
        """
        quota = self.cfg["retrieve"]["per_company_quota"]
        groups: list[list[tuple[str, float]]] = []
        for comp in companies:
            sub_q = f"{comp} {question}"
            vec = self.vector_search(sub_q, self._mode_k(quota * 6, "vector_top_k"))
            bm = self.bm25_search(sub_q, self._mode_k(quota * 6, "bm25_top_k"))
            fused = self._rrf(vec, bm)

            def _ok(cid: str) -> bool:
                c = self.by_id.get(cid, {})
                if c.get("company") != comp:
                    return False
                return year is None or c.get("report_period") == year

            comp_hits = [(cid, s) for cid, s in sorted(fused.items(), key=lambda x: -x[1])
                         if _ok(cid)]
            # 若限年份后无块，放宽配额多取一些再筛（避免该家直接掉出汇总表）
            if not comp_hits and year:
                extra = self._rrf(
                    self.vector_search(sub_q, self._mode_k(quota * 20, "vector_top_k")),
                    self.bm25_search(sub_q, self._mode_k(quota * 20, "bm25_top_k")))
                comp_hits = [(cid, s) for cid, s in sorted(extra.items(), key=lambda x: -x[1])
                             if _ok(cid)]
            if comp_hits:
                groups.append(self._diversify(comp_hits, quota))
        return groups

    def _diversify(self, candidates: list[tuple[str, float]], quota: int) -> list[tuple[str, float]]:
        """在一家公司内部，优先选**不同章节 + 不同块类型**的块。

        原因：同一家公司的 quota 块若全落在同一张表上（如都反 P16），
        就会漏掉文本章节里的归母净利润、销售费用等指标，
        汇总表会出现一列全是“—”。贪心选“章节未出现过”的块能自然散布开来。
        """
        picked: list[tuple[str, float]] = []
        used: set[tuple[str, str]] = set()
        remaining = list(candidates)

        while len(picked) < quota and remaining:
            best = None
            for cid, s in remaining:
                c = self.by_id.get(cid, {})
                # 多样性 key = (章节, 块类型)：既分散章节，也兼顾表格/文本
                # 若只看章节，同一节里的表格会把文本块全部挤掉，
                # 而归母净利润、销售费用这类指标往往只在叙述文本里。
                key = (c.get("section", ""), c.get("block_type", ""))
                score_key = (0 if key not in used else 1, -s)
                if best is None or score_key < best[0]:
                    best = (score_key, cid, s, key)
            _, cid, s, key = best
            picked.append((cid, s))
            used.add(key)
            remaining = [(c, sc) for c, sc in remaining if c != cid]
        return picked

    @staticmethod
    def _interleave(groups: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
        """轮询交错：先各公司第1块，再各公司第2块……

        这样即使最终只取 N 块，也能保证覆盖 N 家不同公司，而非被一家包圆。
        """
        out: list[tuple[str, float]] = []
        depth = max((len(g) for g in groups), default=0)
        for i in range(depth):
            for g in groups:
                if i < len(g):
                    out.append(g[i])
        return out

    # ---------------- 对外检索 ----------------
    def search(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        r = self.cfg["retrieve"]
        top_k = top_k or r["final_top_k"]
        companies = detect_companies(question, self.cfg)
        panorama = is_panorama_question(question)
        year = self._detect_year(question)   # 问题里点名的报告期
        # 问题未点名年份时，默认收敛到项目设定的报告期（config.project.period）。
        # 否则“白洒行业竞争格局如何”这类问题会同时抱回 2023 和 2024 两期，
        # 答案里出现两年的数字而不自知，是最难被发现的一类错。
        default_year = str(self.cfg.get("project", {}).get("period") or "") or None
        soft_year = year or default_year

        vec = self.vector_search(question, r["vector_top_k"])
        bm = self.bm25_search(question, r["bm25_top_k"])
        # 问题点名了年份：年份是硬约束，必须在融合前就过滤，
        # 否则「2024年」的问题会混进 2023 年报的片段，跨年度题尤其致命。
        if year:
            keep = {c["chunk_id"] for c in self.chunks if c["report_period"] == year}
            vec = [(c, s) for c, s in vec if c in keep] or vec
            bm = [(c, s) for c, s in bm if c in keep] or bm
        elif soft_year:
            keep = {c["chunk_id"] for c in self.chunks if c["report_period"] == soft_year}
            v2 = [(c, s) for c, s in vec if c in keep]
            b2 = [(c, s) for c, s in bm if c in keep]
            if len(v2) >= 3 or len(b2) >= 3:      # 退一步：命中太少就不强求
                vec, bm = (v2 or vec), (b2 or bm)
        fused = self._rrf(vec, bm)

        if panorama:
            # 全景题：本行业所有公司都参与配额召回
            target = companies or [c["name"] for c in self.cfg["companies"]]
            groups = self._quota_search(question, target, year=soft_year)
            quota_hits = self._interleave(groups)   # ★ 轮询交错，保证覆盖均衡
            # 全景题必须放宽最终块数：配额是「每家公司 quota 块」，
            # 总量应为 公司数 × quota，否则按 final_top_k 截断会把每家的
            # 第 2/3 块全砍掉，只剩 1 块/家——而毛利率在表格、净利润在叙述文本，
            # 砍完之后汇总表就会有整列是空的。
            quota = int(r["per_company_quota"])
            pano_k = max(top_k, len(target) * quota)

            merged: list[tuple[str, float]] = []
            seen = set()
            for cid, s in quota_hits:
                merged.append((cid, s)); seen.add(cid)
            for cid, s in sorted(fused.items(), key=lambda x: -x[1]):
                if cid not in seen:
                    merged.append((cid, s)); seen.add(cid)
            ranked = merged[:pano_k]
            # 年份约束（全景题同样适用）
            if year:
                filtered = [(c, s) for c, s in ranked
                            if self.by_id.get(c, {}).get("report_period") == year]
                ranked = filtered or ranked
            covered = len({self.by_id[c]["company"] for c, _ in ranked if c in self.by_id})
            strategy = (f"跨公司配额召回（{len(target)}家 × {quota}块，"
                        f"轮询交错后取{pano_k}块，实际覆盖{covered}家）"
                        + (f"；报告期 {soft_year}" if soft_year else "") + " + RRF")
        else:
            if companies:   # 单公司题：先用公司名过滤，避免串公司
                fused = {cid: s for cid, s in fused.items()
                         if self.by_id.get(cid, {}).get("company") in companies} or fused
            ranked = [(cid, s) for cid, s in sorted(fused.items(), key=lambda x: -x[1])][:top_k]
            strategy = "混合检索（向量 + BM25 + RRF）" + ("＋公司过滤" if companies else "")
            if year:
                strategy += f"＋年份锁定{year}"
            elif soft_year:
                strategy += f"＋默认报告期{soft_year}"

        if r.get("use_reranker") and ranked:
            ranked = self._rerank(question, ranked, top_k)

        hits = []
        for cid, score in ranked:
            c = self.by_id.get(cid)
            if not c:
                continue
            hits.append({
                "chunk_id": cid,
                "score": round(score, 4),
                "company": c["company"], "code": c["code"],
                "report_period": c["report_period"], "report_type": c["report_type"],
                "section": c.get("section", ""), "page": c["page"],
                "block_type": c["block_type"], "table_title": c.get("table_title", ""),
                "unit": c.get("unit", "元"),
                "source_file": c.get("source_file", ""),
                "content": c["content"],
            })

        return {
            "question": question,
            "strategy": strategy,
            "is_panorama": panorama,
            "detected_companies": companies,
            "vector_top": [{"chunk_id": c, "score": round(s, 4)} for c, s in vec[:5]],
            "bm25_top": [{"chunk_id": c, "score": round(s, 4)} for c, s in bm[:5]],
            "hits": hits,
        }

    def _rerank(self, question: str, ranked: list[tuple[str, float]], top_k: int):
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(self.cfg["retrieve"]["reranker_model"])
        pairs = [(question, self.by_id[cid]["content"]) for cid, _ in ranked]
        scores = ce.predict(pairs)
        order = sorted(range(len(ranked)), key=lambda i: -scores[i])[:top_k]
        return [(ranked[i][0], float(scores[i])) for i in order]

    # ---------------- 生成 ----------------
    SYSTEM_PROMPT = """你是一个严谨的财报分析助手。请严格遵守以下规则：

1. 只依据「参考片段」回答，不得引入任何外部知识或常识推断。
2. 每个结论后面必须附上出处，格式：【公司 报告期 第X页】。
3. 涉及金额时，必须先声明金额单位与口径（如「合并报表口径、单位：万元」），再给出数字。
4. 若参考片段不足以回答，直接回答「未在检索到的片段中找到相关数据」，不得猜测。
5. 若问题涉及多家公司的比较或排序，先逐家公司列出数值（用表格），再给出比较结论。"""

    def build_prompt(self, question: str, hits: list[dict]) -> str:
        blocks = []
        for i, h in enumerate(hits, 1):
            blocks.append(
                f"[片段{i}] 公司={h['company']} 报告期={h['report_period']} "
                f"章节={h['section'] or '-'} 页码=P{h['page']} "
                f"类型={h['block_type']} 单位={h['unit']}\n{h['content']}"
            )
        context = "\n\n".join(blocks)
        return f"参考片段：\n{context}\n\n问题：{question}\n\n请作答："

    def generate(self, question: str, hits: list[dict]) -> str:
        provider = self.cfg["generate"]["provider"]
        prompt = self.build_prompt(question, hits)

        if provider == "none":
            return "（generate.provider = none：仅展示检索结果）"

        if provider == "demo":
            return self._demo_answer(question, hits)

        if provider == "local":
            return self._local_answer(question, hits)

        if provider == "ollama":
            import requests
            url = self.cfg["generate"]["ollama_base_url"].rstrip("/") + "/api/chat"
            payload = {
                "model": self.cfg["generate"]["ollama_model"],
                "messages": [{"role": "system", "content": self.SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.cfg["generate"]["temperature"]},
            }
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()   # 读环境变量 OPENAI_API_KEY
            r = client.chat.completions.create(
                model=self.cfg["generate"]["openai_model"],
                messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=self.cfg["generate"]["temperature"],
            )
            return r.choices[0].message.content

        return "（未知的 generate.provider）"

    # ---------------- local 模式（进程内本地小模型）----------------
    # 用 transformers 直接跑一个中文 instruct 小模型，不需要 API key、也不需要
    # 另起 ollama 服务。首次运行会从 HuggingFace 下载权重（1.5B 约 3GB）；
    # 国内网络可先设镜像：set HF_ENDPOINT=https://hf-mirror.com
    # 纯 CPU 推理，10 道题的评测量级可以接受；追求质量请改用 ollama / openai。
    _local_bundle = None

    def _local_answer(self, question: str, hits: list[dict]) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gen_cfg = self.cfg["generate"]
        model_name = resolve_model(
            gen_cfg.get("local_model", "Qwen/Qwen2.5-1.5B-Instruct"))

        if Retriever._local_bundle is None:
            device = pick_device()
            print(f"[local] 加载本地模型 {model_name}（首次需下载权重）→ 设备 {device} ...")
            tok = AutoTokenizer.from_pretrained(model_name)
            if device == "cuda":
                # 显卡只有 4GB，fp32 装不下 1.5B（6.2GB）；fp16 是 3.1GB，刚好。
                # 万一还是 OOM（上下文太长时 KV cache 会顶上去），退回 CPU fp32，
                # 慢是慢，但不能因为显存小就直接崩掉。
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, dtype=torch.float16).to("cuda")
                except Exception as e:
                    print(f"[local] 显存不足，退回 CPU：{type(e).__name__}")
                    device = "cpu"
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, dtype=torch.float32)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
            model.eval()
            Retriever._local_bundle = (tok, model, device)
        tok, model, device = Retriever._local_bundle

        # 全景题的上下文可能上百 KB，按模型上限截断，避免超长直接报错
        max_ctx = int(gen_cfg.get("local_max_context", 12288))
        text = tok.apply_chat_template(
            [{"role": "system", "content": self.SYSTEM_PROMPT},
             {"role": "user", "content": self.build_prompt(question, hits)}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_tokens", 1024)),
                do_sample=False,          # 财报问答要事实一致，不做采样
            )
        # .cpu() 是必须的：tokenizer.decode 内部要走 numpy，
        # CUDA 张量直接传进去会报「can't convert cuda:0 device type tensor to numpy」
        gen = out[0][inputs["input_ids"].shape[1]:].cpu()
        return tok.decode(gen, skip_special_tokens=True).strip()

    # ---------------- demo 模式（无需任何模型）----------------
    DEMO_EXT_HEAD = """**⚠️ 当前为 demo 模式（未接入大模型）**

下面是系统检索到的最相关原文片段。真实部署时，这些片段会连同强制引用出处的 Prompt
一起送给大模型（ollama / openai），由它合成带出处的完整答案。
"""

    def _demo_answer(self, question: str, hits: list[dict]) -> str:
        """抽取式答案：直接把最高分片段里的关键信息摆出来，附完整出处。"""
        if not hits:
            return "未在检索到的片段中找到相关数据。"

        hint = ""
        if self.is_panorama_q(question):
            hint = ("\n> 这是**跨公司全景题**：下方片段已按公司做了配额召回，"
                    "每家公司保底 N 块，可逐家核对后再作比较。\n")

        blocks = [hint, self.DEMO_EXT_HEAD]
        for i, h in enumerate(hits[:3], 1):
            content = h["content"]
            if len(content) > 320:
                content = content[:320].rstrip() + " …"
            blocks.append(
                f"**片段 {i}｜{h['company']} {h['report_period']}{h['report_type']} "
                f"· {h['section'] or '—'} · 第 {h['page']} 页**"
                f"（{h['block_type']}｜单位：{h['unit']}｜score={h['score']}）\n\n"
                f"```\n{content}\n```\n"
            )
        blocks.append("\n> 引用格式示例：【贵州茅台 2024 第16页】")
        return "\n".join(blocks)

    @staticmethod
    def is_panorama_q(q: str) -> bool:
        from utils import is_panorama_question
        return is_panorama_question(q)

    def answer(self, question: str, top_k: int | None = None) -> dict:
        trace = self.search(question, top_k)
        try:
            trace["answer"] = self.generate(question, trace["hits"])
        except Exception as e:
            trace["answer"] = f"⚠️ 生成失败：{e}\n（检索结果仍可正常查看）"
        return trace


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "贵州茅台2024年营业收入是多少？"
    r = Retriever()
    out = r.answer(q)
    print(f"\n策略：{out['strategy']}\n")
    print(out["answer"])
    print("\n--- 引用来源 ---")
    for h in out["hits"]:
        print(f"  · {h['company']} {h['report_period']} {h['section']} P{h['page']} "
              f"({h['block_type']}, score={h['score']})")
