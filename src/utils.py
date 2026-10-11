# -*- coding: utf-8 -*-
"""公共工具：配置加载、路径、公司实体识别、单位归一化、格式化。"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 config.yaml。"""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """把配置里的相对路径解析成项目根目录下的绝对路径。"""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def ensure_dir(p: str | Path) -> Path:
    d = resolve(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_model(name: str) -> str:
    """把模型名解析成「本地目录优先，否则 HuggingFace repo id」。

    config.yaml 里写的是可复现的 repo id（如 BAAI/bge-m3），评审 clone 下来
    由 huggingface_hub 自己会去下。但有些内网/镜像环境下 Python 客户端会卡死
    （证书、xet、并发都有关系），此时可以先用 `python tools/fetch_models.py`
    把权重 curl 到 models/<repo 末段>/，这里检测到本地目录就优先用它。

    判定标准是目录里存在 config.json——只有下了一半的目录不会被误用。
    """
    if os.path.isabs(name):
        return name
    d = ROOT / "models" / name.split("/")[-1]
    if (d / "config.json").exists():
        return str(d)
    return name


def pick_device() -> str:
    """有 CUDA 就用 CUDA，否则 CPU。

    这是个数量级的差别：bge-m3（568M 参数）在 16 核 CPU 上编码 7504 个 chunk
    实测要 ~14 小时（235 个 batch，每个 200~330 秒），换到 RTX 3050 上只要几分钟。
    但不能写死 cuda——评审的机器、Colab 免费层、纯 CPU 环境都得能跑起来。
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def company_names(cfg: dict) -> list[str]:
    return [c["name"] for c in cfg["companies"]]


def detect_companies(question: str, cfg: dict) -> list[str]:
    """从问题里识别涉及的公司。返回命中的公司简称列表（可能为空）。"""
    names = company_names(cfg)
    hits = [n for n in names if n in question]
    # 处理简称/别名，例如「茅台」->「贵州茅台」、「汾酒」->「山西汾酒」
    alias = {
        "茅台": "贵州茅台", "五粮液": "五粮液", "老窖": "泸州老窖",
        "汾酒": "山西汾酒", "洋河": "洋河股份", "古井": "古井贡酒",
        "迎驾": "迎驾贡酒", "口子": "口子窖", "舍得": "舍得酒业",
        "水井坊": "水井坊", "老白干": "老白干酒", "今世缘": "今世缘",
    }
    for kw, full in alias.items():
        if kw in question and full not in hits:
            hits.append(full)
    return hits


# 全景问题的信号词：出现这些，走「跨公司配额召回 + 先抽数值再比较」
PANORAMA_KEYWORDS = [
    "哪家", "哪些", "谁", "最高", "最低", "排名", "对比", "比较",
    "行业", "整体", "合计", "总和", "平均", "全部", "所有", "几家", "整体上",
]


def is_panorama_question(question: str) -> bool:
    return any(k in question for k in PANORAMA_KEYWORDS)


UNIT_PATTERN = re.compile(r"单位[:：]\s*(人民币)?\s*(万元|千元|亿元|元)")


def detect_unit(text: str) -> str:
    """从表格附近文本里识别金额单位，默认『元』。"""
    m = UNIT_PATTERN.search(text)
    if not m:
        return "元"
    return m.group(2)


def normalize_amount(value: float, unit: str) -> float:
    """统一折算为『元』，方便跨公司比较。"""
    factor = {"元": 1, "千元": 1e3, "万元": 1e4, "亿元": 1e8}
    return value * factor.get(unit, 1)


def drop_repeated_lines(pages: list[str], ratio: float = 0.5) -> list[str]:
    """剔除在多页重复出现的页眉页脚（如『贵州茅台2024年年度报告 第23页』）。"""
    from collections import Counter

    counter: Counter = Counter()
    for text in pages:
        lines = {ln.strip() for ln in text.splitlines() if 3 < len(ln.strip()) < 40}
        counter.update(lines)

    threshold = max(3, int(len(pages) * ratio))
    repeated = {ln for ln, c in counter.items() if c >= threshold}
    if not repeated:
        return pages

    cleaned = []
    for text in pages:
        keep = [ln for ln in text.splitlines() if ln.strip() not in repeated]
        cleaned.append("\n".join(keep))
    return cleaned


def table_to_markdown(rows: list[list[str | None]], header: list[str] | None = None) -> str:
    """二维数组 -> Markdown 表格。空单元格补空串，竖线转义。"""

    def cell(v: Any) -> str:
        s = "" if v is None else str(v).strip().replace("\n", " ").replace("|", "\\|")
        return s

    clean = [[cell(c) for c in row] for row in rows if any(cell(c) for c in row)]
    if not clean:
        return ""

    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]

    if header:
        head = [cell(c) for c in header] + [""] * (width - len(header))
        head = head[:width]
    else:
        head = clean[0]
        clean = clean[1:]

    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in clean]
    return "\n".join(lines)


def read_jsonl(path: str | Path):
    with open(resolve(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records) -> int:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def faiss_save(index, path: str | Path) -> Path:
    """写 FAISS 索引，绕开 Windows 中文路径。

    faiss 自带的 write_index 内部走 C 的 fopen，在 Windows 上按 **ANSI 代码页**
    解释路径；只要路径里含中文（比如项目放在 `...\\作业\\...`），就会报
    `could not open ... index.faiss for writing: No such file or directory`
    —— 而那个目录其实好好地在那儿，报错信息极具误导性。

    改法：serialize_index 先变成 numpy 字节数组，再交给 Python 的 open() 写。
    Python 走的是宽字符 API，中文路径没问题；文件格式和 write_index 完全一致，
    两个函数可以互相读对方写的文件。
    """
    import faiss
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = faiss.serialize_index(index)
    with open(p, "wb") as f:
        f.write(data.tobytes())
    return p


def faiss_load(path: str | Path):
    """读 FAISS 索引，与 faiss_save 配对（同样为了中文路径能读）。"""
    import faiss
    import numpy as np
    with open(resolve(path), "rb") as f:
        data = np.frombuffer(f.read(), dtype="uint8").copy()
    return faiss.deserialize_index(data)


def context_header(rec: dict) -> str:
    """生成块首上下文头：【公司 代码 · 报告期 · 章节 · P页码】"""
    parts = [f"{rec.get('company', '')} {rec.get('code', '')}".strip(),
             rec.get("report_period", ""),
             rec.get("report_type", "")]
    head = " · ".join([p for p in parts if p])
    tail = []
    if rec.get("section"):
        tail.append(rec["section"])
    if rec.get("page"):
        tail.append(f"P{rec['page']}")
    if rec.get("table_title"):
        tail.append(f"表:{rec['table_title']}")
    if tail:
        head += " · " + " · ".join(tail)
    return f"【{head}】"


def fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def env_or(default: str, *keys: str) -> str:
    """按顺序取第一个存在的环境变量。"""
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return default
