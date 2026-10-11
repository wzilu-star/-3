# -*- coding: utf-8 -*-
"""把模型权重预取到本地 models/ 目录（可选，用于 hf_hub_download 走不通的网络环境）。

平时**不需要**跑这个脚本：`04_index.py` / `05_rag.py` 直接用 HuggingFace repo id
（`BAAI/bge-m3`、`Qwen/Qwen2.5-1.5B-Instruct`）自己下载。

什么时候需要它：
    实测某些国内镜像环境下，hf-mirror.com 用 curl 能跑到 8MB/s，
    而 Python 的 hf_hub_download 客户端会在同一个 URL 上卡住不动
    （缓存停在某个字节数不再增长，换文件、关 xet 都没用）。
    这时用本脚本绕开 Python 客户端，直接把 curl 当下载器：
      - `-C -` 断点续传：中断后重跑不会从 0 开始
      - `--retry-all-errors`：连接被掐断时自动重试
    下载到 models/<repo 末段>/ 后，`src/utils.py:resolve_model()` 会优先用本地目录，
    config.yaml 一个字都不用改。

用法：
    set HF_ENDPOINT=https://hf-mirror.com     # 国内加速，可省
    python tools/fetch_models.py

下载目标：
    models/bge-m3/                    ← 对应 BAAI/bge-m3            （约 2.3GB）
    models/Qwen2.5-1.5B-Instruct/     ← 对应 Qwen/Qwen2.5-1.5B-Instruct（约 3.1GB）
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ENDPOINT = "https://hf-mirror.com"
ROOT = Path(__file__).resolve().parent.parent   # 项目根目录
CURL = r"C:\Windows\System32\curl.exe"

# (HF repo, 本地目录名)
TARGETS = [
    ("BAAI/bge-m3", "bge-m3"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b-instruct"),
]

# 这些子目录用不到，省几百 MB 和一半的文件数
SKIP_PREFIX = ("onnx/", "imgs/", ".git")


def list_files(repo: str) -> list[str]:
    resp = requests.get(f"{ENDPOINT}/api/models/{repo}", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [
        s["rfilename"]
        for s in data.get("siblings", [])
        if not s["rfilename"].startswith(SKIP_PREFIX)
    ]


def fetch(repo: str, rel: str, dest: Path) -> bool:
    url = f"{ENDPOINT}/{repo}/resolve/main/{rel}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        CURL, "-L", "-C", "-", "--retry", "8", "--retry-all-errors",
        "--retry-delay", "3", "--connect-timeout", "20",
        "--speed-limit", "2000", "--speed-time", "60",   # 60s 内低于 2KB/s → 判死并重试
        "-o", str(dest), url,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    ok = r.returncode == 0 and dest.exists()
    if ok:
        mb = dest.stat().st_size / 1e6
        print(f"   [OK] {rel}  {mb:.1f}MB  {time.time() - t0:.0f}s", flush=True)
    else:
        print(f"   [FAIL] {rel}  rc={r.returncode}", flush=True)
        print((r.stderr or r.stdout or "")[-400:], flush=True)
    return ok


def already_complete(repo: str, rel: str, dest: Path) -> bool:
    """本地文件大小和远端一致就算下完了（curl -C - 对完整文件会直接返回 0）。"""
    try:
        r = requests.head(f"{ENDPOINT}/{repo}/resolve/main/{rel}",
                          allow_redirects=True, timeout=30)
        want = int(r.headers.get("Content-Length", "0"))
    except Exception:
        return False
    return dest.exists() and want > 0 and dest.stat().st_size == want


def main() -> int:
    for repo, local in TARGETS:
        dest_dir = ROOT / "models" / local
        files = list_files(repo)
        print(f"\n=== {repo} → models/{local}/ ：{len(files)} 个文件 ===", flush=True)

        failed = []
        for i, rel in enumerate(files, 1):
            dest = dest_dir / rel
            if already_complete(repo, rel, dest):
                print(f"   [skip] ({i}/{len(files)}) {rel}", flush=True)
                continue
            print(f"   ... ({i}/{len(files)}) {rel}", flush=True)
            if not fetch(repo, rel, dest):
                failed.append(rel)

        (dest_dir / "_source_repo.json").write_text(
            json.dumps({"repo": repo, "files": files}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        if failed:
            print(f"[失败] {repo}: {failed}", flush=True)
            return 1
        print(f"[完成] {repo} → {dest_dir}", flush=True)

    print("\n两个模型都已就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
