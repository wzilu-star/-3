# 白酒行业财报 RAG 问答知识库

> 从交易所/巨潮下载 12 家白酒上市公司年报 → 抽取文字与表格（按行列还原）→ 带元数据切块 → 向量 + BM25 混合索引 → 带出处的问答页面 → 10 题评测（含跨公司全景题）。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **零依赖先看效果**：没下 PDF、没装 faiss 也能把整条链路跑一遍——
> `python tools/run_demo.py` 会用仿真数据生成问答页面预览（`docs/demo_preview.html`）。
>
> **不知道怎么跑？看 →** [`docs/怎么部署怎么启动.md`](docs/怎么部署怎么启动.md)
> 从装 Python 开始，A 本地零依赖 / B 本地真实数据 / C GitHub+Colab 上云，三条路照着敲即可。

---

## 一、它做了什么

```
① 下载年报          12 家白酒公司 × 2024 年报（巨潮资讯网）
        ↓
② 抽取文字与表格    pdfplumber 逐页抽表 → Markdown 表格（行列不丢），跨页自动补表头
        ↓
③ 带元数据切块      【公司·代码·报告期·章节·页码】+ 正文；表格整表成块
        ↓
④ 双索引            向量索引（bge-m3 + FAISS） + BM25（jieba + 财报词典）
        ↓
⑤ 混合检索          RRF 融合；全景题走「跨公司配额召回」
        ↓
⑥ 问答页面          答案 + 出处（公司/章节/页码）+ 检索过程可视化
        ↓
⑦ 评测              10 题（含 3 道跨公司全景题），逐题记录召回块与错因
```

## 二、四个关键设计（决定成败）

| 设计 | 解决的问题 |
|---|---|
| **表格按行列还原成 Markdown**，整表成块 | 普通抽取会把表格压成一行乱序文字，行列全丢；财报的信息主体恰恰是表格 |
| **块首注入上下文头** `【贵州茅台 600519 · 2024年报 · 第三节 · P23】` | 检索与生成都看得见来源，跨公司题不会串号 |
| **跨公司配额召回**（每家公司保底 N 块） | 普通 top-k 会被单家公司霸榜，全景题必错 |
| **单位口径显式化**（万元/元写进元数据与片段） | 避免「语义对、数字错」这种最隐蔽的错误 |

## 三、目录结构

```
baijiu-finrag/
├─ config.yaml               ← 改公司清单 / 参数只改这个文件
├─ requirements.txt
├─ src/
│  ├─ utils.py               公共工具（配置、路径、公司识别、单位归一化）
│  ├─ 01_download.py         下载年报 + 生成 manifest.csv（含来源 URL、sha256）
│  ├─ 02_extract.py          抽取文字 + 表格（按行列还原）
│  ├─ 03_chunk.py            语义切块 + 上下文头注入
│  ├─ 04_index.py            建索引（faiss 后端 / light 后端，自动选择）
│  ├─ light_index.py         零依赖的 TF-IDF + BM25 轻量检索后端
│  ├─ 05_rag.py              混合检索 / RRF / 跨公司路由 / 生成
│  └─ 06_eval.py             批量评测 10 题
├─ tools/
│  ├─ make_demo_data.py      生成仿真数据（无需下载 PDF）
│  ├─ fetch_models.py        用 curl 预取模型权重到 models/（HF 客户端卡住时的备选）
│  ├─ mode_compare.py        三种检索模式（向量/BM25/混合）对比 → eval/mode_compare.md
│  ├─ md_to_pdf.py           把「结论_一页.md」导成 PDF（借系统 Edge 排版，`--compact` 压进单页）
│  ├─ shoot.py               自动截问答页面（Playwright 驱动系统 Edge）
│  └─ run_demo.py            一键演示 + 渲染页面预览 HTML
├─ app/streamlit_app.py      问答页面
├─ notebooks/colab_deploy.ipynb  Colab 一键部署
├─ eval/
│  ├─ questions.yaml         10 道评测题
│  ├─ 结论_一页.md / .pdf     交付物③ 一页结论（正文严格排在一页内）
│  ├─ 真值核对.md             10 题真值逐条抄自年报原文，是判对错的依据
│  ├─ results_*.md           逐题原始记录：召回块 / 答案 / 对错 / 错因
│  └─ mode_compare.md        三种检索模式的召回对比
├─ demo/                     仿真数据与演示结果
├─ docs/
│  ├─ index.html             项目展示页（可挂 GitHub Pages）
│  ├─ demo_preview.html      问答页面预览（由 run_demo.py 生成）
│  └─ screenshots/           页面截图
└─ data/ index/              运行产物（不进 Git）
```

## 三·五、两种运行路径

### 路径 A：先看效果（1 分钟，零依赖）

```bash
python tools/run_demo.py            # 自动造仿真数据 + 建轻量索引 + 渲染预览
```

然后用浏览器打开 `docs/demo_preview.html`。这条路径**不需要** faiss、sentence-transformers，
也**不需要**下载任何 PDF，用来先确认「页面长什么样、检索效果如何」。

> 仿真数据是脚本按行业量级模拟生成的，**数字不是真的**；但抽取格式、切块结构、
> 检索链路、出处处全是真实的。换成真年报时，除了数据不同，其他完全一样。

### 路径 B：跑真实年报（完整流程）

见下一节。

## 四、快速开始

```bash
git clone https://github.com/wzilu-star/-3.git baijiu-finrag
cd baijiu-finrag
pip install -r requirements.txt

python src/01_download.py    # 下载 12 家年报（约 200–400MB）
python src/02_extract.py     # 抽取文字与表格 —— 检查输出里每家公司「表格块」是否 > 0
python src/03_chunk.py       # 切块
python src/04_index.py       # 建索引（faiss 后端会下载 bge-m3 权重 ~2GB）
python src/06_eval.py        # 跑 10 道评测题

streamlit run app/streamlit_app.py
```

若已装好依赖但暂时不想下 2GB 权重，可以先强制用轻量后端把页面跑起来：

```bash
python src/04_index.py --backend light   # 秒级完成，TF-IDF + BM25
streamlit run app/streamlit_app.py
```

**权重下载卡住**（国内常见：日志停在 `[编码] 模型 BAAI/bge-m3` 不动，而网速其实没问题）——
换用 curl 下载，下到本地 `models/` 后 `src/utils.py:resolve_model()` 会自动优先用本地目录，
`config.yaml` 不用改：

```bash
set HF_ENDPOINT=https://hf-mirror.com
python tools/fetch_models.py    # 支持断点续传；models/ 已在 .gitignore 里
```

> `index.backend` 支持 `auto`（默认，有 faiss 就用 faiss，否则自动降级 light）。
> **light 不是语义向量**，不做同义改写泛化，只适合演示与联调；正式评测请用 faiss。

**建索引前先确认 torch 是 CUDA 版**——这一步决定你是等 5 分钟还是等 14 小时。
PyPI 上的 torch 是 CPU 版，而 bge-m3 有 568M 参数：

```bash
python -c "import torch; print(torch.cuda.is_available())"      # 不是 True 的话↓
pip install torch==<当前版本>+cu130 --index-url https://download.pytorch.org/whl/cu130
```

实测：16 核 CPU 编码 7504 个 chunk 要 ~14 小时（235 batch × 200~330s），
RTX 3050 4GB 上几分钟。设备选择在 `src/utils.py:pick_device()`，没有 N 卡就换
`BAAI/bge-small-zh-v1.5`，不要硬扛。

**显存只有 4GB 也能跑**：`embed_batch_size` 撞到 CUDA OOM 时 `04_index.py` 会自动减半重试，
不用手改配置。同理 `generate.max_tokens` 对小卡影响很大——生成速度约 1.7 token/s（RTX 3050 fp16），
1024 token 要 10 分钟，512 只要 5 分钟，而这批题答案最长也就几百 token。

> **项目路径别带中文。** faiss 的 `write_index`/`read_index` 内部用 C 的 `fopen`，
> 在 Windows 上按 ANSI 代码页解释路径，路径含中文会报
> `could not open ...index.faiss for writing: No such file or directory`（而目录其实存在）。
> 本项目已用 `src/utils.py:faiss_save()/faiss_load()` 绕开，正常流程不会踩到。

若网络受限、下载脚本失败，**用浏览器手工下载 PDF** 放到 `data/raw/<公司名>_<代码>/2024_年报.pdf` 即可，后续流程完全一样：

```
data/raw/贵州茅台_600519/2024_年报.pdf
data/raw/五粮液_000858/2024_年报.pdf
...
```

## 五、云端部署（Colab + 公网链接）

打开 `notebooks/colab_deploy.ipynb`，从上到下运行。

要点：
1. **浏览器打开 GitHub 上的 notebook**，点顶部的 `Open in Colab` 徽标即可以免配置方式运行；
2. 第 1 格把 `REPO` 改成你自己的仓库地址；
3. 第 8 格用 `cloudflared` 起一个临时公网隧道，输出的 `https://xxxx.trycloudflare.com` **就是你截图要用的页面地址**；
4. 索引体积大、不要在 Colab 上传回 GitHub；需要复用时按第 10 格存到 Google Drive。

> Streamlit 官方也有免费托管：把仓库连到 share.streamlit.io，入口文件填 `app/streamlit_app.py`。
> 注意免费层磁盘与内存有限，建议只放少量公司的数据，或把索引预先构建好再提交。

## 六、评测设计

10 道题的类型配比（见 `eval/questions.yaml`）：

| 类型 | 数量 |
|---|---|
| 单公司 · 单指标抽取 | 3 |
| 单公司 · 表格多指标 | 2 |
| 单公司 · 跨年度对比 | 2 |
| **跨公司 · 全景/比较（要求 ≥2）** | **3** |

每题记录：**召回了哪些块（chunk_id + 公司·页码）→ 答案 → 对错 → 错误归因**。

错因使用固定枚举，便于统计定位短板：

`未召回` / `召回但被忽略` / `表格切碎` / `单位口径混淆` / `跨公司未聚合` / `检索被单公司霸榜` / `数字算错`

对错分四档，**不把「数字对但口径错」算作正确**：
`✅正确` / `⚠️数字对但口径错` / `❌错` / `🚫拒答`

## 七、生成模型配置

`config.yaml` 里 `generate.provider` 五选一：

| 值 | 说明 |
|---|---|
| `local` | **默认**。用 `transformers` 在进程内跑 `Qwen/Qwen2.5-1.5B-Instruct`，**不需要 API key、也不用起 ollama 服务**，纯 CPU 可跑。首次运行从 HuggingFace 下权重（约 3GB），国内网络先 `set HF_ENDPOINT=https://hf-mirror.com` |
| `demo` | 无需任何模型，把最相关片段直接摆出来并附出处，用于零依赖演示 |
| `ollama` | 本地模型，默认 `qwen2.5:7b`，需先 `ollama serve`。质量比 `local` 好，推荐正式演示用 |
| `openai` | 读环境变量 `OPENAI_API_KEY`，或任何兼容 OpenAI 协议的服务 |
| `none` | 只展示检索结果，不做任何加工 |

> `local` 这个后端的意义是**可复现**：评审 clone 下来不用配任何密钥就能看到「带出处的真实回答」，
> 而不是只看到检索片段。1.5B 模型在跨公司聚合题上会明显吃力——这本身就是结论页要分析的现象之一。
>
> 权重来源：`config.yaml` 里写的是 HuggingFace repo id。若本地 `models/<repo 末段>/` 已存在
> （用 `tools/fetch_models.py` 下的），`src/utils.py:resolve_model()` 会自动改用它——
> 所以同一份配置在有网和没网、镜像通和镜像不通的机器上都能跑。

## 七·五、检索参数里最容易配错的一处

`retrieve.panorama_top_k` / `per_company_quota` 决定跨公司全景题的成败：

```yaml
per_company_quota: 3      # 每家公司保底召回几块
panorama_top_k: 36        # 全景题最终送进模型的块数
```

**必须满足：`panorama_top_k ≥ 公司数 × per_company_quota`**。
否则会被截断——12 家公司各召 3 块共 36 块，若最终仍按 `final_top_k=6` 截断，
结果只剩前两家公司，恰好复现了配额本来要解决的「单公司霸榜」问题。

另外配额结果必须**轮询交错**后再截断（先各公司第 1 块、再各公司第 2 块……），
而不是按公司顺序拼接——否则被截断的永远是排在后面的公司。

这两点在 `05_rag.py` 的 `_quota_search` / `_interleave` 里实现，值得对照代码读一遍。

## 八、数据与版权

年报均为上市公司依法公开披露的文件，来源：巨潮资讯网 / 上海证券交易所 / 深圳证券交易所。
本仓库**不包含**原始 PDF 与索引产物（见 `.gitignore`），请通过脚本自行下载。

## 九、交付物对照

| 交付项 | 对应位置 |
|---|---|
| 代码仓库 | 本仓库（含 README 与完整复现步骤） |
| 页面截图 | `docs/screenshots/`：`01_单公司问答.png`、`02_引用来源_表格块.png`、`03_全景题_12家.png` |
| 一页结论 | `eval/结论_一页.md`（PDF 版 `eval/结论_一页.pdf`，正文严格排在一页内） |
| 原始问答与证据记录 | `eval/results_20261011_204441.md`（逐题）、`eval/真值核对.md`（真值）、`eval/mode_compare.md`（召回对比） |

### 一页结论怎么重新生成

```cmd
python tools/md_to_pdf.py eval/结论_一页.md --compact
```

`--compact` 把字号 / 行距 / 页边距各收紧一档——正文一多就会溢到第二页，
而交付要求是「结论单独压缩为一页」。不传这个开关时，同一份 `.md` 会排到 2 页，
上一版（内容更冗长的版本）按默认版式排到了 3 页。导完可自检页数：

```cmd
python -c "import pymupdf;print(pymupdf.open('eval/结论_一页.pdf').page_count)"
```
