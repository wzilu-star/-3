# -*- coding: utf-8 -*-
"""
01 · 下载白酒上市公司年报/半年报

用法：
    python src/01_download.py                 # 按 config.yaml 下载全部公司
    python src/01_download.py --only 贵州茅台   # 只下一家（调试用）
    python src/01_download.py --check          # 只检查已下载文件是否有效

数据源优先级（自动降级）：
    1) 巨潮资讯网 cninfo.com.cn  —— 证监会指定披露平台，最全最稳（首选）
    2) 上交所 / 深交所官网静态 PDF 直链（在 manual_urls.csv 里手工补）
    3) 手工下载：直接用浏览器把 PDF 丢进 data/raw/<公司>_<代码>/ 即可

注意：
  · 首次在沙箱运行需要先授权域名 cninfo.com.cn（或直接在浏览器手工下载，效果一样）
  · 下载会写 data/raw/manifest.csv 记录来源 URL 与 sha256，供报告溯源
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_dir, fmt_size, load_config, resolve  # noqa: E402

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_SEARCH = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"

# 报告类型 -> 公告标题里的用词。任务里的「年报」在标题里写的是「年度报告」，
# 「半年报」写的是「半年度报告」，直接用 report_type 去匹配标题会永远匹配不上。
TITLE_WORD = {"年报": "年度报告", "半年报": "半年度报告"}


def resolve_org_id(code: str, session: requests.Session) -> str:
    """用代码换巨潮的 orgId。

    新版巨潮查询接口要求 stock 参数是「代码,orgId」（如 600519,gssh0600519），
    只传代码会返回 totalRecordNum=0——不报错、但一条都查不到，
    这是本项目下载脚本原先静默失败的原因。
    """
    resp = session.post(CNINFO_SEARCH, data={"keyWord": code, "maxNum": 10}, timeout=20)
    resp.raise_for_status()
    for item in resp.json() or []:
        if item.get("code") == code:
            return item.get("orgId") or ""
    return ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def is_valid_pdf(path: Path, min_bytes: int) -> bool:
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    with open(path, "rb") as f:
        return f.read(5).startswith(b"%PDF")


def query_cninfo(code: str, org_id: str, period: str, report_type: str,
                 session: requests.Session) -> list[dict]:
    """向巨潮查询某公司的公告列表。返回 [{title, adjunctUrl, announcementTime}]。

    两个易错点：
      1) stock 必须是「代码,orgId」，只传代码会静默返回 0 条；
      2) seDate 过滤的是**公告披露日**，不是报告期。2024 年报要到 2025 年 4 月
         才披露，所以按 2024 去查永远查不到。这里把区间放宽到
         [报告期年初, 次年年底]，年报（次年披露）和半年报（当年披露）都能覆盖，
         真正按报告期筛选交给下面的 pick_best。
    """
    category = "category_ndbg_szsh" if report_type == "年报" else "category_bndbg_szsh"
    end_year = int(period) + 1
    payload = {
        "pageNum": 1,
        "pageSize": 30,
        "column": "szse",
        "tabName": "fulltext",
        "stock": f"{code},{org_id}",
        "searchkey": "",
        "secid": "",
        "plate": "",
        "category": category,
        "trade": "",
        "seDate": f"{period}-01-01~{end_year}-12-31",
        "isHLtitle": "true",
    }
    resp = session.post(CNINFO_QUERY, data=payload, timeout=20)
    resp.raise_for_status()
    return resp.json().get("announcements") or []


def pick_best(items: list[dict], company: str, period: str, report_type: str) -> dict | None:
    """从公告列表里挑出最匹配的一条（标题含报告期，且不是摘要/英文版/更正）。

    注意：**不要**再要求标题里出现公司名。查询已经用 `stock=代码,orgId` 限定了
    公司，返回的每条公告都属于该公司；而各家的标题写法并不统一——
    有的写「江苏今世缘酒业股份有限公司2024年年度报告」，有的只写「2024年年度报告」。
    按公司名过滤会把后者（五粮液、泸州老窖、山西汾酒、洋河股份、古井贡酒等）
    整批误杀，表现为「未找到匹配公告」。
    """
    bad_words = ["摘要", "英文", "English", "更正", "补充", "问询", "反馈", "取消", "说明"]
    title_word = TITLE_WORD.get(report_type, report_type)
    best = None
    for it in items:
        title = (it.get("announcementTitle") or "").replace("<em>", "").replace("</em>", "")
        if period not in title or title_word not in title:
            continue
        if any(w in title for w in bad_words):
            continue
        best = {**it, "title": title}
        break
    return best


def download_one(company: dict, cfg: dict, session: requests.Session, overwrite: bool = False) -> dict:
    name, code = company["name"], company["code"]
    period = str(cfg["project"]["period"])
    report_type = cfg["project"]["report_type"]
    min_bytes = cfg["extract"]["min_pdf_bytes"]

    out_dir = ensure_dir(Path(cfg["paths"]["raw_dir"]) / f"{name}_{code}")
    out_file = out_dir / f"{period}_{report_type}.pdf"

    # local_file 存相对项目根目录的路径：manifest.csv 是要进仓库的，
    # 存绝对路径会把「C:\Users\<用户名>\...」写进仓库，既泄露本机信息又换台机器就对不上。
    try:
        rel_file = out_file.relative_to(resolve(".")).as_posix()
    except ValueError:
        rel_file = out_file.name

    rec = {"company": name, "code": code, "report_type": report_type,
           "report_period": period, "source_url": "", "local_file": rel_file, "sha256": ""}

    has_pdf = is_valid_pdf(out_file, min_bytes)

    # 即使本地已有 PDF，也照常查一次来源 URL —— manifest 的价值就在「这份 PDF 从哪来」，
    # 直接写「(已存在)」等于把溯源信息丢了。
    print(f"  [查询] 巨潮资讯网 {name}({code}) ...")
    try:
        org_id = resolve_org_id(code, session)
        if not org_id:
            print(f"  [失败] 未能取到 orgId（{code}）——核对代码是否正确")
            rec["source_url"] = "(本地已有，查询失败)" if has_pdf else ""
        else:
            items = query_cninfo(code, org_id, period, report_type, session)
            hit = pick_best(items, name, period, report_type)
            if hit:
                rec["source_url"] = CNINFO_STATIC + (hit.get("adjunctUrl") or "")
            else:
                print(f"  [失败] 未找到匹配公告（{period} {report_type}）")
                rec["source_url"] = "(本地已有，未匹配到公告)" if has_pdf else ""
    except Exception as e:  # 网络不通时给出明确指引
        print(f"  [失败] 查询巨潮失败：{e}")
        print("         → 请授权域名 cninfo.com.cn 重试，或直接用浏览器手工下载 PDF 到：")
        print(f"           {out_dir}")
        rec["source_url"] = "(本地已有，查询失败)" if has_pdf else ""
        if has_pdf:
            rec["sha256"] = sha256_of(out_file)
        return rec

    if has_pdf and not overwrite:
        print(f"  [跳过] 已存在有效 PDF（{fmt_size(out_file.stat().st_size)}）")
        rec["sha256"] = sha256_of(out_file)
        return rec

    if not rec["source_url"].startswith(CNINFO_STATIC):
        return rec

    url = rec["source_url"]
    print(f"  [下载] {url}")
    try:
        with session.get(url, headers={"User-Agent": UA}, timeout=120, stream=True) as r:
            r.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
    except Exception as e:
        print(f"  [失败] 下载出错：{e}")
        return rec

    if is_valid_pdf(out_file, min_bytes):
        rec["sha256"] = sha256_of(out_file)
        print(f"  [成功] {fmt_size(out_file.stat().st_size)} → {out_file.name}")
    else:
        print("  [失败] 下载到的不是有效 PDF（可能是 HTML 错误页）")
        out_file.unlink(missing_ok=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只处理某家公司（按简称匹配）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="只检查已下载文件有效性")
    args = ap.parse_args()

    cfg = load_config()
    min_bytes = cfg["extract"]["min_pdf_bytes"]
    companies = cfg["companies"]
    if args.only:
        companies = [c for c in companies if args.only in c["name"]]
        if not companies:
            sys.exit(f"未在 config.yaml 找到公司：{args.only}")

    if args.check:
        ok = bad = 0
        for c in cfg["companies"]:
            p = resolve(Path(cfg["paths"]["raw_dir"]) / f"{c['name']}_{c['code']}"
                        / f"{cfg['project']['period']}_{cfg['project']['report_type']}.pdf")
            if is_valid_pdf(p, min_bytes):
                print(f"  ✔ {c['name']:<8} {fmt_size(p.stat().st_size)}")
                ok += 1
            else:
                print(f"  ✘ {c['name']:<8} 缺失或无效")
                bad += 1
        print(f"\n有效 {ok} 份 / 无效 {bad} 份")
        return

    ensure_dir(cfg["paths"]["raw_dir"])
    session = requests.Session()
    session.headers.update({
        "User-Agent": UA,
        # 巨潮的查询接口校验 Referer/X-Requested-With，缺了会返回空结果
        "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
        "X-Requested-With": "XMLHttpRequest",
    })

    records = []
    for c in companies:
        print(f"\n=== {c['name']} ({c['code']}) ===")
        records.append(download_one(c, cfg, session, args.overwrite))
        time.sleep(1.5)  # 礼貌爬取

    manifest = resolve(cfg["paths"]["manifest_path"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    ok = sum(1 for r in records if r["sha256"] and is_valid_pdf(resolve(r["local_file"]), min_bytes))
    print(f"\n完成：成功 {ok}/{len(records)} 份，manifest → {manifest}")
    if ok < 10:
        print("提示：少于 10 份时可用浏览器手工下载缺失的 PDF 到 data/raw/<公司>_<代码>/ 后重跑。")


if __name__ == "__main__":
    main()
