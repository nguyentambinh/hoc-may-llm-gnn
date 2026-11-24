# Nguyễn Tâm Bình - 2324801030003
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "r2vul"
OUT = ROOT / "BaoCao" / "hinh"
NGON_NGU = [("c", "C"), ("c_sharp", "C#"), ("java", "Java"), ("javascript", "JavaScript"), ("python", "Python")]
MAU = ["#33419B", "#43ADE2", "#E17C3C", "#549F76", "#8F6BB0"]
NGUON = [("vuln", 1), ("removed", 1), ("unchanged", 0), ("fixed", 0), ("added", 0)]
BO_CWE = ("NVD-CWE-noinfo", "NVD-CWE-Other")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13})
so = lambda v, n=1: f"{v:,.{n}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def doc_du_lieu():
    dem = {lang: Counter() for lang, _ in NGON_NGU}
    dong = {lang: {0: [], 1: []} for lang, _ in NGON_NGU}
    cwe = defaultdict(Counter)
    repo = defaultdict(Counter)
    nguon = defaultdict(Counter)
    chia = Counter()
    nhan_nguon = dict(NGUON)
    for split in ("train", "valid", "test"):
        with (DATA / f"{split}.jsonl").open(encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                r = json.loads(line)
                lang, target, meta = r["lang"], r["target"], r["meta"]
                if lang not in dem or target not in (0, 1) or r["split"] != split:
                    raise ValueError(f"Ngôn ngữ, nhãn hoặc tập chia không hợp lệ: {split}, dòng {i}")
                origin = meta["origin"]
                if origin not in nhan_nguon or nhan_nguon[origin] != target:
                    raise ValueError(f"Nguồn gốc không khớp nhãn: {split}, dòng {i}")
                if not isinstance(meta["cwe_id"], list):
                    raise ValueError(f"CWE phải là danh sách: {split}, dòng {i}")
                dem[lang][target] += 1
                dong[lang][target].append(r["func"].count("\n") + 1)
                chia[split] += 1
                repo[meta["repo"]][lang] += 1
                nguon[origin][lang] += 1
                if target == 1:
                    for ma in set(meta["cwe_id"]):
                        cwe[ma][lang] += 1
    for lang, _ in NGON_NGU:
        if not dong[lang][0] or not dong[lang][1]:
            raise ValueError(f"Thiếu hàm lỗi hoặc an toàn cho ngôn ngữ {lang}")
    phan_vi = {lang: {k: np.percentile(v, (50, 90, 99)) for k, v in nhan.items()} for lang, nhan in dong.items()}
    return dem, phan_vi, cwe, repo, nguon, chia


def doc_do_thi(dem):
    raw = ROOT / "data/graphs/r2vul/raw"
    paths = sorted(raw.glob("batch_*.jsonl"))
    if not paths or any(not path.with_suffix(".done").exists() for path in paths):
        raise ValueError("Cần chạy xong Joern trước khi vẽ thống kê đồ thị")
    ids = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            ids.update(json.loads(line)["id"] for line in stream if line.strip())
    counts = {lang: Counter() for lang, _ in NGON_NGU}
    for split in ("train", "valid", "test"):
        with (DATA / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                counts[row["lang"]][row["target"]] += row["id"] in ids
    thieu = {lang: {label: (dem[lang][label] - counts[lang][label]) / dem[lang][label] * 100 for label in (0, 1)} for lang, _ in NGON_NGU}
    return thieu, sum(sum(v.values()) for v in counts.values()), sum(sum(v.values()) for v in dem.values())


def dinh_dang(ax, ngang=False, n=0):
    ax.grid(axis="x" if ngang else "y", color="#EEEEEE")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    truc = ax.xaxis if ngang else ax.yaxis
    truc.set_major_formatter(FuncFormatter(lambda v, _: so(v, n)))


def luu(fig, tep, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / tep, dpi=300)
    plt.close(fig)


def chu_giai_ngon_ngu(fig):
    fig.legend(handles=[Patch(color=MAU[i], label=ten) for i, (_, ten) in enumerate(NGON_NGU)], frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1))


def cot_nhom(gia_tri, nhom, nhan_y, tep, ymax, n=0):
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    x = np.arange(len(nhom))
    w = 0.4
    for k, ten in enumerate(("An toàn", "Lỗi")):
        v = [g[k] for g in gia_tri]
        b = ax.bar(x + (k - 0.5) * (w + 0.02), v, w, color=MAU[[0, 2][k]], label=ten)
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + ymax * 0.015, so(val, n), ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x, nhom)
    ax.set_ylabel(nhan_y)
    ax.set_ylim(0, ymax)
    dinh_dang(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    luu(fig, tep)


def ve_so_ham(dem):
    nhom = [f"{ten}\n{so(dem[lang][1] / sum(dem[lang].values()) * 100, 2)}% lỗi" for lang, ten in NGON_NGU]
    v = [[dem[lang][k] for k in (0, 1)] for lang, _ in NGON_NGU]
    ymax = np.ceil(max(max(g) for g in v) * 1.2 / 1000) * 1000
    cot_nhom(v, nhom, "Số hàm", "r2vul_2_1_so_ham.png", ymax)


def ve_do_dai(phan_vi):
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.8), sharey=True)
    x = np.arange(len(NGON_NGU))
    ymax = 10 ** np.ceil(np.log10(max(max(v) for lang in phan_vi.values() for v in lang.values()) * 1.15))
    for target, ax in enumerate(axes):
        for k, ten in enumerate(("p50", "p90", "p99")):
            v = [phan_vi[lang][target][k] for lang, _ in NGON_NGU]
            ax.bar(x + (k - 1) * 0.26, v, 0.24, color=MAU[k], label=ten)
        ax.set_title("Hàm an toàn" if target == 0 else "Hàm lỗi", fontsize=13)
        ax.set_xticks(x, [ten for _, ten in NGON_NGU], rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_ylim(0.8, ymax)
        dinh_dang(ax)
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Số dòng mỗi hàm (thang log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1))
    luu(fig, "r2vul_2_2_do_dai.png", rect=(0, 0, 1, 0.91))


def top_muoi(dem, bo=()):
    return sorted((k for k in dem if k not in bo), key=lambda k: (-sum(dem[k].values()), k))[:10]


def ve_top_cwe(cwe):
    top = top_muoi(cwe, BO_CWE)
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for i, (lang, _) in enumerate(NGON_NGU):
        v = np.array([cwe[ma][lang] for ma in top])
        ax.barh(y, v, left=left, color=MAU[i])
        left += v
    for i, val in enumerate(left):
        ax.text(val + max(left) * 0.015, i, so(val, 0), va="center", fontsize=12)
    ax.set_yticks(y, top)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.15)
    ax.set_xlabel("Số hàm lỗi có mã CWE")
    dinh_dang(ax, ngang=True)
    chu_giai_ngon_ngu(fig)
    luu(fig, "r2vul_2_3_top_cwe.png", rect=(0, 0, 1, 0.91))


def ve_top_du_an(repo):
    top = top_muoi(repo)
    tong = [sum(repo[r].values()) for r in top]
    chinh = [max(range(len(NGON_NGU)), key=lambda i: repo[r][NGON_NGU[i][0]]) for r in top]
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    ax.barh(np.arange(len(top)), tong, color=[MAU[i] for i in chinh])
    for i, val in enumerate(tong):
        ax.text(val + max(tong) * 0.015, i, so(val, 0), va="center", fontsize=12)
    ax.set_yticks(np.arange(len(top)), top)
    ax.invert_yaxis()
    ax.set_xlim(0, max(tong) * 1.18)
    ax.set_xlabel("Số hàm (màu theo ngôn ngữ chính của kho mã)")
    dinh_dang(ax, ngang=True)
    chu_giai_ngon_ngu(fig)
    luu(fig, "r2vul_2_4_top_du_an.png", rect=(0, 0, 1, 0.91))


def ve_thieu_do_thi(thieu):
    v = [[thieu[lang][k] for k in (0, 1)] for lang, _ in NGON_NGU]
    ymax = max(1, np.ceil(max(max(g) for g in v) * 1.25))
    cot_nhom(v, [ten for _, ten in NGON_NGU], "Tỉ lệ hàm thiếu đồ thị CPG (%)", "r2vul_2_5_thieu_do_thi.png", ymax, n=2)


def ve_nguon_goc(nguon):
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    x = np.arange(len(NGUON))
    bottom = np.zeros(len(NGUON))
    for i, (lang, _) in enumerate(NGON_NGU):
        v = np.array([nguon[origin][lang] for origin, _ in NGUON])
        ax.bar(x, v, 0.65, bottom=bottom, color=MAU[i])
        bottom += v
    ymax = np.ceil(max(bottom) * 1.2 / 1000) * 1000
    for i, val in enumerate(bottom):
        ax.text(i, val + ymax * 0.015, so(val, 0), ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x, [f"{origin}\n({'Lỗi' if target else 'An toàn'})" for origin, target in NGUON])
    ax.set_ylabel("Số hàm")
    ax.set_ylim(0, ymax)
    dinh_dang(ax)
    chu_giai_ngon_ngu(fig)
    luu(fig, "r2vul_2_6_nguon_goc.png", rect=(0, 0, 1, 0.91))


def main():
    dem, phan_vi, cwe, repo, nguon, chia = doc_du_lieu()
    thieu, co, tong = doc_do_thi(dem)
    OUT.mkdir(parents=True, exist_ok=True)
    ve_so_ham(dem)
    ve_do_dai(phan_vi)
    ve_top_cwe(cwe)
    ve_top_du_an(repo)
    ve_thieu_do_thi(thieu)
    ve_nguon_goc(nguon)
    print("Số hàm theo tập:", "; ".join(f"{k}={so(chia[k], 0)}" for k in ("train", "valid", "test")))
    print(f"Tổng: {so(tong, 0)} hàm; lỗi={so(sum(v[1] for v in dem.values()), 0)}; an toàn={so(sum(v[0] for v in dem.values()), 0)}")
    for lang, ten in NGON_NGU:
        print(f"{ten}: lỗi={so(dem[lang][1], 0)}; an toàn={so(dem[lang][0], 0)}; tỉ lệ lỗi={so(dem[lang][1] / sum(dem[lang].values()) * 100, 2)}%; thiếu CPG lỗi/an toàn={so(thieu[lang][1], 2)}%/{so(thieu[lang][0], 2)}%")
        print("  p50/p90/p99 lỗi:", "/".join(so(v, 2) for v in phan_vi[lang][1]), "; an toàn:", "/".join(so(v, 2) for v in phan_vi[lang][0]))
    print(f"CPG: có={so(co, 0)}; tổng={so(tong, 0)}; thiếu={so(tong - co, 0)} ({so((tong - co) / tong * 100, 2)}%)")
    for ma in BO_CWE:
        print(f"Bỏ khỏi top CWE: {ma}={so(sum(cwe[ma].values()), 0)} hàm lỗi")
    print("Top CWE:", "; ".join(f"{ma}={so(sum(cwe[ma].values()), 0)}" for ma in top_muoi(cwe, BO_CWE)))
    print("Top kho mã:", "; ".join(f"{r}={so(sum(repo[r].values()), 0)}" for r in top_muoi(repo)))
    print("Nguồn gốc:", "; ".join(f"{origin}={so(sum(nguon[origin].values()), 0)}" for origin, _ in NGUON))


if __name__ == "__main__":
    main()
