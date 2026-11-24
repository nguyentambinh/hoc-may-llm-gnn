# Nguyễn Tâm Bình - 2324801030003
import csv
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib
import sklearn
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from vuldet.eval.cai_thien_f1 import rank_transform, seed_average, stacking
from vuldet.eval.metrics import best_threshold, binary_metrics

RESULTS = ROOT / "results"
OUT = ROOT / "BaoCao" / "hinh"
NGON_NGU = [("c", "C"), ("c_sharp", "C#"), ("java", "Java"), ("javascript", "JavaScript"), ("python", "Python"), ("macro_5", "Trung bình\nmacro")]
MAU = ["#33419B", "#43ADE2", "#E17C3C", "#549F76", "#8F6BB0", "#945D46"]
LUONG = [("gnn_none", "L8 · GNN"), ("llm_gnn_unixcoder", "L9 · LLM+GNN (UniXcoder)"), ("llm_gnn_qwen15", "L10 · LLM+GNN (Qwen)"), ("llm_MLP-unixcoder+qwen15", "L6 · MLP (UniXcoder + Qwen)"), ("llm_LoRA-qwen15", "L7 · LoRA"), ("stacking_thuhang", "L13 · Stacking thứ hạng")]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13, "savefig.dpi": 300})
so = lambda v, n=1: f"{v:,.{n}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


class HaiNoi:
    def __init__(self, *files):
        self.files = files

    def write(self, value):
        for file in self.files:
            file.write(value)
        return len(value)

    def flush(self):
        for file in self.files:
            file.flush()


def doc_csv(name):
    with (RESULTS / "tables" / name).open(encoding="utf-8-sig", newline="") as file:
        return [r for r in csv.DictReader(file) if r["dataset"] == "r2vul"]


def mot(rows, **criteria):
    found = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    if len(found) != 1:
        raise ValueError(f"Cần đúng một dòng: {criteria}; tìm thấy {len(found)}")
    return found[0]


def doc_nhan():
    labels = {}
    for split in ("valid", "test"):
        labels[split] = {}
        with (ROOT / "data" / "interim" / "r2vul" / f"{split}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                if row["id"] in labels[split] or row["lang"] not in dict(NGON_NGU[:5]):
                    raise ValueError(f"ID trùng hoặc ngôn ngữ lạ trong {split}")
                labels[split][row["id"]] = (row["lang"], int(row["target"]))
    if len(labels["test"]) != 1838:
        raise ValueError("Tập test toàn bộ phải có 1.838 hàm")
    return labels


def tinh_stacking(rows, labels):
    members, reference = {}, None
    expected = {"gnn_none", "llm_LR-qwen15", "llm_LR-unixcoder", "llm_LR-unixcoder+qwen15", "llm_MLP-qwen15", "llm_MLP-unixcoder", "llm_MLP-unixcoder+qwen15", "llm_LoRA-qwen15", "llm_gnn_qwen15", "llm_gnn_unixcoder"}
    for path in sorted((RESULTS / "preds").glob("r2vul_sup_*.npz")):
        match = re.fullmatch(r"r2vul_sup_(.+)_s(\d+)", path.stem)
        if match is None or "_toan_bo" in match[1]:
            continue
        name = match[1].replace("_co_do_thi", "")
        if name not in expected:
            continue
        with np.load(path, allow_pickle=False) as data:
            if reference is None:
                reference = {k: data[k].copy() for k in ("ids", "y", "valid_ids", "valid_y")}
            scores = []
            for split, ids_key, y_key, score_key in (("valid", "valid_ids", "valid_y", "valid_score"), ("test", "ids", "y", "score")):
                if len(set(data[ids_key])) != len(data[ids_key]) or set(data[ids_key]) != set(reference[ids_key]):
                    raise ValueError(f"Tập ID không khớp: {path.name}, {split}")
                lookup = dict(zip(data[ids_key], data[score_key].astype(np.float64)))
                truth = dict(zip(data[ids_key], data[y_key]))
                for item, y in zip(reference[ids_key], reference[y_key]):
                    if truth[item] != y or labels[split][item][1] != y:
                        raise ValueError(f"Nhãn không khớp: {path.name}, {item}")
                ordered = np.array([lookup[item] for item in reference[ids_key]])
                if not np.isfinite(ordered).all():
                    raise ValueError(f"Điểm không hữu hạn: {path.name}")
                scores.append(ordered)
            members.setdefault(name, {})[int(match[2])] = tuple(scores)
    if set(members) != expected or len(reference["ids"]) != 1784:
        raise ValueError("Thiếu thành viên L1–L10 hoặc sai số hàm có đồ thị")
    averaged = {}
    for name, seeds in members.items():
        expected_seeds = {42} if "_LR-" in name or "_LoRA-" in name else {42, 43, 44}
        if set(seeds) != expected_seeds:
            raise ValueError(f"Thiếu seed cho {name}: {sorted(seeds)}")
        averaged[name] = seed_average(seeds)
    names = sorted(averaged)
    valid = np.column_stack([rank_transform(averaged[n][0], averaged[n][0]) for n in names])
    test = np.column_stack([rank_transform(averaged[n][0], averaged[n][1]) for n in names])
    oof, score, c, auprc, weights = stacking(reference, valid, test)
    candidates = [r for r in rows if r["cach"] == "stacking_thuhang"]
    selected = max(candidates, key=lambda r: float(r["valid_f1"]))
    if selected["nguong"] != "cheo_5fold":
        raise ValueError("Ngưỡng L13 có F1 valid cao nhất không phải cheo_5fold")
    threshold, valid_f1 = best_threshold(reference["valid_y"], oof)
    metrics = binary_metrics(reference["y"], score, threshold)
    print("KIỂM STACKING L13: mọi mô hình có giám sát L1–L10; trung bình xác suất seed rồi đổi về logit.")
    print(f"Valid n={len(oof)}; test n={len(score)}; C={so(c, 2)}; AUPRC chéo valid={so(auprc, 8)}; F1 chéo valid={so(valid_f1, 8)}; ngưỡng logit={so(threshold, 12)}")
    for n, weight in zip(names, weights):
        print(f"  {n}: seed={','.join(map(str, sorted(members[n])))}; hệ số={so(weight, 8)}")
    matched = all(f"{metrics[k]:.4f}" == selected[k] for k in ("f1", "auroc"))
    matched = matched and f"{metrics['f1']:.4f}" == "0.7835" and f"{metrics['auroc']:.4f}" == "0.8449"
    print(f"Đối chiếu CSV: {'KHỚP' if matched else 'KHÔNG KHỚP'}; F1 tính lại={so(metrics['f1'], 10)} / CSV={so(float(selected['f1']), 4)}; AUROC tính lại={so(metrics['auroc'], 10)} / CSV={so(float(selected['auroc']), 4)}")
    print("L13 test:", "; ".join(f"{k}={so(metrics[k], 8)}" for k in ("accuracy", "precision", "recall", "f1", "auroc", "auprc")))
    if not matched:
        raise ValueError("Stacking không khớp CSV; dừng, giữ nguyên số tính lại")
    return reference, members, score, threshold, selected


def dinh_dang(ax, n=0, ca_hai=False):
    ax.grid(axis="both" if ca_hai else "y", color="#EEEEEE")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: so(v, n)))
    if ca_hai:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: so(v, n)))


def luu(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                raise ValueError(f"Chữ vượt khung ảnh {name}: {text.get_text()}")
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)
    print(f"Ảnh: BaoCao/hinh/{name} (300 dpi)")


def f1_ngon_ngu(rows, name, kind="f1_valid", full=False):
    values = []
    for lang, _ in NGON_NGU:
        selected = [r for r in rows if r["tap"] == ("toan_bo" if full else "co_do_thi") and r["chon_nguong"] == "valid_chung" and r["lang"] == lang]
        if full:
            selected = [r for r in selected if r["mo_hinh"] == "LoRA-qwen15" and r["seed"] == "42" and r["cach"] == "" and r["nguong"] == ""]
        elif name == "stacking_thuhang":
            selected = [r for r in selected if r["cach"] == name and r["nguong"] == kind]
        else:
            selected = [r for r in selected if r["mo_hinh"] == name and r["cach"] == "tung_seed" and r["nguong"] == kind]
            expected = {"42"} if name == "llm_LoRA-qwen15" else {"42", "43", "44"}
            if {r["seed"] for r in selected} != expected or len(selected) != len(expected):
                raise ValueError(f"Sai seed: {name}, {lang}")
        if not selected or ((full or name == "stacking_thuhang") and len(selected) != 1):
            raise ValueError(f"Không xác định được F1: {name}, {lang}")
        values.append(np.mean([float(r["f1_05" if full else "f1"]) for r in selected]) * 100)
    return values


def cot_nhom(values, labels, colors, filename, note, highlight=False):
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    x = np.arange(6)
    width = 0.82 / len(values)
    ax.axvspan(4.5, 5.5, color="#F3F3F3", zorder=0)
    for i, (v, label, color) in enumerate(zip(values, labels, colors)):
        bars = ax.bar(x + (i - (len(values) - 1) / 2) * width, v, width * 0.92, color=color, alpha=0.4 if highlight and i < 5 else 1, label=label, edgecolor=color, linewidth=0.7)
        if highlight and i >= 5:
            for bar in bars:
                bar.set_linewidth(1.2)
        bar = bars[-1]
        ax.text(bar.get_x() + bar.get_width() / 2, v[-1] + 1.2, so(v[-1], 1), rotation=90, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [ten for _, ten in NGON_NGU])
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.55, 5.55)
    dinh_dang(ax)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=11.5)
    fig.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=11)
    luu(fig, filename, rect=(0, 0.085, 1, 0.83 if highlight else 0.86))


def ve_f1(rows, kind):
    values = [f1_ngon_ngu(rows, n, kind if n == "stacking_thuhang" else "f1_valid") for n, _ in LUONG]
    print("\nHÌNH 4.1 — F1 (%), có đồ thị, ngưỡng valid chung; trung bình F1 từng seed:")
    for (_, label), v in zip(LUONG, values):
        print(label, "; ".join(f"{ten.replace(chr(10), ' ')}={so(a, 4)}" for (_, ten), a in zip(NGON_NGU, v)))
    cot_nhom(values, [label for _, label in LUONG], MAU, "r2vul_4_1_f1_ngon_ngu.png", "Test có đồ thị: 1.784 hàm · Ngưỡng chọn trên valid chung\nL8, L9, L10, L6: trung bình F1 của 3 seed; L7: seed 42; L13: dự đoán chéo 5 phần")
    return values[-1]


def ve_duong_cong(reference, members, score, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.3))
    y = reference["y"]
    print("\nHÌNH 4.2 — ROC và Precision–Recall, cùng 1.784 hàm test:")
    for i, (name, label) in enumerate([LUONG[k] for k in (0, 3, 4, 2, 5)]):
        s = score if name == "stacking_thuhang" else members[name][42][1]
        th = threshold if name == "stacking_thuhang" else best_threshold(reference["valid_y"], members[name][42][0])[0]
        metrics = binary_metrics(y, s, th)
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        short = label.split(" · ")[0] + (" (chéo 5 phần)" if name == "stacking_thuhang" else " (seed 42)")
        axes[0].plot(fpr, tpr, color=MAU[i], lw=2, label=f"{short}: {so(metrics['auroc'], 4)}")
        axes[1].step(recall, precision, where="post", color=MAU[i], lw=2, label=f"{short}: {so(metrics['auprc'], 4)}")
        print(short, "; ".join(f"{k}={so(metrics[k], 8)}" for k in ("accuracy", "precision", "recall", "f1", "auroc", "auprc")))
    axes[0].plot([0, 1], [0, 1], color="#AAAAAA", ls="--", lw=1)
    axes[1].axhline(y.mean(), color="#AAAAAA", ls="--", lw=1)
    for ax, title, xlabel, ylabel, metric in zip(axes, ("(a) Đường cong ROC", "(b) Đường cong Precision–Recall"), ("Tỉ lệ dương tính giả (FPR)", "Độ bao phủ (Recall)"), ("Tỉ lệ dương tính thật (TPR)", "Độ chuẩn xác (Precision)"), ("AUROC", "AUPRC (Average Precision)")):
        ax.set(xlim=(0, 1), ylim=(0, 1.025), title=title, xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        dinh_dang(ax, n=1, ca_hai=True)
        ax.legend(title=metric, frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10.5, title_fontsize=11, loc="lower right" if ax is axes[0] else "lower left")
    print(f"Đường cơ sở PR: tỉ lệ hàm lỗi={so(y.mean(), 8)}")
    luu(fig, "r2vul_4_2_roc_pr.png")


def ve_nham_lan(reference, score, threshold, labels, lang_rows, kind):
    fig = plt.figure(figsize=(12.0, 6.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.65, 1, 1, 1])
    cmap = LinearSegmentedColormap.from_list("r2vul", ["#FFFFFF", MAU[0]])
    pred = score >= threshold
    langs = np.array([labels["test"][item][0] for item in reference["ids"]])
    print(f"\nHÌNH 4.3 — L13, ngưỡng logit={so(threshold, 12)}; tỉ lệ chuẩn hóa theo hàng nhãn thật:")
    for i, (lang, title) in enumerate([("all", "Tổng: 1.784 hàm")] + NGON_NGU[:5]):
        mask = np.ones(len(pred), dtype=bool) if i == 0 else langs == lang
        cm = confusion_matrix(reference["y"][mask], pred[mask], labels=[0, 1])
        ratio = cm / cm.sum(axis=1, keepdims=True)
        tn, fp, fn, tp = cm.ravel()
        print(f"{title}: n={mask.sum()}; TP={tp}; FP={fp}; TN={tn}; FN={fn}; tỉ lệ hàng TN/FP/FN/TP (%)=" + "/".join(so(v * 100, 6) for v in ratio.ravel()))
        if i:
            row = mot(lang_rows, tap="co_do_thi", chon_nguong="valid_chung", lang=lang, cach="stacking_thuhang", nguong=kind)
            if not np.isclose(binary_metrics(reference["y"][mask], score[mask], threshold)["f1"], float(row["f1"]), atol=1e-12):
                raise ValueError(f"F1 L13 theo ngôn ngữ không khớp: {lang}")
        ax = fig.add_subplot(gs[:, 0] if i == 0 else gs[(i - 1) // 3, 1 + (i - 1) % 3])
        ax.imshow(ratio, cmap=cmap, vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{so(cm[r, c], 0)}\n({so(ratio[r, c] * 100, 1)}%)", ha="center", va="center", fontsize=15 if i == 0 else 12, color="white" if ratio[r, c] > 0.55 else "#222222")
        ax.set_xticks([0, 1], ["An toàn", "Lỗi"])
        ax.set_yticks([0, 1], ["An toàn", "Lỗi"])
        ax.set_xlabel("Nhãn dự đoán", fontsize=12)
        ax.set_ylabel("Nhãn thật", fontsize=12)
        ax.set_title(title if i == 0 else f"{title} (n={mask.sum()})", fontsize=13, pad=10)
        ax.tick_params(labelsize=12)
        for spine in ax.spines.values():
            spine.set_color("#DDDDDD")
    note = fig.add_subplot(gs[1, 3])
    note.axis("off")
    note.text(0, 0.52, "L13 · Stacking thứ hạng\nNgưỡng từ valid chéo 5 phần\n\nMỗi ô: số hàm\nvà tỉ lệ theo hàng nhãn thật.\n\nMàu đậm: tỉ lệ cao hơn.", va="center", fontsize=12, linespacing=1.5)
    luu(fig, "r2vul_4_3_nham_lan.png")


def ve_do_dai(rows):
    names = ["GNN (RGCN)", "MLP-qwen15", "LLM+GNN (RGCN+node-unixcoder+func-qwen15)", "LoRA-qwen15", "LoRA + LLM+GNN (w)"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x, width = np.arange(2), 0.155
    groups = ["token <= 512", "token > 512 (bị cắt)"]
    stats = {}
    print("\nHÌNH 4.4 — AUROC theo độ dài; trung bình ± độ lệch chuẩn 3 seed, riêng LoRA seed 42:")
    for i, name in enumerate(names):
        values, errors = [], []
        for group in groups:
            candidates = [r for r in rows if r["phan"] == "do_dai" and r["mo_hinh"] == name and r["nhom"] == group]
            seed = "42" if name == "LoRA-qwen15" else ("tb" if any(r["seed"] == "tb" for r in candidates) else "42")
            row = mot(candidates, seed=seed)
            parts = [float(v.strip()) for v in row["auroc"].split("±")]
            values.append(parts[0])
            errors.append(parts[1] if len(parts) > 1 else 0)
            stat = (int(row["n"]), float(row["ti_le_loi"]))
            if group in stats and stats[group] != stat:
                raise ValueError("Số hàm hoặc tỉ lệ lỗi giữa các mô hình không khớp")
            stats[group] = stat
            auroc_text = so(values[-1], 4) + (f" ± {so(errors[-1], 4)}" if len(parts) > 1 else "")
            print(f"{name}; {group}; seed={seed}; AUROC={auroc_text}; n={stat[0]}; lỗi={so(stat[1] * 100, 2)}%; accuracy={row['accuracy'].replace('.', ',')}")
        label = ["GNN (RGCN)", "MLP-qwen15", "LLM+GNN", "LoRA", "LoRA + LLM+GNN"][i]
        bars = ax.bar(x + (i - 2) * width, values, width * 0.92, yerr=None if name == "LoRA-qwen15" else errors, capsize=3, color=MAU[i], label=label, error_kw={"lw": 1})
        for bar, v, error in zip(bars, values, errors):
            ax.text(bar.get_x() + bar.get_width() / 2, v + error + 0.012, so(v, 4), ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, [f"{title}\nn = {so(stats[g][0], 0)}; hàm lỗi: {so(stats[g][1] * 100, 2)}%" for g, title in zip(groups, ("≤ 512 token", "> 512 token"))])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    dinh_dang(ax, n=1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=12)
    fig.text(0.5, 0.02, "Cột: AUROC trung bình 3 seed; thanh sai số: độ lệch chuẩn · LoRA: seed 42", ha="center", fontsize=11)
    luu(fig, "r2vul_4_4_do_dai.png", rect=(0, 0.06, 1, 0.87))


def ve_so_bai_bao(rows, stack_values, ket_qua):
    paper = {'nguon': 'Weyssow et al., R2Vul, arXiv 2504.04699v2, Table 1 + Table 7 (Appendix A.3); P/R/F1 (%), SFT/R2Vul trung bình 3 seed', 'c_sharp': {'CodeBERT CLS': [70.37, 82.61, 76.0], 'Qwen2.5-1.5B CLS': [73.91, 73.91, 73.91], 'Qwen2.5-1.5B SFT': [81.83, 91.3, 86.28], 'R2Vul 1.5B': [90.91, 86.96, 88.89]}, 'javascript': {'CodeBERT CLS': [71.43, 71.43, 71.43], 'Qwen2.5-1.5B CLS': [66.67, 57.14, 61.54], 'Qwen2.5-1.5B SFT': [69.58, 68.25, 68.91], 'R2Vul 1.5B': [69.29, 78.1, 73.42]}, 'java': {'CodeBERT CLS': [79.71, 71.9, 75.6], 'Qwen2.5-1.5B CLS': [78.2, 79.74, 78.96], 'Qwen2.5-1.5B SFT': [74.46, 89.54, 81.3], 'R2Vul 1.5B': [80.52, 92.56, 86.13]}, 'python': {'CodeBERT CLS': [73.17, 57.69, 64.52], 'Qwen2.5-1.5B CLS': [67.21, 78.85, 72.57], 'Qwen2.5-1.5B SFT': [67.15, 79.49, 72.8], 'R2Vul 1.5B': [75.38, 84.4, 79.64]}, 'c': {'CodeBERT CLS': [74.94, 65.67, 70.0], 'Qwen2.5-1.5B CLS': [73.57, 74.2, 73.89], 'Qwen2.5-1.5B SFT': [65.64, 76.69, 70.73], 'R2Vul 1.5B': [70.17, 85.43, 77.05]}, 'macro': {'CodeBERT CLS': [73.92, 69.86, 71.51], 'Qwen2.5-1.5B CLS': [71.92, 72.77, 72.17], 'Qwen2.5-1.5B SFT': [71.73, 81.06, 76.0], 'R2Vul 1.5B': [77.25, 85.5, 81.03]}, 'llm_f1': {'Claude-Opus-4': {'c_sharp': 70.59, 'javascript': 65.5, 'java': 72.39, 'python': 71.07, 'c': 66.4, 'macro': 69.19}}}
    names = ["CodeBERT CLS", "Qwen2.5-1.5B CLS", "Qwen2.5-1.5B SFT", "R2Vul 1.5B", "Claude-Opus-4"]
    values = [[paper["llm_f1"][name]["macro" if lang == "macro_5" else lang] if name == "Claude-Opus-4" else paper["macro" if lang == "macro_5" else lang][name][2] for lang, _ in NGON_NGU] for name in names]
    lora = f1_ngon_ngu(rows, "llm_LoRA-qwen15", full=True)
    lora_npz = RESULTS / "preds" / "r2vul_sup_llm_LoRA-qwen15_toan_bo_s42.npz"
    with np.load(lora_npz, allow_pickle=False) as data:
        metrics = binary_metrics(data["y"], data["score"], 0.0)
        if len(data["ids"]) != 1838:
            raise ValueError("LoRA toàn bộ test phải có 1.838 hàm")
    row = mot(ket_qua, mo_hinh="LoRA-qwen15", tap="toan_bo", seed="42")
    if f"{metrics['f1']:.4f}" != row["f1_05"]:
        raise ValueError("F1 LoRA tại xác suất 0,5 không khớp CSV")
    names += ["Nhóm · L7 LoRA (toàn bộ, p ≥ 0,5)", "Nhóm · L13 (có đồ thị, ngưỡng valid)"]
    values += [lora, stack_values]
    print("\nHÌNH 4.5 — So với số bài R2Vul; F1 (%):")
    print("Nguồn:", paper["nguon"])
    print("Bài báo và nhóm khác phạm vi test/ngưỡng; không suy diễn so sánh trực tiếp cùng điều kiện.")
    print("L7 toàn bộ test:", "; ".join(f"{k}={so(metrics[k], 8)}" for k in ("accuracy", "precision", "recall", "f1", "auroc", "auprc")))
    for name, v in zip(names, values):
        print(name, "; ".join(f"{ten.replace(chr(10), ' ')}={so(a, 4)}" for (_, ten), a in zip(NGON_NGU, v)))
    cot_nhom(values, names, [*MAU[:5], MAU[2], MAU[0]], "r2vul_4_5_so_bai_bao.png", "Số bài R2Vul: bảng công bố · Nhóm L7: toàn bộ 1.838 hàm, xác suất 0,5\nNhóm L13: 1.784 hàm có đồ thị, ngưỡng valid · Phạm vi test/ngưỡng khác nhau", highlight=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = RESULTS / "so_lieu_hinh_chuong4.txt"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with log.open("w", encoding="utf-8") as file, redirect_stdout(HaiNoi(sys.stdout, file)):
        print("SỐ LIỆU HÌNH CHƯƠNG 4 — R2Vul")
        print(f"Python {sys.version.split()[0]}; sklearn {sklearn.__version__}; matplotlib {matplotlib.__version__}")
        labels = doc_nhan()
        ket_qua, cai_thien, languages, analysis = [doc_csv(n) for n in ("ket_qua.csv", "cai_thien_f1.csv", "r2vul_tung_ngon_ngu.csv", "phan_tich_gnn.csv")]
        reference, members, score, threshold, selected = tinh_stacking(cai_thien, labels)
        stack_values = ve_f1(languages, selected["nguong"])
        ve_duong_cong(reference, members, score, threshold)
        ve_nham_lan(reference, score, threshold, labels, languages, selected["nguong"])
        ve_do_dai(analysis)
        ve_so_bai_bao(languages, stack_values, ket_qua)
        print("\nHoàn tất 5 ảnh; kiểm tra tự động: chữ nằm trong khung ảnh.")


if __name__ == "__main__":
    main()
