# Nguyễn Tâm Bình - 2324801030003
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from vuldet.eval.metrics import accuracy_metrics, best_threshold, binary_metrics, fixed_metrics
from vuldet.eval.languages import append_language_result
from vuldet.train.common import report

OUTPUT = Path("results/tables/cai_thien_f1.csv")
KEYS = ("dataset", "cach", "mo_hinh", "nguong")
METRICS = ("threshold", "accuracy", "precision", "recall", "f1", "auroc", "auprc", "accuracy_05", "precision_05", "recall_05", "f1_05", "threshold_acc", "accuracy_acc", "precision_acc", "recall_acc", "f1_acc")
COLUMNS = (*KEYS, "so_seed", "n_test", *METRICS, "f1_std", "recall_std", "precision_std", "valid_f1", "ghi_chu")
SEEDS = (42, 43, 44)


def logit(probability):
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return np.log(probability / (1 - probability))


def sigmoid(scores):
    return 1 / (1 + np.exp(-np.asarray(scores, dtype=np.float64)))


def load_members(dataset):

    members, reference = {}, None
    for path in sorted(Path("results/preds").glob(f"{dataset}_sup_*.npz")):
        match = re.fullmatch(rf"{dataset}_sup_(.+)_s(\d+)", path.stem)
        if match is None or "_toan_bo" in match.group(1):
            continue
        name = match.group(1).replace("_co_do_thi", "")
        data = np.load(path)
        if "valid_score" not in data:
            continue
        if reference is None:
            reference = {"valid_ids": data["valid_ids"], "valid_y": data["valid_y"], "test_ids": data["ids"], "test_y": data["y"]}
        valid = dict(zip(data["valid_ids"], data["valid_score"].astype(np.float64)))
        test = dict(zip(data["ids"], data["score"].astype(np.float64)))
        members.setdefault(name, {})[int(match.group(2))] = (np.array([valid[item] for item in reference["valid_ids"]]), np.array([test[item] for item in reference["test_ids"]]))
    return members, reference


def seed_average(member):

    return tuple(logit(np.mean([sigmoid(member[seed][part]) for seed in sorted(member)], axis=0)) for part in (0, 1))


def bootstrap_threshold(labels, scores, rounds=200, seed=0):

    generator = np.random.default_rng(seed)
    thresholds = []
    for _ in range(rounds):
        index = generator.integers(0, len(labels), len(labels))
        if labels[index].sum() == 0:
            continue
        thresholds.append(best_threshold(labels[index], scores[index])[0])
    return float(np.median(thresholds))


def evaluate(reference, valid_scores, test_scores, threshold, threshold_scores=None):

    valid_y, test_y = reference["valid_y"], reference["test_y"]
    source = valid_scores if threshold_scores is None else threshold_scores
    metrics = {**binary_metrics(test_y, test_scores, threshold), **fixed_metrics(test_y, test_scores), **accuracy_metrics(valid_y, source, test_y, test_scores)}
    metrics["valid_f1"] = binary_metrics(valid_y, source, threshold)["f1"]
    return metrics


def both_thresholds(reference, valid_scores, test_scores):
    valid_y = reference["valid_y"]
    return {"f1_valid": evaluate(reference, valid_scores, test_scores, best_threshold(valid_y, valid_scores)[0]), "bootstrap_trung_vi": evaluate(reference, valid_scores, test_scores, bootstrap_threshold(valid_y, valid_scores))}


def mean_rows(results):

    row = {name: float(np.mean([item[name] for item in results])) for name in (*METRICS, "valid_f1") if name in results[0] and results[0][name] is not None}
    row.update({f"{name}_std": float(np.std([item[name] for item in results])) for name in ("f1", "recall", "precision")})
    return row


def rank_transform(valid_scores, scores):

    return np.searchsorted(np.sort(valid_scores), scores, side="right") / len(valid_scores)


def stacking(reference, valid_matrix, test_matrix):

    valid_y = reference["valid_y"]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(valid_matrix, valid_y))
    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        out_of_fold = np.zeros(len(valid_y))
        for train, held in folds:
            out_of_fold[held] = LogisticRegression(C=c, max_iter=2000).fit(valid_matrix[train], valid_y[train]).decision_function(valid_matrix[held])
        auprc = average_precision_score(valid_y, out_of_fold)
        if best is None or auprc > best[0]:
            best = (auprc, c, out_of_fold)
    auprc, c, out_of_fold = best
    model = LogisticRegression(C=c, max_iter=2000).fit(valid_matrix, valid_y)
    return out_of_fold, model.decision_function(test_matrix), c, auprc, model.coef_[0]


def greedy(valid_y, valid_scores, combine):

    chosen, best = [], 0.0
    while True:
        gains = {name: roc_auc_score(valid_y, combine([valid_scores[item] for item in chosen + [name]])) for name in valid_scores if name not in chosen}
        if not gains:
            break
        name = max(gains, key=gains.get)
        if gains[name] <= best + 1e-4:
            break
        chosen.append(name)
        best = gains[name]
    return chosen, best


def write_rows(dataset, rows, replace_all):
    old = []
    if OUTPUT.exists():
        with open(OUTPUT, encoding="utf-8", newline="") as file:
            old = list(csv.DictReader(file))
    new_keys = {tuple(str(row[key]) for key in KEYS) for row in rows}
    kept = [row for row in old if not (row["dataset"] == dataset and (replace_all or tuple(row[key] for key in KEYS) in new_keys))]
    for row in rows:
        for name, value in row.items():
            if isinstance(value, (float, np.floating)):
                row[name] = f"{value:.4f}"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(kept + rows)
    temporary.replace(OUTPUT)


def line(dataset, method, metrics):
    text = f"XONG {dataset} cai_thien_f1 {method} Acc={metrics['accuracy']:.4f} Acc@0.5={metrics['accuracy_05']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}"
    return text


def run(args):
    dataset = args.dataset
    members, reference = load_members(dataset)
    valid_y = reference["valid_y"]

    base = {"dataset": dataset, "n_test": len(reference["test_y"])}
    rows, lines, averaged = [], [], {}
    def languages(valid, test, metrics, model, method, kind, seed):
        append_language_result(dataset, {"huong": "giam_sat", "cau_hinh": "Kết hợp", "mo_hinh": model, "tap": "co_do_thi", "seed": seed, "cach": method, "nguong": kind}, reference["valid_ids"], valid_y, valid, reference["test_ids"], reference["test_y"], test, metrics["threshold"])
    for name, member in members.items():
        seeds = sorted(member)
        averaged[name] = seed_average(member)

        per_seed = {"f1_valid": [], "bootstrap_trung_vi": []}
        for seed in seeds:
            for kind, metrics in both_thresholds(reference, *member[seed]).items():
                per_seed[kind].append(metrics)
                languages(*member[seed], metrics, name, "tung_seed", kind, seed)
        for kind, results in per_seed.items():
            rows.append({**base, **mean_rows(results), "cach": "tung_seed", "mo_hinh": name, "nguong": kind, "so_seed": len(seeds), "ghi_chu": "trung bình các seed, mỗi seed chọn ngưỡng riêng trên valid"})
        if len(seeds) > 1:
            for kind, metrics in both_thresholds(reference, *averaged[name]).items():
                rows.append({**base, **metrics, "cach": "tb_diem_3seed", "mo_hinh": name, "nguong": kind, "so_seed": len(seeds), "ghi_chu": "trung bình xác suất các seed rồi chọn ngưỡng trên valid"})
                languages(*averaged[name], metrics, name, "tb_diem_3seed", kind, "tb_diem")
    valid_scores = {name: scores[0] for name, scores in averaged.items()}
    test_scores = {name: scores[1] for name, scores in averaged.items()}

    combiners = {"kethop_tb_xacsuat": (lambda items: logit(np.mean([sigmoid(item) for item in items], axis=0)), lambda name, scores: scores), "kethop_tb_thuhang": (lambda items: logit(np.mean(items, axis=0)), lambda name, scores: rank_transform(valid_scores[name], scores))}
    for method, (combine, transform) in combiners.items():
        transformed = {name: transform(name, scores) for name, scores in valid_scores.items()}
        chosen, auroc = greedy(valid_y, transformed, combine)
        valid_combined = combine([transformed[name] for name in chosen])
        test_combined = combine([transform(name, test_scores[name]) for name in chosen])
        for kind, metrics in both_thresholds(reference, valid_combined, test_combined).items():
            languages(valid_combined, test_combined, metrics, "+".join(chosen), method, kind, "tb_diem")
            rows.append({**base, **metrics, "cach": method, "mo_hinh": "+".join(chosen), "nguong": kind, "so_seed": 3, "ghi_chu": f"thêm dần theo AUROC valid = {auroc:.4f}; thành viên là điểm trung bình 3 seed" + ("; Acc@0,5 ở đây là ngưỡng thứ hạng trung vị, không phải xác suất 0,5" if method == "kethop_tb_thuhang" else "")})
            if kind == "bootstrap_trung_vi":
                lines.append(line(dataset, f"{method}({'+'.join(chosen)}) ngưỡng bootstrap", metrics))

    names = sorted(valid_scores)
    features = {"stacking_logit": (np.column_stack([valid_scores[name] for name in names]), np.column_stack([test_scores[name] for name in names])), "stacking_thuhang": (np.column_stack([rank_transform(valid_scores[name], valid_scores[name]) for name in names]), np.column_stack([rank_transform(valid_scores[name], test_scores[name]) for name in names]))}

    for method, (valid_matrix, test_matrix) in features.items():
        out_of_fold, test_stacked, c, auprc, weights = stacking(reference, valid_matrix, test_matrix)
        note = f"LR C={c} AUPRC chéo valid={auprc:.4f}; hệ số " + " ".join(f"{name}={weight:.2f}" for name, weight in zip(names, weights))
        thresholds = {"cheo_5fold": best_threshold(valid_y, out_of_fold)[0], "cheo_5fold_bootstrap": bootstrap_threshold(valid_y, out_of_fold)}
        for kind, threshold in thresholds.items():
            metrics = evaluate(reference, out_of_fold, test_stacked, threshold, threshold_scores=out_of_fold)
            languages(out_of_fold, test_stacked, metrics, "tất cả", method, kind, "cheo_5fold")
            rows.append({**base, **metrics, "cach": method, "mo_hinh": "tất cả", "nguong": kind, "so_seed": 3, "ghi_chu": note})
            if kind == "cheo_5fold":
                lines.append(line(dataset, f"{method} ngưỡng chéo 5 fold", metrics))
    best_single = max((row for row in rows if row["cach"] == "tb_diem_3seed" and row["nguong"] == "bootstrap_trung_vi"), key=lambda row: row["valid_f1"])
    lines.insert(0, line(dataset, f"tb_diem_3seed {best_single['mo_hinh']} ngưỡng bootstrap", best_single))
    write_rows(dataset, rows, True)
    for text in lines:
        print(text)
        if not args.no_report:
            report(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))

    parser.add_argument("--no-report", action="store_true")
    run(parser.parse_args())
