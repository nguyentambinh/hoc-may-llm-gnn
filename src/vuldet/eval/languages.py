# Nguyễn Tâm Bình - 2324801030003
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vuldet.eval.metrics import best_threshold, binary_metrics

LANGUAGES = ("c", "c_sharp", "java", "javascript", "python")
METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "accuracy_05", "precision_05", "recall_05", "f1_05")
KEYS = ("dataset", "huong", "cau_hinh", "mo_hinh", "tap", "seed", "cach", "nguong", "chon_nguong", "lang")
COLUMNS = (*KEYS, "n_valid", "n_test", "threshold", "nguon_nguong", *METRICS)
OUTPUT = Path("results/tables/r2vul_tung_ngon_ngu.csv")


def language_rows(valid_y, valid_scores, valid_langs, y, scores, langs, threshold=None, fixed_threshold=0.0):

    valid_y, valid_scores, valid_langs, y, scores, langs = (np.asarray(value) for value in (valid_y, valid_scores, valid_langs, y, scores, langs))
    if len(valid_y) != len(valid_scores) or len(valid_y) != len(valid_langs) or len(y) != len(scores) or len(y) != len(langs):
        raise ValueError("Số nhãn, điểm và ngôn ngữ không khớp")
    if (set(valid_langs) | set(langs)) - set(LANGUAGES):
        raise ValueError("Ngôn ngữ không thuộc R2Vul")
    if not len(valid_y) or not np.isfinite(valid_scores).all() or not np.isfinite(scores).all():
        raise ValueError("Cần valid và điểm dự đoán hữu hạn")
    if (set(valid_y) | set(y)) - {0, 1} or not np.isfinite(fixed_threshold) or (threshold is not None and not np.isfinite(threshold)):
        raise ValueError("Nhãn phải là 0/1 và ngưỡng phải hữu hạn")
    global_threshold = best_threshold(valid_y, valid_scores)[0] if threshold is None else float(threshold)
    rows = []
    for mode in ("valid_chung", "valid_tung_ngon_ngu"):
        members = []
        for lang in LANGUAGES:
            valid_mask, mask = valid_langs == lang, langs == lang
            selected, source = global_threshold, "valid_chung"
            if mode == "valid_tung_ngon_ngu" and len(np.unique(valid_y[valid_mask])) == 2:
                selected, source = best_threshold(valid_y[valid_mask], valid_scores[valid_mask])[0], "valid_tung_ngon_ngu"
            metrics = {key: float("nan") for key in METRICS}
            if mask.any():
                metrics.update({key: value for key, value in binary_metrics(y[mask], scores[mask], selected).items() if key != "threshold"})
                fixed = binary_metrics(y[mask], scores[mask], fixed_threshold)
                metrics.update({f"{key}_05": fixed[key] for key in ("accuracy", "precision", "recall", "f1")})
            members.append({"lang": lang, "chon_nguong": mode, "n_valid": int(valid_mask.sum()), "n_test": int(mask.sum()), "threshold": selected, "nguon_nguong": source, **metrics})
        rows.extend(members)
        rows.append({"lang": "macro_5", "chon_nguong": mode, "n_valid": len(valid_y), "n_test": len(y), "threshold": global_threshold if mode == "valid_chung" else "", "nguon_nguong": mode, **{key: float(np.mean([member[key] for member in members])) for key in METRICS}})
    return rows


def language_index(interim):
    result = {}
    for split in ("valid", "test"):
        result[split] = {}
        with (Path(interim) / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in result[split] or row["lang"] not in LANGUAGES:
                    raise ValueError(f"ID/ngôn ngữ không hợp lệ: {row['id']}")
                result[split][row["id"]] = (int(row["target"]), row["lang"])
    return result


def append_language_result(dataset, context, valid_ids, valid_y, valid_scores, ids, y, scores, threshold=None, output=OUTPUT, interim="data/interim/r2vul", fixed_threshold=0.0):

    index = language_index(interim)
    languages = {}
    for split, sample_ids, labels in (("valid", valid_ids, valid_y), ("test", ids, y)):
        if len(sample_ids) != len(set(sample_ids)) or len(sample_ids) != len(labels):
            raise ValueError(f"ID trùng hoặc lệch nhãn: {split}")
        languages[split] = []
        for record_id, label in zip(sample_ids, labels):
            if record_id not in index[split] or index[split][record_id][0] != int(label):
                raise ValueError(f"ID/nhãn không khớp interim: {split}:{record_id}")
            languages[split].append(index[split][record_id][1])
    base = {key: context.get(key, "") for key in KEYS if key not in ("dataset", "lang", "chon_nguong")}
    rows = [{**base, "dataset": dataset, **row} for row in language_rows(valid_y, valid_scores, languages["valid"], y, scores, languages["test"], threshold, fixed_threshold)]
    output = Path(output)
    old = []
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            old = list(csv.DictReader(stream))
    new_keys = {tuple(str(row.get(key, "")) for key in KEYS) for row in rows}
    old = [row for row in old if tuple(row.get(key, "") for key in KEYS) not in new_keys]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(old + rows)
    temporary.replace(output)
    return rows


def evaluate_predictions(dataset="r2vul", preds="results/preds", output=OUTPUT):

    count = 0
    for path in sorted(Path(preds).glob("r2vul_*.npz")):
        with np.load(path) as data:
            if not {"valid_ids", "valid_y", "valid_score", "ids", "y", "score"} <= set(data.files):
                continue
            append_language_result(dataset, {"huong": "giam_sat", "mo_hinh": path.stem, "tap": "toan_bo" if "_toan_bo_" in path.stem else "co_do_thi", "cach": "npz"}, data["valid_ids"], data["valid_y"], data["valid_score"], data["ids"], data["y"], data["score"], float(data["threshold"]) if "threshold" in data else None, output)
        count += 1
    print(f"{count} tệp dự đoán; {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("r2vul",), default="r2vul")
    parser.add_argument("--preds", default="results/preds")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    evaluate_predictions(args.dataset, args.preds, args.output)
