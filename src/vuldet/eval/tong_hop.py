# Nguyễn Tâm Bình - 2324801030003
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from vuldet.train.common import RESULT_FILE

GROUP_KEYS = ("dataset", "huong", "cau_hinh", "mo_hinh", "tap")
METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "accuracy_05", "f1_05", "accuracy_acc", "f1_acc")


def summarize(output="results/tables/tong_hop.csv"):
    groups = defaultdict(list)
    with open(RESULT_FILE, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            groups[tuple(row[key] for key in GROUP_KEYS)].append(row)
    rows = []
    for key, members in groups.items():
        row = dict(zip(GROUP_KEYS, key), so_seed=len(members), seeds=" ".join(sorted(member["seed"] for member in members)))
        for metric in METRICS:
            values = [float(member[metric]) for member in members if member.get(metric) not in (None, "", "nan")]
            if values:
                row[metric] = f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}" if len(values) > 1 else f"{values[0]:.4f}"
        rows.append(row)

    missing = [row for row in rows if "accuracy" not in row or (row["huong"] == "giam_sat" and ("accuracy_05" not in row or "accuracy_acc" not in row))]
    if missing:
        raise ValueError(f"Thiếu accuracy: {[(row['dataset'], row['mo_hinh'], row['tap']) for row in missing]}")
    with open(Path(output), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(*GROUP_KEYS, "so_seed", "seeds", *METRICS), restval="")
        writer.writeheader()
        writer.writerows(rows)

    columns = ("huong", "cau_hinh", "mo_hinh", "tap", "so_seed", "accuracy", "f1", "auroc", "auprc", "precision", "recall", "accuracy_05", "f1_05", "accuracy_acc", "f1_acc")
    for dataset in sorted({row["dataset"] for row in rows}):
        with open(Path(output).with_name(f"tong_hop_{dataset}.csv"), "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: (row["tap"], row["huong"], row["cau_hinh"], row["mo_hinh"])))
    return len(rows)


if __name__ == "__main__":
    print(summarize(), "dòng")
