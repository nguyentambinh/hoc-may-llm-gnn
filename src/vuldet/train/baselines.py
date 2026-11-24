# Nguyễn Tâm Bình - 2324801030003
import argparse

import numpy as np

from vuldet.eval.metrics import binary_metrics, fixed_metrics
from vuldet.train.common import append_result, graph_id_sets, load_interim




def run(dataset, seed):
    return run_r2vul(seed)


def run_r2vul(seed):
    graph_ids = graph_id_sets("r2vul")
    data = {split: load_interim("r2vul", split)[:2] for split in ("train", "valid", "test")}
    for subset in ("co_do_thi", "toan_bo"):
        parts = {}
        for split, (ids, labels) in data.items():
            keep = np.array([subset == "toan_bo" or item in graph_ids[split] for item in ids])
            parts[split] = (np.asarray(ids)[keep], labels[keep])
        rate = parts["train"][1].mean()
        generator = np.random.default_rng(seed)
        random_scores = {split: rate - generator.random(len(parts[split][0])) for split in ("valid", "test")}
        for model, scores in (("tat_ca_loi", {split: np.ones(len(parts[split][0])) for split in ("valid", "test")}), ("tat_ca_an_toan", {split: -np.ones(len(parts[split][0])) for split in ("valid", "test")}), ("ngau_nhien", random_scores)):
            valid_ids, valid_y = parts["valid"]
            ids, y = parts["test"]
            append_result({"dataset": "r2vul", "huong": "moc", "cau_hinh": "Baseline", "mo_hinh": f"Baseline-{model}", "tap": subset, "seed": seed, "n_train": len(parts["train"][1]), "n_test": len(y), **binary_metrics(y, scores["test"], 0.0), **fixed_metrics(y, scores["test"]), "ghi_chu": f"ngưỡng baseline cố định; ti_le_loi_train={rate:.4f}"}, (valid_ids, valid_y, scores["valid"], ids, y, scores["test"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.dataset, args.seed)
