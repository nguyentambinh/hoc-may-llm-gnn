# Nguyễn Tâm Bình - 2324801030003
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

SPLITS = ("train", "valid", "test")
RESULT_FILE = Path("results/tables/ket_qua.csv")
RESULT_KEYS = ("dataset", "huong", "cau_hinh", "mo_hinh", "tap", "seed")
RESULT_COLUMNS = (*RESULT_KEYS, "n_train", "n_test", "threshold", "accuracy", "precision", "recall", "f1", "auroc", "auprc", "accuracy_05", "precision_05", "recall_05", "f1_05", "threshold_acc", "accuracy_acc", "precision_acc", "recall_acc", "f1_acc", "ghi_chu", "thoi_gian")


def setup(seed):
    threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return threads


def load_interim(dataset, split):
    ids, labels, metas = [], [], []
    with open(f"data/interim/{dataset}/{split}.jsonl", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            ids.append(row["id"])
            labels.append(int(row["target"]))
            metas.append(row.get("meta") or {})
    return ids, np.array(labels), metas


def func_dir(dataset, model):
    return Path(f"data/embeddings/{dataset}") / ("" if model == "unixcoder" else model)


def load_func(dataset, split, model):
    folder = func_dir(dataset, model)
    matrix = np.load(folder / f"func_{split}.npy").astype(np.float32)
    ids = (folder / f"func_{split}_ids.txt").read_text(encoding="utf-8").split()
    if len(ids) != len(matrix):
        raise ValueError(f"{dataset} {split} {model}: {len(ids)} id, {len(matrix)} dòng")
    return matrix, ids


def load_graphs(dataset, split, limit=0):
    graphs = []
    for path in sorted(Path(f"data/graphs/{dataset}/pyg").glob(f"{split}_*.pt")):
        graphs.extend(torch.load(path, weights_only=False))
        if limit and len(graphs) >= limit:
            return graphs[:limit]
    return graphs


def graph_id_sets(dataset):
    cache = Path(f"results/cache/graph_ids_{dataset}.json")
    if cache.exists():
        return {split: set(ids) for split, ids in json.loads(cache.read_text()).items()}
    sets = {split: [graph.id for graph in load_graphs(dataset, split)] for split in SPLITS}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sets))
    return {split: set(ids) for split, ids in sets.items()}


def node_features(code_id, node_matrix):
    unique, inverse = torch.unique(code_id, return_inverse=True)
    return torch.from_numpy(node_matrix[unique.numpy()].astype(np.float32))[inverse]


def append_result(row, predictions=None):
    if predictions is not None:
        from vuldet.eval.languages import append_language_result
        append_language_result("r2vul", row, *predictions, threshold=row.get("threshold"))
    row = {**row, "thoi_gian": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 7 * 3600))}
    for name, value in row.items():
        if isinstance(value, (float, np.floating)):
            row[name] = f"{value:.4f}"
    rows = []
    if RESULT_FILE.exists():
        with open(RESULT_FILE, encoding="utf-8", newline="") as file:
            rows = [old for old in csv.DictReader(file) if any(str(old[key]) != str(row[key]) for key in RESULT_KEYS)]
    rows.append(row)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_FILE.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULT_FILE)


def report(line):
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    with open("results/logs/chuoi_chay.txt", "a", encoding="utf-8") as file:
        file.write(line + "\n")
