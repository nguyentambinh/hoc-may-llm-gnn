# Nguyễn Tâm Bình - 2324801030003
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vuldet.eval.metrics import accuracy_metrics, best_threshold, binary_metrics, fixed_metrics
from vuldet.eval.languages import append_language_result

COLUMNS = ("dataset", "phan", "nhom", "mo_hinh", "seed", "n", "ti_le_loi", "w", "valid_auroc", "accuracy", "accuracy_05", "accuracy_acc", "f1", "precision", "recall", "auroc", "auprc", "delta_auroc", "ci_thap", "ci_cao", "ti_le_duong", "ghi_chu")
METRICS = ("valid_auroc", "accuracy", "accuracy_05", "accuracy_acc", "f1", "precision", "recall", "auroc", "auprc", "w", "delta_auroc", "ci_thap", "ci_cao", "ti_le_duong")
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)


def logit(probability):
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return np.log(probability / (1 - probability))


class Scores:

    def __init__(self, path):
        data = np.load(path)
        self.valid = dict(zip(data["valid_ids"], 1 / (1 + np.exp(-data["valid_score"].astype(np.float64)))))
        self.test = dict(zip(data["ids"], 1 / (1 + np.exp(-data["score"].astype(np.float64)))))
        self.valid_ids, self.valid_y, self.test_ids, self.test_y = data["valid_ids"], data["valid_y"], data["ids"], data["y"]
        self.threshold = float(data["threshold"])

    def get(self, split, ids):
        table = self.valid if split == "valid" else self.test
        return np.array([table[item] for item in ids])


def evaluate(valid_y, valid_p, test_y, test_p):

    valid_s, test_s = logit(valid_p), logit(test_p)
    threshold, _ = best_threshold(valid_y, valid_s)
    metrics = {**binary_metrics(test_y, test_s, threshold), **fixed_metrics(test_y, test_s), **accuracy_metrics(valid_y, valid_s, test_y, test_s)}
    result = {key: metrics[key] for key in ("accuracy", "accuracy_05", "accuracy_acc", "f1", "precision", "recall", "auroc", "auprc")} | {"valid_auroc": roc_auc_score(valid_y, valid_p)}
    return result


def best_weight(valid_y, first, second):

    scores = [roc_auc_score(valid_y, w * first + (1 - w) * second) for w in WEIGHTS]
    return float(WEIGHTS[int(np.argmax(scores))])


def bootstrap(labels, first, second, rounds=1000, seed=0):
    generator = np.random.default_rng(seed)
    deltas = []
    while len(deltas) < rounds:
        index = generator.integers(0, len(labels), len(labels))
        if labels[index].min() != labels[index].max():
            deltas.append(roc_auc_score(labels[index], second[index]) - roc_auc_score(labels[index], first[index]))
    deltas = np.array(deltas)
    return {"delta_auroc": roc_auc_score(labels, second) - roc_auc_score(labels, first), "ci_thap": np.percentile(deltas, 2.5), "ci_cao": np.percentile(deltas, 97.5), "ti_le_duong": float(np.mean(deltas > 0))}


def token_lengths(dataset):
    cache = Path(f"results/cache/qwen_tokens_{dataset}_test.json")
    if cache.exists():
        return json.loads(cache.read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    lengths = {}
    with open(f"data/interim/{dataset}/test.jsonl", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            lengths[row["id"]] = len(tokenizer(row["func"], add_special_tokens=False)["input_ids"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lengths))
    return lengths


def run(dataset):
    preds = Path("results/preds")
    lora = Scores(preds / f"{dataset}_sup_llm_LoRA-qwen15_co_do_thi_s42.npz")
    lora_all = Scores(preds / f"{dataset}_sup_llm_LoRA-qwen15_toan_bo_s42.npz")
    valid_ids, valid_y, test_ids, test_y = lora.valid_ids, lora.valid_y, lora.test_ids, lora.test_y
    lora_v, lora_t = lora.get("valid", valid_ids), lora.get("test", test_ids)

    def ev(valid_labels, valid_p, labels, test_p):

        language_ids = (valid_ids, test_ids) if len(labels) == len(test_ids) else (lora_all.valid_ids, lora_all.test_ids)
        language_evaluations.append((*language_ids, valid_labels, valid_p, labels, test_p))
        return evaluate(valid_labels, valid_p, labels, test_p)

    language_evaluations = []
    lengths = token_lengths(dataset)
    long = np.array([lengths[item] > 512 for item in test_ids])
    groups = {"token > 512 (bị cắt)": long, "token <= 512": ~long}
    seeds = sorted(int(path.stem.rsplit("_s", 1)[1]) for path in preds.glob(f"{dataset}_sup_llm_gnn_qwen15_s*.npz"))
    rows = []

    def add(part, group, model, seed, metrics, labels, note=""):
        rows.append({"dataset": dataset, "phan": part, "nhom": group, "mo_hinh": model, "seed": seed, "n": len(labels), "ti_le_loi": float(np.mean(labels)), **metrics, "ghi_chu": note})
        if language_evaluations:
            v_ids, t_ids, v_y, v_p, t_y, t_p = language_evaluations.pop()
            append_language_result(dataset, {"huong": "giam_sat", "cau_hinh": "Kết hợp", "mo_hinh": model, "tap": group, "seed": seed, "cach": "phan_tich_gnn", "nguong": part}, v_ids, v_y, logit(v_p), t_ids, t_y, logit(t_p))

    def by_length(model, seed, probability, threshold):
        for group, mask in groups.items():
            scores = logit(probability[mask])
            metrics = {**binary_metrics(test_y[mask], scores, threshold), **fixed_metrics(test_y[mask], scores)}
            add("do_dai", group, model, seed, {key: metrics[key] for key in ("accuracy", "accuracy_05", "f1", "precision", "recall", "auroc", "auprc")}, test_y[mask], "ngưỡng tối ưu F1 trên toàn bộ valid")

    add("ket_hop", "co_do_thi", "LoRA-qwen15", 42, ev(valid_y, lora_v, test_y, lora_t), test_y)
    by_length("LoRA-qwen15", 42, lora_t, lora.threshold)
    for path in sorted(preds.glob(f"{dataset}_sup_gnn_none_s*.npz")):
        gnn_only = Scores(path)
        seed = int(path.stem.rsplit("_s", 1)[1])
        add("ket_hop", "co_do_thi", "GNN (RGCN)", seed, ev(valid_y, gnn_only.get("valid", valid_ids), test_y, gnn_only.get("test", test_ids)), test_y)
        by_length("GNN (RGCN)", seed, gnn_only.get("test", test_ids), gnn_only.threshold)
    for seed in seeds:
        gnn = Scores(preds / f"{dataset}_sup_llm_gnn_qwen15_s{seed}.npz")
        mlp_q = Scores(preds / f"{dataset}_sup_llm_MLP-qwen15_co_do_thi_s{seed}.npz")
        mlp_uq = Scores(preds / f"{dataset}_sup_llm_MLP-unixcoder+qwen15_co_do_thi_s{seed}.npz")
        gnn_v, gnn_t = gnn.get("valid", valid_ids), gnn.get("test", test_ids)
        mq_v, mq_t = mlp_q.get("valid", valid_ids), mlp_q.get("test", test_ids)
        base_v, base_t = (lora_v + mlp_uq.get("valid", valid_ids)) / 2, (lora_t + mlp_uq.get("test", test_ids)) / 2
        add("ket_hop", "co_do_thi", "MLP-qwen15", seed, ev(valid_y, mq_v, test_y, mq_t), test_y)
        add("ket_hop", "co_do_thi", "LLM+GNN (RGCN+node-unixcoder+func-qwen15)", seed, ev(valid_y, gnn_v, test_y, gnn_t), test_y)
        add("ket_hop", "co_do_thi", "LoRA + MLP-unixcoder+qwen15 (trung bình đều)", seed, ev(valid_y, base_v, test_y, base_t), test_y)
        by_length("MLP-qwen15", seed, mq_t, mlp_q.threshold)
        by_length("LLM+GNN (RGCN+node-unixcoder+func-qwen15)", seed, gnn_t, gnn.threshold)
        combos = {}
        for name, (first_v, first_t, second_v, second_t) in {"LoRA + MLP-qwen15 (w)": (lora_v, lora_t, mq_v, mq_t), "LoRA + LLM+GNN (w)": (lora_v, lora_t, gnn_v, gnn_t), "LoRA + MLP-unixcoder+qwen15 + LLM+GNN (w)": (base_v, base_t, gnn_v, gnn_t)}.items():
            w = best_weight(valid_y, first_v, second_v)
            combos[name] = (w * first_v + (1 - w) * second_v, w * first_t + (1 - w) * second_t)
            add("ket_hop", "co_do_thi", name, seed, {**ev(valid_y, *combos[name][:1], test_y, combos[name][1]), "w": w}, test_y, "w là trọng số của thành phần đứng trước, chọn theo AUROC valid")
        combos["LoRA + MLP-unixcoder+qwen15 (trung bình đều)"] = (base_v, base_t)
        threshold, _ = best_threshold(valid_y, logit(combos["LoRA + LLM+GNN (w)"][0]))
        by_length("LoRA + LLM+GNN (w)", seed, combos["LoRA + LLM+GNN (w)"][1], threshold)
        for name, first, second in (("LoRA + LLM+GNN (w) trừ LoRA", lora_t, combos["LoRA + LLM+GNN (w)"][1]), ("LoRA + MLP-qwen15 (w) trừ LoRA", lora_t, combos["LoRA + MLP-qwen15 (w)"][1]), ("LoRA + MLP + LLM+GNN (w) trừ LoRA + MLP", base_t, combos["LoRA + MLP-unixcoder+qwen15 + LLM+GNN (w)"][1])):
            add("bootstrap", "co_do_thi", name, seed, bootstrap(test_y, first, second), test_y, "hiệu AUROC test, bootstrap 1000 lần, khoảng tin cậy 95%")

        all_valid_ids, all_valid_y, all_test_ids, all_test_y = lora_all.valid_ids, lora_all.valid_y, lora_all.test_ids, lora_all.test_y
        add("toan_bo", "toan_bo", "LoRA-qwen15", 42, ev(all_valid_y, lora_all.get("valid", all_valid_ids), all_test_y, lora_all.get("test", all_test_ids)), all_test_y) if seed == seeds[0] else None
        for name, (combo_v, combo_t) in combos.items():
            valid_table, test_table = {**lora_all.valid, **dict(zip(valid_ids, combo_v))}, {**lora_all.test, **dict(zip(test_ids, combo_t))}
            metrics = ev(all_valid_y, np.array([valid_table[item] for item in all_valid_ids]), all_test_y, np.array([test_table[item] for item in all_test_ids]))
            add("toan_bo", "toan_bo", name, seed, metrics, all_test_y, f"{len(all_test_ids) - len(test_ids)} hàm test không có đồ thị dùng xác suất LoRA")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phan"], row["nhom"], row["mo_hinh"])].append(row)
    for members in grouped.values():
        if len(members) > 1:
            mean = {**members[0], "seed": "tb", "ghi_chu": f"trung bình ± độ lệch chuẩn của {len(members)} seed"}
            for metric in METRICS:
                if metric in members[0]:
                    values = [member[metric] for member in members]
                    mean[metric] = f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"
            rows.append(mean)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, np.floating)):
                row[key] = f"{value:.4f}"
    output = Path("results/tables/phan_tich_gnn.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if output.exists():
        with open(output, encoding="utf-8", newline="") as file:
            old = [row for row in csv.DictReader(file) if row["dataset"] != dataset]
    with open(output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(old + rows)
    print(len(rows), "dòng;", {group: int(mask.sum()) for group, mask in groups.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    run(parser.parse_args().dataset)
