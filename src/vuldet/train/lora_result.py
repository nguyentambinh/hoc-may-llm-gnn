# Nguyễn Tâm Bình - 2324801030003
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vuldet.eval.metrics import accuracy_metrics, best_threshold, binary_metrics, fixed_metrics
from vuldet.train.common import append_result, graph_id_sets, load_interim, report


def read_predictions(path, dataset, split):
    labels = dict(zip(*load_interim(dataset, split)[:2]))
    with open(f"data/interim/{dataset}/{split}.jsonl", encoding="utf-8") as stream:
        languages = {item["id"]: item["lang"] for item in map(json.loads, stream) if item}
    ids, targets, logits = [], [], []
    with open(path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["id"].startswith(f"r2vul5_{split}_"):
                row["id"] = "r2vul_" + row["id"][len("r2vul5_"):]
            if labels.get(row["id"]) != int(float(row["target"])):
                raise ValueError(f"{path}: id {row['id']} không có trong interim {split} hoặc lệch nhãn")
            if row.get("lang") and row["lang"] != languages[row["id"]]:
                raise ValueError(f"{path}: ngôn ngữ không khớp interim: {row['id']}")
            probability = min(max(float(row["prob"]), 1e-7), 1 - 1e-7)
            ids.append(row["id"])
            targets.append(labels[row["id"]])
            logits.append(np.log(probability / (1 - probability)))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: trùng id")
    return np.array(ids), np.array(targets), np.array(logits), len(labels)


def run(args):
    folder = Path(args.dir or ("results/lora/r2vul5"))
    info = json.loads((folder / "info.json").read_text(encoding="utf-8")) if (folder / "info.json").exists() else {}
    flat = {**{key: value for key, value in info.items() if isinstance(value, (str, int, float, bool))}, **{f"lora_{key}": value for key, value in (info.get("lora") or {}).items()}}
    if info.get("history"):
        best = max(info["history"], key=lambda item: item.get("valid_auroc", 0))
        flat.update(epoch_tot=best.get("epoch"), so_epoch=len(info["history"]), valid_auroc=round(best.get("valid_auroc", 0), 4))
    note = " ".join(f"{key}={value}" for key, value in flat.items()).replace(",", ";")[:400]
    graph_ids = graph_id_sets(args.dataset)
    data = {split: read_predictions(folder / f"pred_{split}.csv", args.dataset, split) for split in ("valid", "test")}
    for subset in ("co_do_thi", "toan_bo"):
        part = {}
        for split, (ids, targets, logits, _) in data.items():
            keep = np.array([subset == "toan_bo" or item in graph_ids[split] for item in ids])
            part[split] = (ids[keep], targets[keep], logits[keep])
        (valid_ids, valid_y, valid_scores), (ids, y, scores) = part["valid"], part["test"]
        threshold, valid_f1 = best_threshold(valid_y, valid_scores)
        metrics = {**binary_metrics(y, scores, threshold), **fixed_metrics(y, scores), **accuracy_metrics(valid_y, valid_scores, y, scores)}
        extra = f"valid_f1={valid_f1:.4f} test_co_du_doan={len(data['test'][0])}/{data['test'][3]}"

        print(subset, metrics, extra)
        if args.dry_run:
            continue
        append_result({"dataset": args.dataset, "huong": "giam_sat", "cau_hinh": "LLM", "mo_hinh": "LoRA-qwen15", "tap": subset, "seed": info.get("seed", 42), "n_train": info.get("n_train", info.get("train_size", "")), "n_test": len(y), **metrics, "ghi_chu": f"{extra} {note}"}, (valid_ids, valid_y, valid_scores, ids, y, scores))
        Path("results/preds").mkdir(parents=True, exist_ok=True)
        np.savez(f"results/preds/{args.dataset}_sup_llm_LoRA-qwen15_{subset}_s{info.get('seed', 42)}.npz", ids=ids, y=y, score=scores, threshold=threshold, valid_ids=valid_ids, valid_y=valid_y, valid_score=valid_scores)
        if subset == "co_do_thi" and not args.no_report:
            report(f"XONG {args.dataset} giam_sat LLM LoRA-qwen15 Acc={metrics['accuracy']:.4f} Acc@0.5={metrics['accuracy_05']:.4f} F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    run(parser.parse_args())
