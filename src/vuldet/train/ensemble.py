# Nguyễn Tâm Bình - 2324801030003
import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vuldet.eval.metrics import accuracy_metrics, best_threshold, binary_metrics, fixed_metrics
from vuldet.train.common import append_result, report


def load_members(dataset):

    members = {}
    for path in sorted(Path("results/preds").glob(f"{dataset}_sup_*.npz")):
        match = re.fullmatch(rf"{dataset}_sup_(.+)_s(\d+)", path.stem)
        if "_toan_bo" in match.group(1):
            continue
        data = np.load(path)
        if "valid_score" in data:
            members.setdefault(match.group(1).replace("_co_do_thi", ""), {})[int(match.group(2))] = data
    return members


def aligned(data, prefix, reference):
    ids, scores = (data["valid_ids"], data["valid_score"]) if prefix == "valid" else (data["ids"], data["score"])
    lookup = dict(zip(ids, 1 / (1 + np.exp(-scores.astype(np.float64)))))
    return np.array([lookup[item] for item in reference])


def average(members, names, seed, prefix, reference):

    probability = np.mean([aligned(members[name].get(seed, members[name][42]), prefix, reference) for name in names], axis=0)
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def run(args):
    members = {name: seeds for name, seeds in load_members(args.dataset).items() if 42 in seeds}
    first = next(iter(members.values()))[42]
    valid_ids, valid_y, test_ids, test_y = first["valid_ids"], first["valid_y"], first["ids"], first["y"]

    chosen, best = [], 0.0
    while True:
        gains = {name: roc_auc_score(valid_y, average(members, chosen + [name], 42, "valid", valid_ids)) for name in members if name not in chosen}
        if not gains:
            break
        name = max(gains, key=gains.get)
        print(f"thêm {name}: valid_auroc {gains[name]:.4f} (đang có {best:.4f})")
        if gains[name] <= best + 1e-4:
            break
        chosen.append(name)
        best = gains[name]
    print("chọn", chosen)
    if args.dry_run or len(chosen) < 2:
        return
    model_name = "Ensemble(" + "+".join(chosen) + ")"
    for seed in args.seeds:
        valid_scores, scores = average(members, chosen, seed, "valid", valid_ids), average(members, chosen, seed, "test", test_ids)
        threshold, valid_f1 = best_threshold(valid_y, valid_scores)
        metrics = {**binary_metrics(test_y, scores, threshold), **fixed_metrics(test_y, scores), **accuracy_metrics(valid_y, valid_scores, test_y, scores)}

        append_result({"dataset": args.dataset, "huong": "giam_sat", "cau_hinh": "Kết hợp", "mo_hinh": model_name, "tap": "co_do_thi", "seed": seed, "n_test": len(test_y), **metrics, "ghi_chu": f"trung bình xác suất; chọn theo AUROC valid seed 42 = {best:.4f}; valid_f1={valid_f1:.4f}"}, (valid_ids, valid_y, valid_scores, test_ids, test_y, scores))
        np.savez(f"results/preds/{args.dataset}_ensemble_s{seed}.npz", ids=test_ids, y=test_y, score=scores, threshold=threshold, valid_ids=valid_ids, valid_y=valid_y, valid_score=valid_scores)
        print(seed, metrics)
        if seed == 42 and not args.no_report:
            report(f"XONG {args.dataset} giam_sat KetHop {model_name} Acc={metrics['accuracy']:.4f} Acc@0.5={metrics['accuracy_05']:.4f} F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    run(parser.parse_args())
