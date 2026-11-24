# Nguyễn Tâm Bình - 2324801030003
import argparse
import copy
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from vuldet.eval.metrics import accuracy_metrics, best_threshold, binary_metrics, fixed_metrics
from vuldet.models.classifiers import MLP
from vuldet.train.common import SPLITS, append_result, graph_id_sets, load_func, load_interim, report, setup
from vuldet.utils.log import get_logger

MLP_GRID = {"hidden": [(512, 128), (256,), (1024, 256)], "dropout": [0.2, 0.5], "lr": [1e-3, 3e-4], "weight_decay": [1e-4, 1e-2], "class_weight": [True, False]}
MLP_DEFAULT = {"hidden": (512, 128), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "class_weight": True}


def mlp_scores(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(x[start:start + 4096])) for start in range(0, len(x), 4096)]).numpy()


def train_mlp(x_train, y_train, x_valid, y_valid, config, seed, epochs=40, patience=6):

    torch.manual_seed(seed)
    model = MLP(x_train.shape[1], config["hidden"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio if config["class_weight"] else 1.0))
    features, targets = torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))
    best_auroc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(features))
        for start in range(0, len(order), 256):
            index = order[start:start + 256]
            optimizer.zero_grad()
            loss_fn(model(features[index]), targets[index]).backward()
            optimizer.step()
        auroc = roc_auc_score(y_valid, mlp_scores(model, x_valid))
        if auroc > best_auroc:
            best_auroc, best_state, best_epoch, bad = auroc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auroc, best_epoch


def search_mlp(x, y, trials, logger):

    grid = [dict(zip(MLP_GRID, values)) for values in itertools.product(*MLP_GRID.values())]
    candidates = [MLP_DEFAULT] + random.Random(0).sample(grid, trials)
    best = None
    for config in candidates:
        _, auroc, epoch = train_mlp(x["train"], y["train"], x["valid"], y["valid"], config, 42)
        logger.info(f"dò MLP {config} valid_auroc {auroc:.4f} epoch {epoch}")
        if best is None or auroc > best[0]:
            best = (auroc, config)
    logger.info(f"chọn MLP {best[1]} valid_auroc {best[0]:.4f}")
    return best[1]




def test_metrics(y, valid_scores, test_scores):
    threshold, valid_f1 = best_threshold(y["valid"], valid_scores)
    metrics = {**binary_metrics(y["test"], test_scores, threshold), **fixed_metrics(y["test"], test_scores), **accuracy_metrics(y["valid"], valid_scores, y["test"], test_scores)}
    return metrics, valid_f1


def save_scores(name, ids, y, valid_scores, test_scores, threshold):
    np.savez(f"results/preds/{name}.npz", ids=ids["test"], y=y["test"], score=test_scores, threshold=threshold, valid_ids=ids["valid"], valid_y=y["valid"], valid_score=valid_scores)


def run(args):
    dataset, model_name = args.dataset, args.model
    threads = setup(42)
    logger = get_logger(f"train_{dataset}_llm_{model_name}")
    graph_ids = graph_id_sets(dataset)
    data = {}
    for split in SPLITS:
        interim_ids, labels, _ = load_interim(dataset, split)
        matrices = []
        for part in model_name.split("+"):
            matrix, ids = load_func(dataset, split, part)
            if ids != interim_ids:
                raise ValueError(f"{dataset} {split} {part}: thứ tự id embedding khác interim")
            matrices.append(matrix)
        data[split] = (np.concatenate(matrices, axis=1), labels, np.array(interim_ids), np.array([item in graph_ids[split] for item in interim_ids]))
    Path("results/preds").mkdir(parents=True, exist_ok=True)
    mlp_config = None


    for subset in ("co_do_thi", "toan_bo"):
        start = time.time()
        parts = {}
        for split, (matrix, labels, ids, has_graph) in data.items():
            keep = has_graph if subset == "co_do_thi" else np.ones(len(ids), dtype=bool)
            parts[split] = (matrix[keep], labels[keep], ids[keep])
        scaler = StandardScaler().fit(parts["train"][0])
        x = {split: scaler.transform(parts[split][0]).astype(np.float32) for split in SPLITS}
        y = {split: parts[split][1] for split in SPLITS}
        ids = {split: parts[split][2] for split in SPLITS}
        prediction_data = lambda valid, test: (ids["valid"], y["valid"], valid, ids["test"], y["test"], test)

        base = {"dataset": dataset, "cau_hinh": "LLM", "tap": subset, "n_train": len(y["train"]), "n_test": len(y["test"])}
        best = None
        for c in (0.001, 0.01, 0.1, 1.0):
            classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=1000).fit(x["train"], y["train"])
            auroc = roc_auc_score(y["valid"], classifier.decision_function(x["valid"]))
            logger.info(f"{subset} dò LR C={c} valid_auroc {auroc:.4f}")
            if best is None or auroc > best[0]:
                best = (auroc, classifier, c)

        lr_valid_auroc, classifier, c = best
        valid_scores, scores = classifier.decision_function(x["valid"]), classifier.decision_function(x["test"])
        lr_metrics, valid_f1 = test_metrics(y, valid_scores, scores)
        append_result({**base, **lr_metrics, "huong": "giam_sat", "mo_hinh": f"LR-{model_name}", "seed": 42, "ghi_chu": f"C={c} valid_auroc={lr_valid_auroc:.4f} valid_f1={valid_f1:.4f} (LR tất định, 1 seed)"}, prediction_data(valid_scores, scores))
        save_scores(f"{dataset}_sup_llm_LR-{model_name}_{subset}_s42", ids, y, valid_scores, scores, lr_metrics["threshold"])
        logger.info(f"{subset} LR test {lr_metrics}")
        if mlp_config is None:
            mlp_config = search_mlp(x, y, args.trials, logger) if args.trials else MLP_DEFAULT
        mlp_results = []
        for seed in args.seeds:
            mlp, mlp_valid_auroc, epoch = train_mlp(x["train"], y["train"], x["valid"], y["valid"], mlp_config, seed)
            valid_scores, scores = mlp_scores(mlp, x["valid"]), mlp_scores(mlp, x["test"])
            mlp_metrics, valid_f1 = test_metrics(y, valid_scores, scores)
            note = " ".join(f"{key}={value}" for key, value in mlp_config.items()).replace(",", "-")
            append_result({**base, **mlp_metrics, "huong": "giam_sat", "mo_hinh": f"MLP-{model_name}", "seed": seed, "ghi_chu": f"{note} epoch={epoch} valid_auroc={mlp_valid_auroc:.4f} valid_f1={valid_f1:.4f}"}, prediction_data(valid_scores, scores))
            save_scores(f"{dataset}_sup_llm_MLP-{model_name}_{subset}_s{seed}", ids, y, valid_scores, scores, mlp_metrics["threshold"])
            logger.info(f"{subset} MLP seed {seed} test {mlp_metrics}")
            mlp_results.append((mlp_valid_auroc, mlp_metrics))
        logger.info(f"{subset} xong sau {time.time() - start:.0f} giây")
        if subset == "co_do_thi" and not args.no_report:
            name, chosen = (f"MLP-{model_name}", mlp_results[0][1]) if mlp_results[0][0] > lr_valid_auroc else (f"LR-{model_name}", lr_metrics)
            report(f"XONG {dataset} giam_sat LLM {name} Acc={chosen['accuracy']:.4f} Acc@0.5={chosen['accuracy_05']:.4f} F1={chosen['f1']:.4f} AUROC={chosen['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--model", default="unixcoder", choices=("unixcoder", "qwen15", "unixcoder+qwen15"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--no-report", action="store_true")


    run(parser.parse_args())
