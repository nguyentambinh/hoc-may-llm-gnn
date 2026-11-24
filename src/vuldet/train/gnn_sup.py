# Nguyễn Tâm Bình - 2324801030003
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from vuldet.eval.metrics import accuracy_metrics, best_threshold, binary_metrics, fixed_metrics
from vuldet.models.gnn import GraphClassifier
from vuldet.train.common import SPLITS, append_result, load_func, load_graphs, node_features, report, setup
from vuldet.utils.log import get_logger


class FuncTable:
    def __init__(self, dataset, model):
        matrices, ids = zip(*(load_func(dataset, split, model) for split in SPLITS))
        mean, std = matrices[0].mean(0), matrices[0].std(0) + 1e-6
        self.matrix = (np.concatenate(matrices) - mean) / std
        self.index = {item: row for row, item in enumerate(sum(ids, []))}
        self.dim = self.matrix.shape[1]

    def rows(self, ids):
        return torch.from_numpy(self.matrix[[self.index[item] for item in ids]].astype(np.float32))


def forward(model, batch, node_matrix, func_table):
    node_llm = node_features(batch.code_id, node_matrix) if node_matrix is not None else None
    func = func_table.rows(batch.id) if func_table is not None else None
    return model(batch.node_type, node_llm, batch.edge_index, batch.edge_type, batch.batch, func)


@torch.no_grad()
def predict(model, loader, node_matrix, func_table):
    model.eval()
    scores, labels, ids = [], [], []
    for batch in loader:
        scores.append(forward(model, batch, node_matrix, func_table))
        labels.append(batch.y)
        ids.extend(batch.id)
    return torch.cat(scores).numpy(), torch.cat(labels).numpy().astype(int), np.array(ids)


def save(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(args):
    setup(args.seed)
    use_llm = args.mode == "llm_gnn"
    func_model = args.func_model if use_llm else "none"

    name = f"{args.dataset}_sup_{args.mode}_{func_model}_s{args.seed}"
    logger = get_logger(f"train_{name}")
    graphs = {split: load_graphs(args.dataset, split, args.limit) for split in SPLITS}
    node_matrix = np.load(f"data/embeddings/{args.dataset}/node_codes.npy", mmap_mode="r") if use_llm else None
    func_table = FuncTable(args.dataset, func_model) if func_model != "none" else None
    loaders = {split: DataLoader(graphs[split], batch_size=args.batch, shuffle=split == "train") for split in SPLITS}
    node_options = {"num_node_types": len(json.loads(Path("results/tables/graphs_r2vul.json").read_text(encoding="utf-8"))["node_labels"])}
    model = GraphClassifier(args.hidden, args.layers, 768 if use_llm else 0, func_table.dim if func_table else 0, args.dropout, **node_options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_labels = torch.cat([graph.y for graph in graphs["train"]]).float()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(train_labels == 0).sum() / (train_labels == 1).sum().clamp(min=1))
    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = checkpoint_dir / f"{name}_last.pt", checkpoint_dir / f"{name}_best.pt"
    state = {"epoch": 0, "best_auprc": -1.0, "best_epoch": 0, "bad": 0, "seconds": []}
    if last_path.exists() and not args.limit:
        saved = torch.load(last_path, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        logger.info(f"Chạy tiếp từ epoch {state['epoch']}")
    logger.info(f"{name}: {sum(p.numel() for p in model.parameters())} tham số, train {len(graphs['train'])}, valid {len(graphs['valid'])}, test {len(graphs['test'])}, args {vars(args)}")
    while state["epoch"] < args.epochs and state["bad"] < args.patience:
        start = time.time()
        model.train()
        total, count = 0.0, 0
        for step, batch in enumerate(loaders["train"], 1):
            optimizer.zero_grad()
            loss = loss_fn(forward(model, batch, node_matrix, func_table), batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
            if step % 50 == 0 and state["epoch"] == 0:
                logger.info(f"epoch 1 bước {step}/{len(loaders['train'])} loss {total / count:.4f} {time.time() - start:.0f} giây")
        scores, labels, _ = predict(model, loaders["valid"], node_matrix, func_table)
        _, f1 = best_threshold(labels, scores)
        valid = binary_metrics(labels, scores, 0.0)
        state["epoch"] += 1
        state["seconds"].append(time.time() - start)

        if valid["auprc"] > state["best_auprc"]:
            state.update(best_auprc=valid["auprc"], best_epoch=state["epoch"], bad=0)
            if not args.limit:
                save(model.state_dict(), best_path)
        else:
            state["bad"] += 1
        logger.info(f"epoch {state['epoch']} loss {total / count:.4f} valid_f1 {f1:.4f} valid_auroc {valid['auroc']:.4f} valid_auprc {valid['auprc']:.4f} {state['seconds'][-1]:.0f} giây")
        if not args.limit:
            save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, last_path)
    if args.limit:
        return
    model.load_state_dict(torch.load(best_path, weights_only=False))
    valid_scores, valid_labels, valid_ids = predict(model, loaders["valid"], node_matrix, func_table)
    threshold, valid_f1 = best_threshold(valid_labels, valid_scores)
    scores, labels, ids = predict(model, loaders["test"], node_matrix, func_table)
    metrics = {**binary_metrics(labels, scores, threshold), **fixed_metrics(labels, scores), **accuracy_metrics(valid_labels, valid_scores, labels, scores)}

    Path("results/preds").mkdir(parents=True, exist_ok=True)
    np.savez(f"results/preds/{name}.npz", ids=ids, y=labels, score=scores, threshold=threshold, valid_ids=valid_ids, valid_y=valid_labels, valid_score=valid_scores)
    config = "LLM+GNN" if use_llm else "GNN"
    model_name = ("RGCN" if not use_llm else f"RGCN+node-unixcoder+func-{func_model}")
    valid_auroc = binary_metrics(valid_labels, valid_scores, threshold)["auroc"]
    note = f"hidden={args.hidden} layers={args.layers} lr={args.lr} batch={args.batch} epoch_tot={state['best_epoch']}/{state['epoch']} valid_f1={valid_f1:.4f} valid_auroc={valid_auroc:.4f} valid_auprc={state['best_auprc']:.4f} giay/epoch={np.mean(state['seconds']):.0f}"
    append_result({"dataset": args.dataset, "huong": "giam_sat", "cau_hinh": config, "mo_hinh": model_name, "tap": "co_do_thi", "seed": args.seed, "n_train": len(graphs["train"]), "n_test": len(labels), **metrics, "ghi_chu": note}, (valid_ids, valid_labels, valid_scores, ids, labels, scores))
    logger.info(f"test {metrics} | {note}")
    if not args.no_report:
        report(f"XONG {args.dataset} giam_sat {config} {model_name} Acc={metrics['accuracy']:.4f} Acc@0.5={metrics['accuracy_05']:.4f} F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--mode", required=True, choices=("gnn", "llm_gnn"))
    parser.add_argument("--func-model", default="unixcoder", choices=("none", "unixcoder", "qwen15"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)


    parser.add_argument("--no-report", action="store_true")
    run(parser.parse_args())
