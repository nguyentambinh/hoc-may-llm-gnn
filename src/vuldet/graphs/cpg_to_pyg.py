# Nguyễn Tâm Bình - 2324801030003
import argparse
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import torch

from vuldet.utils.checkpoint import SPLITS, config, fingerprint, guard, read_jsonl, records, write_json

NODE_LABELS = ("UNKNOWN", "ANNOTATION", "ANNOTATION_LITERAL", "ANNOTATION_PARAMETER", "ANNOTATION_PARAMETER_ASSIGN", "ARRAY_INITIALIZER", "BINDING", "BLOCK", "CALL", "CLOSURE_BINDING", "COMMENT", "CONFIG_FILE", "CONTROL_STRUCTURE", "DEPENDENCY", "FIELD_IDENTIFIER", "FILE", "IDENTIFIER", "IMPORT", "JUMP_LABEL", "JUMP_TARGET", "LITERAL", "LOCAL", "MEMBER", "META_DATA", "METHOD", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_REF", "METHOD_RETURN", "MODIFIER", "NAMESPACE", "NAMESPACE_BLOCK", "RETURN", "TAG", "TAG_NODE_PAIR", "TYPE", "TYPE_ARGUMENT", "TYPE_DECL", "TYPE_PARAMETER", "TYPE_REF")
NODE_TYPES = {label: index for index, label in enumerate(NODE_LABELS)}
EDGE_TYPES = {"AST": 0, "CFG": 1, "CDG": 2, "DDG": 3}


def node_labels(dataset, raw_files):

    extra = {"FINDING", "KEY_VALUE_PAIR"}
    for path in raw_files:
        for item in read_jsonl(path):
            extra.update(node[1] for node in item["nodes"])
    return (*NODE_LABELS, *sorted(extra - set(NODE_LABELS)))


def code_index(database, code):
    code = " ".join(str(code or "").split())
    row = database.execute("SELECT id FROM codes WHERE code = ?", (code,)).fetchone()
    if row is not None:
        return row[0]
    index = database.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM codes").fetchone()[0]
    database.execute("INSERT INTO codes VALUES (?, ?)", (index, code))
    return index


def make_graph(item, target, split, database, max_nodes, node_types=None):
    from torch_geometric.data import Data
    node_types = NODE_TYPES if node_types is None else node_types

    nodes = list({tuple(node[:3]): node for node in item["nodes"]}.values()) if len({node[0] for node in item["nodes"]}) != len(item["nodes"]) else item["nodes"]
    nodes = nodes[:max_nodes]
    mapping = {node[0]: index for index, node in enumerate(nodes)}
    if len(mapping) != len(nodes):
        raise ValueError(f"Duplicate node ids: {item['id']}")
    edges = [(mapping[source], mapping[destination], EDGE_TYPES[kind]) for source, destination, kind in item["edges"] if source in mapping and destination in mapping and kind in EDGE_TYPES]
    edge_index = torch.tensor([(source, destination) for source, destination, _ in edges], dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return Data(node_type=torch.tensor([node_types.get(node[1], 0) for node in nodes], dtype=torch.long), code_id=torch.tensor([code_index(database, node[2]) for node in nodes], dtype=torch.long), edge_index=edge_index, edge_type=torch.tensor([kind for _, _, kind in edges], dtype=torch.long), y=torch.tensor([target], dtype=torch.long), id=item["id"], split=split, num_nodes=len(nodes), num_orig_nodes=len(item["nodes"]), truncated=len(item["nodes"]) > max_nodes)


def save_shard(output, split, index, graphs, database):
    database.commit()
    target = output / f"{split}_{index:03d}.pt"
    if target.exists():
        return
    temporary = target.with_suffix(".pt.tmp")
    torch.save(graphs, temporary)
    temporary.replace(target)


def convert(dataset, rebuild=False):
    settings = config(dataset)
    root = Path(settings["graphs"])
    output = root / "pyg"
    raw_files = sorted((root / "raw").glob("batch_*.jsonl"))
    if not raw_files:
        raise FileNotFoundError("No raw graph batches")
    for path in raw_files:
        if not path.with_suffix(".done").exists():
            raise RuntimeError(f"Unfinished Joern batch: {path}")
    labels = node_labels(dataset, raw_files)
    types = {label: index for index, label in enumerate(labels)}
    signature = fingerprint([*(Path(settings["interim"]) / f"{split}.jsonl" for split in SPLITS), *raw_files], {"max_nodes": settings["max_nodes"], "shard_size": settings["shard_size"], "node_labels": labels})
    if rebuild:
        if output.exists():
            shutil.rmtree(output)
        (root / "codes.sqlite").unlink(missing_ok=True)
        (root / "node_codes.txt").unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    guard(output / "input.json", signature)
    if int(settings["max_nodes"]) <= 0 or int(settings["shard_size"]) <= 0:
        raise ValueError("max_nodes and shard_size must be positive")
    if any(output.glob("*.pt")) and not (root / "codes.sqlite").exists():
        raise RuntimeError("Missing codes.sqlite for existing graph shards")
    database = sqlite3.connect(root / "codes.sqlite")
    try:
        database.execute("CREATE TABLE IF NOT EXISTS codes (id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE)")
        database.execute("CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, target INTEGER, split TEXT, seen INTEGER DEFAULT 0)")
        database.execute("DELETE FROM samples")
        languages = {}
        for item in records(settings):
            database.execute("INSERT INTO samples(id, target, split) VALUES (?, ?, ?)", (item["id"], int(item["target"]), item["split"]))
            languages[item["id"]] = item["lang"]
        database.commit()
        sample_count = database.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        expected_batches = (sample_count + int(settings["batch_size"]) - 1) // int(settings["batch_size"])
        if [path.name for path in raw_files] != [f"batch_{index:05d}.jsonl" for index in range(expected_batches)]:
            raise RuntimeError("Joern export is incomplete; finish all batches before converting")
        buffers = {split: [] for split in SPLITS}
        counts = Counter()
        truncated = 0
        for path in raw_files:
            for item in read_jsonl(path):
                row = database.execute("SELECT target, split, seen FROM samples WHERE id = ?", (item["id"],)).fetchone()
                if row is None or row[2]:
                    raise ValueError(f"Unknown or duplicate graph id: {item['id']}")
                target, split, _ = row
                database.execute("UPDATE samples SET seen = 1 WHERE id = ?", (item["id"],))
                existing_shard = output / f"{split}_{counts[split] // int(settings['shard_size']):03d}.pt"
                counts[split] += 1
                truncated += int(len(item["nodes"]) > int(settings["max_nodes"]))
                if existing_shard.exists():
                    continue
                graph = make_graph(item, target, split, database, int(settings["max_nodes"]), types)
                graph.lang = languages[item["id"]]
                buffers[split].append(graph)
                if len(buffers[split]) == int(settings["shard_size"]):
                    save_shard(output, split, (counts[split] - 1) // int(settings["shard_size"]), buffers[split], database)
                    buffers[split] = []
        for split in SPLITS:
            if buffers[split]:
                save_shard(output, split, (counts[split] - 1) // int(settings["shard_size"]), buffers[split], database)
        database.commit()
        temporary = root / "node_codes.txt.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for (code,) in database.execute("SELECT code FROM codes ORDER BY id"):
                stream.write(json.dumps(code, ensure_ascii=False)[1:-1] + "\n")
        temporary.replace(root / "node_codes.txt")
        missing = dict(database.execute("SELECT split, COUNT(*) FROM samples WHERE seen = 0 GROUP BY split"))
        missing_by_label = dict(database.execute("SELECT target, COUNT(*) FROM samples WHERE seen = 0 GROUP BY target"))
        write_json(f"results/tables/graphs_{dataset}.json", {"graphs": {split: counts[split] for split in SPLITS}, "truncated": truncated, "missing": sum(missing.values()), "missing_by_label": {str(label): missing_by_label.get(label, 0) for label in (0, 1)}, "missing_by_split": {split: missing.get(split, 0) for split in SPLITS}, "node_labels": labels, "edge_types": EDGE_TYPES})
    finally:
        database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    convert(args.dataset, args.rebuild)
