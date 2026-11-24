# Nguyễn Tâm Bình - 2324801030003
import hashlib
import json
from itertools import islice
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")


def config(dataset):
    with open("configs/datasets.yaml", encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    return settings["defaults"] | settings["datasets"][dataset]


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def records(settings):
    for split in SPLITS:
        yield from read_jsonl(Path(settings["interim"]) / f"{split}.jsonl")


def batches(items, size):
    if size <= 0:
        raise ValueError("Batch size must be positive")
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(paths, options=None):
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    for path in paths:
        path = Path(path)
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def guard(path, signature):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError(f"Input changed; move or remove old outputs in {path.parent} before rerunning")
    write_json(path, signature)
