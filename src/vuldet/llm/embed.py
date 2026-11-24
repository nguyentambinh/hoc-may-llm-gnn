# Nguyễn Tâm Bình - 2324801030003
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from vuldet.utils.checkpoint import SPLITS, batches, config, fingerprint, guard, read_jsonl, write_json
from vuldet.utils.log import get_logger

MODEL = "microsoft/unixcoder-base"
MODELS = {"unixcoder": MODEL, "qwen15": "Qwen/Qwen2.5-Coder-1.5B"}
CHUNK_SIZE = 2000


def node_records(path):
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            yield {"id": str(index), "func": json.loads('"' + line.rstrip("\n") + '"')}


def encode(texts, tokenizer, model, max_length, device="cpu"):
    prefix = [tokenizer.cls_token_id, tokenizer.convert_tokens_to_ids("<encoder-only>"), tokenizer.sep_token_id]
    tokenized = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_length - 4)["input_ids"]
    sequences = [tokens + [tokenizer.sep_token_id] for tokens in tokenized]
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention[index, :len(sequence)] = 1
    input_ids = input_ids.to(device)
    attention = attention.to(device)
    with torch.inference_mode():
        hidden = model(input_ids=input_ids, attention_mask=attention).last_hidden_state
        mask = attention.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.float().cpu().to(torch.float16).numpy()


def encode_decoder(texts, tokenizer, model, max_length, device="cpu"):
    inputs = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.float().cpu().to(torch.float16).numpy()


def merge_parts(parts, target, ids_target, hidden_size):
    count = sum(part["count"] for part in parts)
    temporary = target.with_name(target.name + ".tmp")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(count, hidden_size))
    offset = 0
    ids_temporary = ids_target.with_name(ids_target.name + ".tmp") if ids_target else None
    ids_stream = ids_temporary.open("w", encoding="utf-8") if ids_temporary else None
    try:
        for part in parts:
            values = np.load(part["array"], mmap_mode="r", allow_pickle=False)
            matrix[offset:offset + len(values)] = values
            offset += len(values)
            del values
            if ids_stream:
                with Path(part["ids"]).open(encoding="utf-8") as stream:
                    for line in stream:
                        ids_stream.write(line)
        matrix.flush()
    finally:
        del matrix
        if ids_stream:
            ids_stream.close()
    temporary.replace(target)
    if ids_temporary:
        ids_temporary.replace(ids_target)


def embed(dataset, mode, model_key="unixcoder"):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    if model_key != "unixcoder" and mode != "func":
        raise ValueError(f"Model {model_key} only supports mode func; mode node is not supported")
    device = os.environ.get("EMBED_DEVICE", "cpu")
    batch_override = os.environ.get("EMBED_BATCH")
    if batch_override is not None:
        batch_override = int(batch_override)
        if batch_override <= 0:
            raise ValueError("EMBED_BATCH must be a positive integer")
    model_name = MODELS[model_key]
    encoder = encode if model_key == "unixcoder" else encode_decoder
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    settings = config(dataset)
    output = Path(settings["embeddings"])
    if model_key != "unixcoder":
        output = output / model_key
    output.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"embed_{dataset}" if model_key == "unixcoder" else f"embed_{dataset}_{model_key}")
    tokenizer = model = None
    groups = SPLITS if mode == "func" else ("node_codes",)
    for group in groups:
        source = Path(settings["interim"]) / f"{group}.jsonl" if mode == "func" else Path(settings["graphs"]) / "node_codes.txt"
        name = f"func_{group}" if mode == "func" else "node_codes"
        max_length, batch_size = (512, 16) if mode == "func" else (32, 128)
        if model_key != "unixcoder":
            max_length, batch_size = 512, 8
        if batch_override is not None:
            batch_size = batch_override
        signature = fingerprint([source], {"model": model_name, "mode": mode, "max_length": max_length, "chunk_size": CHUNK_SIZE, "pooling": "masked_mean", "tokens": "encoder-only" if model_key == "unixcoder" else "decoder"})
        checkpoint = output / "parts" / name
        checkpoint.mkdir(parents=True, exist_ok=True)
        guard(checkpoint / "input.json", signature)
        target = output / f"{name}.npy"
        ids_target = output / f"{name}_ids.txt" if mode == "func" else None
        final_done = output / f"{name}.done"
        if final_done.exists() and target.exists() and (ids_target is None or ids_target.exists()):
            if json.loads(final_done.read_text(encoding="utf-8")) == signature:
                logger.info("%s complete; skipped", name)
                continue
        items = read_jsonl(source) if mode == "func" else node_records(source)
        parts = []
        if model_key != "unixcoder" and model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
            model.requires_grad_(False)
        hidden_size = 768 if model_key == "unixcoder" else model.config.hidden_size
        for index, chunk in enumerate(batches(items, CHUNK_SIZE)):
            part = checkpoint / f"part_{index:05d}.npy"
            ids = checkpoint / f"part_{index:05d}_ids.txt"
            done = checkpoint / f"part_{index:05d}.done"
            expected_ids = [item["id"] for item in chunk]
            if done.exists() and part.exists() and ids.exists():
                values = np.load(part, mmap_mode="r", allow_pickle=False)
                if values.shape[0] != len(chunk) or values.ndim != 2 or values.dtype != np.float16 or ids.read_text(encoding="utf-8").splitlines() != expected_ids:
                    raise RuntimeError(f"Invalid embedding checkpoint: {part}")
                hidden_size = values.shape[1]
                del values
                logger.info("%s part=%d skipped", name, index)
            else:
                if model is None:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if model_key != "unixcoder" and tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval() if model_key != "unixcoder" else AutoModel.from_pretrained(MODEL).to(device).eval()
                    model.requires_grad_(False)
                start = time.monotonic()
                hidden_size = model.config.hidden_size
                temporary = part.with_name(part.name + ".tmp")
                matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(len(chunk), hidden_size))
                offset = 0
                for batch in batches(chunk, batch_size):
                    matrix[offset:offset + len(batch)] = encoder([item["func"] for item in batch], tokenizer, model, max_length, device)
                    offset += len(batch)
                matrix.flush()
                del matrix
                temporary.replace(part)
                ids_temporary = ids.with_name(ids.name + ".tmp")
                ids_temporary.write_text("".join(record_id + "\n" for record_id in expected_ids), encoding="utf-8")
                ids_temporary.replace(ids)
                elapsed = time.monotonic() - start
                write_json(done, {"count": len(chunk), "hidden_size": hidden_size})
                logger.info("%s part=%d count=%d seconds=%.2f items_per_second=%.2f", name, index, len(chunk), elapsed, len(chunk) / max(elapsed, 1e-6))
            parts.append({"array": part, "ids": ids, "count": len(chunk)})
        merge_parts(parts, target, ids_target, hidden_size)
        write_json(final_done, signature)
        logger.info("%s merged count=%d", name, sum(part["count"] for part in parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--mode", required=True, choices=("func", "node"))
    parser.add_argument("--model", choices=tuple(MODELS), default="unixcoder")
    args = parser.parse_args()
    embed(args.dataset, args.mode, args.model)
