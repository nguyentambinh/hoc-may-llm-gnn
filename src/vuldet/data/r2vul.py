# Nguyễn Tâm Bình - 2324801030003
import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import pyarrow as pa

LANGUAGES = ("c", "c_sharp", "java", "javascript", "python")
SPLITS = ("train", "valid", "test")
META_FIELDS = ("cve_id", "cwe_id", "repo", "commit_URL", "parent_commit_sha", "origin", "date", "file")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_key(row, original=False):
    return (row["lang"], row["function" if original else "func"], int(row["vulnerable" if original else "target"]), row.get("cve_id") or "", row.get("origin") or "")


def arrow_metadata(raw, split):
    folder = Path(raw) / ("validation" if split == "valid" else split)
    paths = sorted(folder.glob("*.arrow"))
    if not paths:
        raise FileNotFoundError(f"Không có Arrow: {folder}")
    index = defaultdict(deque)
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            try:
                table = pa.ipc.open_stream(source).read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                table = pa.ipc.open_file(source).read_all()
            for row in table.select(["lang", "function", "vulnerable", *META_FIELDS]).to_pylist():
                if row["lang"] in LANGUAGES:
                    meta = {key: row[key] for key in META_FIELDS}
                    meta["cwe_id"] = list(row["cwe_id"] or [])
                    index[source_key(row, True)].append(meta)
    return index


def split_records(source, raw, split):
    index = arrow_metadata(raw, split)
    for k, row in enumerate(read_jsonl(Path(source) / f"{split}.jsonl")):
        key = source_key(row)
        if row["lang"] not in LANGUAGES or int(row["target"]) not in (0, 1) or not isinstance(row["func"], str) or not row["func"].strip():
            raise ValueError(f"Mẫu không hợp lệ: {split}:{k}")
        if not index.get(key):
            raise ValueError(f"Không khớp metadata Arrow: {split}:{k}")
        yield {"id": f"r2vul_{split}_{k}", "func": row["func"], "target": int(row["target"]), "lang": row["lang"], "split": split, "meta": index[key].popleft()}
    if any(index.values()):
        raise ValueError(f"Bản chia thiếu {sum(map(len, index.values()))} mẫu Arrow của 5 ngôn ngữ: {split}")


def prepare(source="du_lieu_da_chia/r2vul5", raw="r2vul/goc/r2vul_dataset", output="data/interim/r2vul"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for split in SPLITS:
        temporary = output / f"{split}.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for row in split_records(source, raw, split):
                counts[split, row["lang"], row["target"]] += 1
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output / f"{split}.jsonl")
    print("split\tlang\tan_toan\tloi\ttong")
    for split in SPLITS:
        for lang in LANGUAGES:
            safe, vulnerable = (counts[split, lang, target] for target in (0, 1))
            print(f"{split}\t{lang}\t{safe}\t{vulnerable}\t{safe + vulnerable}")
        print(f"{split}\tALL\t{sum(counts[split, lang, 0] for lang in LANGUAGES)}\t{sum(counts[split, lang, 1] for lang in LANGUAGES)}\t{sum(counts[split, lang, target] for lang in LANGUAGES for target in (0, 1))}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="du_lieu_da_chia/r2vul5")
    parser.add_argument("--raw", default="r2vul/goc/r2vul_dataset")
    parser.add_argument("--output", default="data/interim/r2vul")
    args = parser.parse_args()
    prepare(args.source, args.raw, args.output)
