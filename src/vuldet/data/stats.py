# Nguyễn Tâm Bình - 2324801030003
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vuldet.utils.checkpoint import SPLITS, config, read_jsonl, write_json


def quantiles(values):
    return dict(zip(("50", "90", "99", "max"), np.percentile(values, [50, 90, 99, 100]).tolist())) if values else {key: None for key in ("50", "90", "99", "max")}


def summarize(dataset):
    settings = config(dataset)
    summary = {"dataset": dataset, "splits": {}}
    projects, cwes = Counter(), Counter()
    all_lines, all_characters = [], []
    for split in SPLITS:
        labels, lines, characters = Counter(), [], []
        for item in read_jsonl(Path(settings["interim"]) / f"{split}.jsonl"):
            labels[int(item["target"])] += 1
            lines.append(len(item["func"].splitlines()))
            characters.append(len(item["func"]))
            meta = item.get("meta", {})
            project = meta.get("repo")
            if project:
                projects[str(project)] += 1
            cwe = meta.get("cwe_id")
            if cwe:
                cwes.update(str(value) for value in (cwe if isinstance(cwe, list) else [cwe]))
        count = sum(labels.values())
        summary["splits"][split] = {"count": count, "labels": {str(label): {"count": labels[label], "ratio": labels[label] / count if count else 0} for label in (0, 1)}, "lines": quantiles(lines), "characters": quantiles(characters)}
        all_lines.extend(lines)
        all_characters.extend(characters)
    summary.update({"lines": quantiles(all_lines), "characters": quantiles(all_characters), "top_projects": projects.most_common(10), "top_cwe": cwes.most_common(20)})
    write_json(f"results/tables/stats_{dataset}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    summarize(parser.parse_args().dataset)
