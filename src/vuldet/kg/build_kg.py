# Nguyễn Tâm Bình - 2324801030003
import argparse
import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import networkx as nx


MULTI_API_NAME = re.compile(r"^\s*(?:(?:await|new)\s+)?([A-Za-z_$][\w$]*(?:(?:\.|::|->)[A-Za-z_$][\w$]*)*)\s*\(")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def batches(raw_dir):
    paths = sorted(Path(raw_dir).glob("batch_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No batch_*.jsonl in {raw_dir}")
    for path in paths:
        yield from read_jsonl(path)


def api_calls(item):
    calls = Counter()
    for _, label, code, _ in item["nodes"]:
        match = MULTI_API_NAME.match(code or "") if label == "CALL" else None
        if match:
            calls[match.group(1)] += 1
    return calls


def statistics(graph):
    apis, faulty, cwes, projects = Counter(), Counter(), Counter(), Counter()
    for key, attrs in graph.nodes(data=True):
        if attrs["kind"] != "function":
            continue
        if attrs["target"] == 1:
            projects[attrs["project"]] += 1
        for _, dst, edge in graph.out_edges(key, data=True):
            if edge["rel"] == "has_cwe":
                cwes[dst.removeprefix("cwe:")] += 1
            elif edge["rel"] == "calls":
                api = dst.removeprefix("api:")
                apis[api] += 1
                faulty[api] += attrs["target"] == 1
    rates = [{"api": api, "functions": count, "vulnerable_functions": faulty[api], "vulnerable_ratio": faulty[api] / count} for api, count in apis.items() if count >= 50]
    rates.sort(key=lambda row: (-row["vulnerable_ratio"], -row["functions"], row["api"]))
    return {"nodes_by_kind": dict(Counter(attrs["kind"] for _, attrs in graph.nodes(data=True))), "edges_by_rel": dict(Counter(attrs["rel"] for _, _, attrs in graph.edges(data=True))), "top_api_by_functions": apis.most_common(20), "top_api_by_vulnerable_ratio": rates[:20], "top_cwe": cwes.most_common(20), "top_projects_by_vulnerable_functions": projects.most_common(10)}


def build(interim_dir, raw_dir, output_dir, stats_path, min_api_functions, dataset="r2vul"):
    if min_api_functions < 1:
        raise ValueError("min_api_functions must be positive")
    graph = nx.MultiDiGraph()
    graph.graph["dataset"] = dataset
    for split in ("train", "valid", "test"):
        for item in read_jsonl(Path(interim_dir) / f"{split}.jsonl"):
            meta = item["meta"]
            key = f"function:{item['id']}"
            if key in graph:
                raise ValueError(f"Duplicate function id: {item['id']}")
            project, commit = str(meta.get("repo") or ""), str(meta.get("commit_URL") or "")
            cwes = list(dict.fromkeys(meta.get("cwe_id") or []))
            graph.add_node(key, kind="function", id=item["id"], target=int(item["target"]), split=item["split"], project=project, cwe="|".join(cwes))
            lang, cve = item["lang"], str(meta.get("cve_id") or "")
            graph.nodes[key].update(lang=lang, cve=cve, commit_URL=commit, parent_commit_sha=meta.get("parent_commit_sha") or "")
            graph.add_node(f"language:{lang}", kind="language")
            graph.add_edge(key, f"language:{lang}", rel="in_language")
            if cve:
                graph.add_node(f"cve:{cve}", kind="cve", label_derived=True)
                graph.add_edge(key, f"cve:{cve}", rel="has_cve", label_derived=True)
            parent = str(meta.get("parent_commit_sha") or "")
            if parent:
                parent_key = f"commit:{project}@{parent}"
                graph.add_node(parent_key, kind="commit")
                graph.add_edge(key, parent_key, rel="from_parent_commit")
            if project:
                graph.add_node(f"project:{project}", kind="project")
                graph.add_edge(key, f"project:{project}", rel="in_project")
            if commit:
                graph.add_node(f"commit:{commit}", kind="commit")
                graph.add_edge(key, f"commit:{commit}", rel="from_commit")
                if project and not graph.has_edge(f"commit:{commit}", f"project:{project}"):
                    graph.add_edge(f"commit:{commit}", f"project:{project}", rel="in_project")
            for cwe in cwes:
                graph.add_node(f"cwe:{cwe}", kind="cwe", label_derived=True)
                graph.add_edge(key, f"cwe:{cwe}", rel="has_cwe", label_derived=True)
    frequencies, seen = Counter(), set()
    for item in batches(raw_dir):
        key = f"function:{item['id']}"
        if key not in graph:
            continue
        if key in seen:
            raise ValueError(f"Duplicate CPG id: {item['id']}")
        seen.add(key)
        frequencies.update(api_calls(item).keys())
    retained = {api for api, count in frequencies.items() if count >= min_api_functions}
    for item in batches(raw_dir):
        key = f"function:{item['id']}"
        if key not in graph:
            continue
        for api, count in api_calls(item).items():
            if api in retained:
                graph.add_node(f"api:{api}", kind="api")
                graph.add_edge(key, f"api:{api}", rel="calls", count=count)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{dataset}_kg.pkl").open("wb") as stream:
        pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
    fields = ["node_key", "kind", "id", "target", "split", "project", "cwe", "label_derived"]
    fields += ["lang", "cve", "commit_URL", "parent_commit_sha"]
    with (output_dir / "nodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, attrs in graph.nodes(data=True):
            writer.writerow({"node_key": key, **attrs})
    with (output_dir / "edges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["src", "dst", "rel", "count", "label_derived"])
        writer.writeheader()
        for src, dst, attrs in graph.edges(data=True):
            writer.writerow({"src": src, "dst": dst, "rel": attrs["rel"], "count": attrs.get("count", 1), "label_derived": attrs.get("label_derived", False)})
    summary = statistics(graph)
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("r2vul",), default="r2vul")
    parser.add_argument("--interim-dir", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--min-api-functions", type=int, default=20)
    args = parser.parse_args()
    build(args.interim_dir or f"data/interim/{args.dataset}", args.raw_dir or f"data/graphs/{args.dataset}/raw", args.output_dir or (f"data/kg/{args.dataset}"), args.stats_path or f"results/tables/kg_{args.dataset}.json", args.min_api_functions, args.dataset)
