# Nguyễn Tâm Bình - 2324801030003
import argparse
import pickle
import random
from collections import Counter
from pathlib import Path

import networkx as nx


def node_attributes(key, attrs):
    return {"kind": attrs["kind"], "target": attrs.get("target", -1), "cwe": attrs.get("cwe", "").split("|")[0], "project": attrs.get("project", ""), "label": key.split(":", 1)[1][:60], **({"label_derived": True} if attrs.get("label_derived") else {}), **({"lang": attrs["lang"], "cve": attrs["cve"]} if "lang" in attrs else {})}


def overview(graph):
    cwes = Counter({key: graph.in_degree(key) for key, attrs in graph.nodes(data=True) if attrs["kind"] == "cwe"})
    top_cwes = set(key for key, _ in cwes.most_common(20))
    vulnerable, safe = [], []
    for key, attrs in graph.nodes(data=True):
        if attrs["kind"] == "function":
            if attrs["target"] == 0:
                safe.append(key)
            elif any(dst in top_cwes for dst in graph.successors(key)):
                vulnerable.append(key)
    rng = random.Random(42)
    selected = rng.sample(vulnerable, min(1500, len(vulnerable))) + rng.sample(safe, min(1500, len(safe)))
    apis = Counter()
    for key in selected:
        apis.update(dst for dst in graph.successors(key) if graph.nodes[dst]["kind"] == "api")
    top_apis = set(key for key, _ in apis.most_common(100))
    while True:
        neighbors = {dst for key in selected for dst in graph.successors(key) if graph.nodes[dst]["kind"] == "project" or dst in top_cwes or dst in top_apis or graph.nodes[dst]["kind"] in ("language", "cve", "commit")}
        if len(selected) + len(neighbors) <= 5000:
            break
        selected.pop()
    keep = set(selected) | neighbors
    result = nx.DiGraph()
    for key in sorted(keep):
        result.add_node(key, **node_attributes(key, graph.nodes[key]))
    for src in sorted(keep):
        for _, dst, attrs in graph.out_edges(src, data=True):
            if dst not in keep:
                continue
            if result.has_edge(src, dst):
                result[src][dst]["weight"] += attrs.get("count", 1)
            else:
                result.add_edge(src, dst, weight=attrs.get("count", 1), rel=attrs["rel"], label_derived=attrs.get("label_derived", False))
    return result


def cwe_api(graph):
    counts = Counter()
    for key, attrs in graph.nodes(data=True):
        if attrs["kind"] != "function" or attrs["target"] != 1:
            continue
        neighbors = list(graph.successors(key))
        cwes = [dst for dst in neighbors if graph.nodes[dst]["kind"] == "cwe"]
        apis = [dst for dst in neighbors if graph.nodes[dst]["kind"] == "api"]
        counts.update((cwe, api) for cwe in cwes for api in apis)
    result = nx.Graph()
    for (cwe, api), count in sorted(counts.items()):
        if count >= 5:
            result.add_node(cwe, **node_attributes(cwe, graph.nodes[cwe]), bipartite=0)
            result.add_node(api, **node_attributes(api, graph.nodes[api]), bipartite=1)
            result.add_edge(cwe, api, weight=count, label_derived=True)
    return result




def export(kg_path, output_dir):
    with Path(kg_path).open("rb") as stream:
        graph = pickle.load(stream)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(overview(graph), output_dir / "kg_overview.gexf")
    nx.write_gexf(cwe_api(graph), output_dir / "kg_cwe_api.gexf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("r2vul",), default="r2vul")
    parser.add_argument("--kg-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    kg_dir = f"data/kg/{args.dataset}"
    export(args.kg_path or f"{kg_dir}/{args.dataset}_kg.pkl", args.output_dir or f"results/gephi/{args.dataset}")
