# Nguyễn Tâm Bình - 2324801030003
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from lxml import etree

plt.rcParams["font.family"] = "Times New Roman"
NS = {"g": "http://www.gexf.net/1.2draft", "viz": "http://www.gexf.net/1.2draft/viz"}
MAU_NN = {"C": "#E63946", "C#": "#9B5DE5", "Java": "#F77F00", "JavaScript": "#FFBE0B", "Python": "#06D6A0"}


def doc_gexf(tep):
    goc = etree.parse(str(tep)).getroot()
    G = nx.Graph()
    for n in goc.iterfind(".//g:node", NS):
        c = n.find("viz:color", NS)
        s = n.find("viz:size", NS)
        loai = n.find(".//g:attvalue", NS)
        G.add_node(n.get("id"), label=n.get("label"), mau="#%02X%02X%02X" % tuple(int(c.get(k)) for k in "rgb") if c is not None else "#999999",
                   size=float(s.get("value")) if s is not None else 5.0, kind=loai.get("value") if loai is not None else "")
    for e in goc.iterfind(".//g:edge", NS):
        G.add_edge(e.get("source"), e.get("target"), w=float(e.get("weight", 1)))
    return G


def ve(ax, G, so_nhan, tieu_de, seed):
    pos = nx.forceatlas2_layout(G, max_iter=800, scaling_ratio=4.0, distributed_action=True, weight="w", seed=seed)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.35, alpha=0.25, edge_color=[G.nodes[u]["mau"] for u, _ in G.edges()])
    size = [min(900, 18 + 6 * G.nodes[n]["size"]) for n in G]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=size, node_color=[G.nodes[n]["mau"] for n in G], linewidths=0.4, edgecolors="white")
    deg = dict(G.degree())
    chon = [n for n in G if G.nodes[n]["kind"] == "language"] + sorted((n for n in G if G.nodes[n]["kind"] != "language"), key=lambda n: -deg[n])[:so_nhan]
    for n in chon:
        nn = G.nodes[n]["kind"] == "language"
        ax.text(*pos[n], G.nodes[n]["label"], fontsize=13 if nn else 8.5, fontweight="bold" if nn else "normal", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_title(tieu_de, fontsize=14)
    ax.axis("off")


def main(thu_muc, ra):
    G1 = doc_gexf(Path(thu_muc) / "kg_tong_quan_gon_mau.gexf")
    G2 = doc_gexf(Path(thu_muc) / "kg_cwe_api_mau.gexf")
    G2 = G2.subgraph(max(nx.connected_components(G2), key=len)).copy()
    print("toan canh", G1.number_of_nodes(), G1.number_of_edges(), "| cwe-api", G2.number_of_nodes(), G2.number_of_edges())
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.2))
    ve(axs[0], G1, 30, "(a) Toàn cảnh: ngôn ngữ, kho mã, CWE và API", 42)
    ve(axs[1], G2, 22, "(b) Mạng CWE – API", 42)
    tay = [plt.Line2D([], [], marker="o", ls="", color=m, markersize=10, label=k) for k, m in MAU_NN.items()]
    tay += [plt.Line2D([], [], marker="o", ls="", color="#FF006E", markersize=10, label="CWE (hình a)"), plt.Line2D([], [], marker="o", ls="", color="#3A86FF", markersize=10, label="API (hình a)")]
    fig.legend(handles=tay, loc="lower center", ncol=7, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(ra, dpi=300)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
