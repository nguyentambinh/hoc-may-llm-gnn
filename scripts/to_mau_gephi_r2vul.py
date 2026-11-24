# Nguyễn Tâm Bình - 2324801030003
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
KG = ROOT / 'data/kg/r2vul'
OUT = ROOT / 'results/gephi/r2vul'
LANG = {'c': '#E63946', 'c_sharp': '#9B5DE5', 'java': '#F77F00', 'javascript': '#FFBE0B', 'python': '#06D6A0'}
TEN = {'c': 'C', 'c_sharp': 'C#', 'java': 'Java', 'javascript': 'JavaScript', 'python': 'Python'}
CWE_MAU, API_MAU = '#FF006E', '#3A86FF'
BO_API = {'sizeof', '_', 'len', 'str', 'assert', 'isinstance', 'print', 'format', 'get', 'append', 'join', 'int', 'list', 'dict', 'range', 'super'}

def mau(hexcode, a=1.0):
    return {'r': int(hexcode[1:3], 16), 'g': int(hexcode[3:5], 16), 'b': int(hexcode[5:7], 16), 'a': a}

def viz(hexcode, size, pos):
    return {'color': mau(hexcode), 'size': float(size), 'position': {'x': float(pos[0]), 'y': float(pos[1]), 'z': 0.0}}

def doc_kg():
    funcs = {}
    with open(KG / 'nodes.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['kind'] == 'function':
                funcs[r['node_key']] = (r['lang'], r['target'] == '1', r['project'])
    cwe, api = defaultdict(set), defaultdict(set)
    with open(KG / 'edges.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['rel'] == 'has_cwe' and r['src'] in funcs:
                cwe[r['src']].add(r['dst'].split(':', 1)[1])
            elif r['rel'] == 'calls' and r['src'] in funcs:
                api[r['src']].add(r['dst'].split(':', 1)[1])
    return funcs, cwe, api

def ngon_ngu_chinh(counter):
    return counter.most_common(1)[0][0]

def tong_quan(funcs, cwe, api):
    loi = [k for k, (_, t, _) in funcs.items() if t]
    du_an = Counter(funcs[k][2] for k in loi)
    ma_cwe = Counter(c for k in loi for c in cwe[k] if not c.startswith('NVD-'))
    ham_api = Counter(a for k in loi for a in api[k] if a not in BO_API)
    top_da = [p for p, _ in du_an.most_common(25)]
    top_cwe = [c for c, _ in ma_cwe.most_common(18)]
    top_api = [a for a, _ in ham_api.most_common(40)]
    lang_da = {p: ngon_ngu_chinh(Counter(funcs[k][0] for k in loi if funcs[k][2] == p)) for p in top_da}
    g = nx.Graph()
    for l in LANG:
        g.add_node(f'lang:{l}', label=TEN[l], kind='language', mau=LANG[l], size=40)
    for p in top_da:
        g.add_node(f'project:{p}', label=p.split('/')[-1], kind='project', mau=LANG[lang_da[p]], size=10 + 1.2 * math.sqrt(du_an[p]))
        g.add_edge(f'project:{p}', f'lang:{lang_da[p]}', weight=du_an[p], mau=LANG[lang_da[p]])
    for c in top_cwe:
        g.add_node(f'cwe:{c}', label=c, kind='cwe', mau=CWE_MAU, size=10 + 1.2 * math.sqrt(ma_cwe[c]))
    for a in top_api:
        g.add_node(f'api:{a}', label=a, kind='api', mau=API_MAU, size=6 + math.sqrt(ham_api[a]))
    chung = Counter()
    for k in loi:
        p = funcs[k][2]
        cs = [c for c in cwe[k] if c in top_cwe]
        as_ = [a for a in api[k] if a in top_api]
        if p in lang_da:
            chung.update((f'project:{p}', f'cwe:{c}') for c in cs)
            chung.update((f'project:{p}', f'api:{a}') for a in as_)
        chung.update((f'cwe:{c}', f'api:{a}') for c in cs for a in as_)
    for (u, v), n in chung.items():
        if n >= 3:
            g.add_edge(u, v, weight=n, mau=g.nodes[u]['mau'] if g.nodes[u]['kind'] == 'project' else CWE_MAU)
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    return g

def cwe_api(funcs, cwe, api):
    g = nx.read_gexf(OUT / 'kg_cwe_api.gexf')
    lang = defaultdict(Counter)
    for k, (l, t, _) in funcs.items():
        if t:
            for c in cwe[k]:
                lang[f'cwe:{c}'][l] += 1
            for a in api[k]:
                lang[f'api:{a}'][l] += 1
    h = nx.Graph()
    for n, d in g.nodes(data=True):
        l = ngon_ngu_chinh(lang[n]) if lang[n] else 'c'
        h.add_node(n, label=d.get('label', n.split(':', 1)[1]), kind=d['kind'], lang=TEN[l], mau=LANG[l], size=(8 if d['kind'] == 'cwe' else 4) + 2 * math.sqrt(g.degree(n)))
    for u, v, d in g.edges(data=True):
        h.add_edge(u, v, mau=h.nodes[u]['mau'] if h.nodes[u]['kind'] == 'cwe' else h.nodes[v]['mau'])
    return h

def ghi(g, ten, k):
    pos = nx.spring_layout(g, seed=42, k=k / math.sqrt(len(g)), iterations=300, weight=None, scale=1000)
    out = nx.Graph()
    for n, d in g.nodes(data=True):
        out.add_node(n, label=d['label'], kind=d['kind'], viz=viz(d['mau'], d['size'], pos[n]))
    for u, v, d in g.edges(data=True):
        out.add_edge(u, v, weight=float(d.get('weight', 1)), viz={'color': mau(d['mau'], 0.55)})
    nx.write_gexf(out, OUT / ten)
    print(f'{ten}: {out.number_of_nodes()} nút, {out.number_of_edges()} cạnh', Counter(d['kind'] for _, d in out.nodes(data=True)))

def main():
    funcs, cwe, api = doc_kg()
    ghi(tong_quan(funcs, cwe, api), 'kg_tong_quan_gon_mau.gexf', 2.5)
    ghi(cwe_api(funcs, cwe, api), 'kg_cwe_api_mau.gexf', 2.0)

if __name__ == '__main__':
    main()
