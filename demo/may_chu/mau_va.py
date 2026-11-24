# Nguyễn Tâm Bình - 2324801030003
import difflib
import json
import os
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TEP = Path(os.environ.get("DU_LIEU_MAU", "/work/data/interim/r2vul/train.jsonl"))
SO_DONG_DIFF = 60


def _ten(ma):
    for dong in ma.split("\n"):
        m = re.search(r"([A-Za-z_]\w*)\s*\(", dong)
        if m and m.group(1) not in ("if", "for", "while", "switch", "return", "sizeof"):
            return m.group(1)
    return ""


class MauVa:
    """Chỉ mục các cặp hàm lỗi → bản vá chính thức trong tập train của R2Vul, tra theo độ giống TF-IDF."""

    def __init__(self):
        nhom = {}
        with TEP.open(encoding="utf-8") as f:
            for dong in f:
                x = json.loads(dong)
                m = x["meta"]
                if m.get("origin") not in ("vuln", "fixed"):
                    continue
                nhom.setdefault((m.get("cve_id"), m.get("file"), _ten(x["func"])), {})[m["origin"]] = x
        self.cap = {}
        for v in nhom.values():
            if "vuln" in v and "fixed" in v and v["vuln"]["func"] != v["fixed"]["func"]:
                a, b = v["vuln"], v["fixed"]
                self.cap.setdefault(a["lang"], []).append((a["func"], b["func"], [c for c in a["meta"].get("cwe_id") or [] if c.startswith("CWE-")], a["meta"].get("cve_id")))
        self.vec, self.mt = {}, {}
        for nn, ds in self.cap.items():
            self.vec[nn] = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w+", sublinear_tf=True)
            self.mt[nn] = self.vec[nn].fit_transform([a for a, _, _, _ in ds])

    def tim(self, ma, ngon_ngu, k=2, nguong=0.08):
        if ngon_ngu not in self.vec:
            return []
        diem = linear_kernel(self.vec[ngon_ngu].transform([ma]), self.mt[ngon_ngu]).ravel()
        kq = []
        for i in diem.argsort()[::-1][:k]:
            if diem[i] < nguong:
                break
            loi, va, cwe, cve = self.cap[ngon_ngu][i]
            diff = [d for d in difflib.unified_diff(loi.split("\n"), va.split("\n"), lineterm="", n=2) if not d.startswith(("---", "+++"))]
            kq.append({"diem": round(float(diem[i]), 3), "cwe": cwe, "cve": cve, "diff": "\n".join(diff[:SO_DONG_DIFF])})
        return kq
