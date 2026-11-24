# Nguyễn Tâm Bình - 2324801030003
import json
import os
import threading
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOC = "Qwen/Qwen2.5-Coder-1.5B"
MAX_LEN = 2048
LEN_DIEM_DONG = int(os.environ.get("LEN_DIEM_DONG", "1024"))
LOP_CUOI = 4
ADAPTER = Path(os.environ.get("ADAPTER", "/work/results/lora/r2vul5/best_adapter.pt"))
NGUONG = json.loads((Path(__file__).parent / "nguong.json").read_text(encoding="utf-8"))


class MoHinh:
    def __init__(self):
        torch.set_num_threads(int(os.environ.get("SO_LUONG", "4")))
        self.tok = AutoTokenizer.from_pretrained(GOC)
        goc = AutoModelForSequenceClassification.from_pretrained(GOC, num_labels=2, torch_dtype=torch.float32, attn_implementation="eager")
        goc.config.pad_token_id = self.tok.pad_token_id
        cau_hinh = LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(goc, cau_hinh)
        luu = torch.load(ADAPTER, map_location="cpu")
        thieu = [k for k in luu if k not in self.model.state_dict()]
        if thieu:
            raise RuntimeError(f"{len(thieu)} khóa adapter không khớp, ví dụ {thieu[0]}")
        self.model.load_state_dict(luu, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.khoa = threading.Lock()

    def _xac_suat(self, mh):
        with torch.no_grad():
            self.model.base_model.model.config._attn_implementation = "sdpa"
            try:
                logit = self.model(input_ids=mh["input_ids"], attention_mask=mh["attention_mask"]).logits[0]
            finally:
                self.model.base_model.model.config._attn_implementation = "eager"
        return torch.softmax(logit, -1)[1].item()

    def cham(self, ma, ngon_ngu):

        with self.khoa:
            mh = self.tok(ma, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True, return_tensors="pt")
            vt_day_du = mh.pop("offset_mapping")[0].tolist()
            n = mh["input_ids"].shape[1]
            xs = self._xac_suat(mh) if n > LEN_DIEM_DONG else None
            ids, att = mh["input_ids"][:, :LEN_DIEM_DONG], mh["attention_mask"][:, :LEN_DIEM_DONG]
            vt = vt_day_du[:LEN_DIEM_DONG]
            nhung = self.model.get_input_embeddings()(ids).detach().requires_grad_(True)
            ra = self.model(inputs_embeds=nhung, attention_mask=att, output_attentions=True)
            logit = ra.logits[0]
            if xs is None:
                xs = torch.softmax(logit, -1)[1].item()
            (logit[1] - logit[0]).backward()
            chu_y = torch.stack([a[0, :, -1, :] for a in ra.attentions[-LOP_CUOI:]]).mean((0, 1)).detach()
            dao_ham = (nhung.grad[0] * nhung[0]).sum(-1).abs().detach()
            del ra, nhung
        chu_y[0], dao_ham[0] = 0, 0
        diem_token = 0.5 * _chuan(chu_y) + 0.5 * _chuan(dao_ham)
        dau_dong = [0] + [i + 1 for i, c in enumerate(ma) if c == "\n"]
        cong, dem = {}, {}
        for (a, b), d in zip(vt, diem_token.tolist()):
            if b <= a:
                continue
            dong = _tim_dong(dau_dong, a)
            cong[dong] = cong.get(dong, 0.0) + d
            dem[dong] = dem.get(dong, 0) + 1
        dong_ma = ma.split("\n")
        dong_dau = next((i for i, x in enumerate(dong_ma) if x.strip()), 0)
        diem = {}
        for dong, s in cong.items():
            noi = dong_ma[dong].strip() if dong < len(dong_ma) else ""
            if len(noi) <= 1 or noi.startswith(("//", "/*", "*", "#")) or (dong == dong_dau and len(dong_ma) > 3):
                continue
            diem[dong] = s / dem[dong] ** 0.5
        if diem:
            lon = max(diem.values())
            diem = {k: round(v / lon, 4) for k, v in diem.items()}
        het = len(ma.rstrip())
        cat = n >= MAX_LEN and vt_day_du[-1][1] < het
        cat_diem = len(vt) >= LEN_DIEM_DONG and vt[-1][1] < het
        nguong = NGUONG[ngon_ngu]["nguong"]
        return {"xac_suat": round(xs, 4), "nguong": nguong, "co_loi": xs >= nguong, "diem_dong": diem, "bi_cat": cat,
                "dong_cuoi_duoc_cham": _tim_dong(dau_dong, vt_day_du[-1][0]) if cat else None,
                "dong_cuoi_cham_diem": _tim_dong(dau_dong, vt[-1][0]) if cat_diem else None}


def _chuan(x):
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def _tim_dong(dau_dong, vi_tri):
    lo, hi = 0, len(dau_dong) - 1
    while lo < hi:
        giua = (lo + hi + 1) // 2
        if dau_dong[giua] <= vi_tri:
            lo = giua
        else:
            hi = giua - 1
    return lo
