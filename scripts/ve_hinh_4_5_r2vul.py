# Nguyễn Tâm Bình - 2324801030003
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ve_hinh_chuong4_r2vul import MAU, NGON_NGU, ROOT, cot_nhom

paper = json.loads((ROOT / "r2vul" / "so_lieu_bai_bao.json").read_text(encoding="utf-8"))
khoa = lambda lang: "macro" if lang == "macro_5" else lang
ten = ["CodeBERT CLS", "Claude-Sonnet-4", "Claude-Opus-4", "GPT-4o", "GPT-4.1"]
gia_tri = [[paper[khoa(l)]["CodeBERT CLS"][2] for l, _ in NGON_NGU]] + [[paper["llm_f1"][n][khoa(l)] for l, _ in NGON_NGU] for n in ten[1:]]
nhom = {"Nhóm em · L7 LoRA (toàn bộ, p ≥ 0,5)": [72.83, 77.27, 82.10, 76.06, 68.90, 75.43], "Nhóm em · L13 (có đồ thị, ngưỡng valid)": [76.94, 84.00, 84.30, 75.11, 77.97, 79.66]}
for k, v in zip(ten, gia_tri):
    print(k, v)
cot_nhom(gia_tri + list(nhom.values()), ten + list(nhom), [*MAU[:5], MAU[2], MAU[0]], "r2vul_4_5_so_bai_bao.png", "Số bài R2Vul: bảng công bố · Nhóm em L7: toàn bộ 1.838 hàm, xác suất 0,5\nNhóm L13: 1.784 hàm có đồ thị, ngưỡng valid · Phạm vi test/ngưỡng khác nhau", highlight=True)
