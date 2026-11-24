# Nguyễn Tâm Bình - 2324801030003
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEN = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
_khoa = threading.Lock()
_mh = {}


def _nap():
    if not _mh:
        _mh["tok"] = AutoTokenizer.from_pretrained(TEN)
        _mh["model"] = AutoModelForCausalLM.from_pretrained(TEN, torch_dtype=torch.bfloat16).eval()
    return _mh["tok"], _mh["model"]


def viet_nhap(ma, dong_bat_dau, dong_danh_dau, xac_suat, ngon_ngu):

    danh_so = "\n".join(f"{dong_bat_dau + i:4d} | {x}" for i, x in enumerate(ma.split("\n")))
    hoi = (f"Hàm {ngon_ngu} dưới đây có xác suất chứa lỗ hổng bảo mật là {xac_suat:.2f}. "
           f"Với từng dòng {', '.join(map(str, dong_danh_dau))}, hãy nói ngắn gọn dòng đó có nguy hiểm không, vì sao, thuộc loại lỗi CWE nào và cách sửa. "
           f"Sau đó tóm tắt hàm trong một câu.\n\n{danh_so}")
    with _khoa:
        tok, model = _nap()
        dau_vao = tok.apply_chat_template([{"role": "system", "content": f"Bạn là chuyên gia bảo mật mã nguồn {ngon_ngu}, trả lời bằng tiếng Việt."}, {"role": "user", "content": hoi}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            ra = model.generate(**dau_vao, max_new_tokens=320, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(ra[0, dau_vao["input_ids"].shape[1]:], skip_special_tokens=True).strip()
