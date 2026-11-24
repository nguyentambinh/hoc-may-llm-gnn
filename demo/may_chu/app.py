# Nguyễn Tâm Bình - 2324801030003
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from giai_thich import giai_thich, kiem_tra_sua
from mau_va import MauVa
from mo_hinh import NGUONG, MoHinh
from qwen_instruct import viet_nhap
from tach_ham import TEN_NGON_NGU, doan_ngon_ngu, tach_ham

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("app")
GOC = Path(__file__).parent
WEB = GOC.parent / "web" / "build" / "web"
TOI_DA_KY_TU, TOI_DA_HAM, SO_DONG_DANH_DAU = 60000, 20, 3
trang_thai = {}
SO_LUOT, KHOANG_GIAY = 12, 600
luot_goi = {}


def _kiem_luot(ip):
    bay_gio = time.time()
    ds = [t for t in luot_goi.get(ip, []) if bay_gio - t < KHOANG_GIAY]
    if len(ds) >= SO_LUOT:
        raise HTTPException(429, "Bạn đã phân tích nhiều lần liên tiếp, vui lòng đợi vài phút rồi thử lại.")
    luot_goi[ip] = ds + [bay_gio]


@asynccontextmanager
async def khoi_dong(app):
    t = time.time()
    trang_thai["mo_hinh"] = MoHinh()
    trang_thai["mau_va"] = MauVa()
    log.info("nap mo hinh xong sau %.0f giay", time.time() - t)
    yield


app = FastAPI(lifespan=khoi_dong, docs_url=None, redoc_url=None)


class YeuCau(BaseModel):
    ma: str = Field(min_length=1)
    ngon_ngu: str = "tu_dong"
    giai_thich: bool = True
    ma_truoc: str | None = None


@app.get("/api/suc-khoe")
def suc_khoe():
    return {"san_sang": "mo_hinh" in trang_thai}


@app.get("/api/vi-du")
def vi_du():
    tep = GOC / "vi_du.json"
    return json.loads(tep.read_text(encoding="utf-8")) if tep.exists() else []


@app.post("/api/phan-tich")
def phan_tich(yc: YeuCau, req: Request):
    ip = req.headers.get("cf-connecting-ip") or (req.client.host if req.client else "")
    if ip != "127.0.0.1" or req.headers.get("cf-connecting-ip"):
        _kiem_luot(ip)
    if "mo_hinh" not in trang_thai:
        raise HTTPException(503, "Mô hình đang khởi động, vui lòng thử lại sau ít giây.")
    if yc.ngon_ngu not in (*NGUONG, "tu_dong"):
        raise HTTPException(400, "Ngôn ngữ chưa được hỗ trợ; hãy chọn C, C#, Java, JavaScript hoặc Python.")
    if len(yc.ma) > TOI_DA_KY_TU:
        raise HTTPException(400, f"Đoạn mã dài quá {TOI_DA_KY_TU:,} ký tự, hãy tách bớt.".replace(",", "."))
    nn = doan_ngon_ngu(yc.ma) if yc.ngon_ngu == "tu_dong" else yc.ngon_ngu
    ds = tach_ham(yc.ma, nn)
    if len(ds) > TOI_DA_HAM:
        raise HTTPException(400, f"Đoạn mã có {len(ds)} hàm; mỗi lần chỉ phân tích tối đa {TOI_DA_HAM} hàm.")
    m = trang_thai["mo_hinh"]
    t = time.time()
    kq = []
    for h in ds:
        c = m.cham(h["ma"], nn)
        diem = {h["dong_bat_dau"] + d: s for d, s in c["diem_dong"].items()}
        danh_dau = sorted(diem, key=lambda d: -diem[d])[:SO_DONG_DANH_DAU]
        kq.append({**h, "xac_suat": c["xac_suat"], "nguong": c["nguong"], "co_loi": c["co_loi"], "bi_cat": c["bi_cat"],
                   "dong_cuoi_duoc_cham": None if c["dong_cuoi_duoc_cham"] is None else h["dong_bat_dau"] + c["dong_cuoi_duoc_cham"],
                   "dong_cuoi_cham_diem": None if c["dong_cuoi_cham_diem"] is None else h["dong_bat_dau"] + c["dong_cuoi_cham_diem"],
                   "diem_dong": {str(k): v for k, v in sorted(diem.items())}, "dong_danh_dau": danh_dau, "giai_thich": None})
    kiem = None
    if yc.ma_truoc and len(yc.ma_truoc) <= TOI_DA_KY_TU and yc.ma_truoc != yc.ma:
        kiem = trang_thai.setdefault("pool", ThreadPoolExecutor(max_workers=2)).submit(kiem_tra_sua, yc.ma_truoc, yc.ma, TEN_NGON_NGU[nn])
    if yc.giai_thich:
        can = [x for x in kq if x["co_loi"]] or sorted(kq, key=lambda x: -x["xac_suat"])[:1]
        nhap = [viet_nhap(x["ma"], x["dong_bat_dau"], x["dong_danh_dau"], x["xac_suat"], TEN_NGON_NGU[nn]) for x in can]
        with ThreadPoolExecutor(max_workers=4) as pool:
            ra = pool.map(lambda xn: giai_thich(xn[0]["ma"], xn[0]["dong_bat_dau"], xn[0]["dong_danh_dau"], xn[0]["xac_suat"], xn[0]["co_loi"], xn[1], TEN_NGON_NGU[nn], trang_thai["mau_va"].tim(xn[0]["ma"], nn)), zip(can, nhap))
            for x, g in zip(can, ra):
                x["giai_thich"] = g
                x["giai_thich_loi"] = g is None
    log.info("phan tich %d ham, ngon ngu %s, %.1f giay", len(kq), nn, time.time() - t)
    return {"ham": kq, "kiem_tra_sua": kiem.result() if kiem else None, "ngon_ngu": nn, "ten_ngon_ngu": TEN_NGON_NGU[nn], "thoi_gian": round(time.time() - t, 1)}


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{duong_dan:path}")
    def web(duong_dan: str):
        tep = (WEB / duong_dan).resolve()
        if duong_dan and tep.is_file() and WEB.resolve() in tep.parents:
            return FileResponse(tep)
        return FileResponse(WEB / "index.html")
