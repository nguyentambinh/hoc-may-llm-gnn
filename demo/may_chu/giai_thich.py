# Nguyễn Tâm Bình - 2324801030003
import json
import os
import logging

import anthropic

MO_HINH = os.environ.get("MO_HINH_GIAI_THICH", "claude-sonnet-5-5")
log = logging.getLogger("giai_thich")
client = anthropic.Anthropic(timeout=90.0, max_retries=1)

HUONG_DAN = """Bạn hiệu đính bản nháp giải thích do mô hình Qwen2.5-Coder-0.5B-Instruct viết cho một hàm {ngon_ngu} mà hệ thống phát hiện lỗ hổng đã chấm xác suất và đánh dấu các dòng nó chú ý nhiều nhất. Đầu ra bằng tiếng Việt, ngắn gọn, cho giảng viên và sinh viên đọc trên màn chiếu. Bản nháp có thể viết bằng ngôn ngữ khác (tiếng Anh, tiếng Trung...) hoặc sai kiến thức; dù vậy mọi trường đầu ra (tom_tat, ly_do, goi_y_sua) phải viết hoàn toàn bằng tiếng Việt, chỉ giữ nguyên tên biến, tên hàm, đoạn mã và mã CWE.

Hiệu đính nghĩa là: giữ những ý đúng của bản nháp, sửa chỗ sai về kỹ thuật, bổ sung chỗ bản nháp bỏ sót, viết lại câu cho rõ. Với từng dòng được đánh dấu: nói rõ dòng đó có đáng ngờ thật không. Nếu đáng ngờ, nêu lý do cụ thể dựa trên chính mã (biến nào, kích thước nào, điều kiện nào thiếu), loại lỗi CWE gần nhất (dạng "CWE-787: Ghi ngoài vùng nhớ") và cách sửa trong một câu. Nếu dòng đó thực ra an toàn, ghi mức "an toàn" và giải thích vì sao hệ thống có thể chú ý tới nó; không cố tìm lỗi không có.

Ngoài các dòng được đánh dấu, nếu thấy tối đa 2 dòng khác có lỗ hổng rõ ràng thì thêm vào với nguon = "bo_sung". Không bịa CWE khi không chắc; khi đó để cwe là chuỗi rỗng. Tóm tắt một hoặc hai câu về hàm. Trường ma_va là TOÀN BỘ hàm sau khi vá mọi lỗ hổng mức "cao" và "trung_binh" trong một lần (sửa đồng thời mọi chỗ liên quan, kể cả dòng không được đánh dấu), theo cách sửa chuẩn OWASP/SEI CERT; nếu có bản vá thật của hàm tương tự được cung cấp thì ưu tiên làm theo đúng kiểu vá đó khi phù hợp. Giữ nguyên tên hàm, tham số, kiểu trả về, thụt lề và phong cách mã gốc, thay đổi tối thiểu, không thêm chú thích, hàm phải biên dịch được. Nếu hàm an toàn thì ma_va là chuỗi rỗng. Với dòng mức "cao" hoặc "trung_binh", trường ma_sua dùng cách sửa chuẩn được khuyến nghị rộng rãi (OWASP, SEI CERT) và gọn nhất: hàm thư viện an toàn có sẵn (snprintf/strncpy có giới hạn, câu lệnh SQL có tham số, yaml.safe_load, json thay cho pickle hoặc eval, escape HTML, tắt thực thể ngoài XML, kiểm tra biên và NULL ngay trước chỗ dùng), không viết lại cả hàm; đó là đoạn mã thay thế cho các dòng từ dong tới den_dong; nếu câu lệnh ở dòng dong kéo dài nhiều dòng (ví dụ lời gọi hay phép nối chuỗi xuống dòng) thì den_dong là dòng cuối của câu lệnh để thay trọn cả câu, còn câu một dòng thì den_dong bằng dong (ma_sua có thể gồm nhiều dòng nối bằng ký tự xuống dòng, ví dụ thêm câu kiểm tra hoặc cấu hình ngay sau dòng), giữ nguyên thụt lề của dòng gốc, cú pháp đúng của ngôn ngữ, không đổi tên hàm hay tham số, không thêm chú thích; khi ghép vào thay dòng gốc thì hàm phải còn biên dịch được. Dòng an toàn, mức thấp hoặc khi cách sửa nằm ở chỗ khác thì ma_sua là chuỗi rỗng. Không nhắc tới tên mô hình, nhà cung cấp hay việc bạn là trợ lý AI."""

LUOC_DO = {
    "type": "object",
    "properties": {
        "tom_tat": {"type": "string"},
        "ma_va": {"type": "string"},
        "dong": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dong": {"type": "integer"},
                    "den_dong": {"type": "integer"},
                    "nguon": {"type": "string", "enum": ["mo_hinh", "bo_sung"]},
                    "muc_do": {"type": "string", "enum": ["cao", "trung_binh", "thap", "an_toan"]},
                    "ly_do": {"type": "string"},
                    "cwe": {"type": "string"},
                    "goi_y_sua": {"type": "string"},
                    "ma_sua": {"type": "string"},
                },
                "required": ["dong", "den_dong", "nguon", "muc_do", "ly_do", "cwe", "goi_y_sua", "ma_sua"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tom_tat", "ma_va", "dong"],
    "additionalProperties": False,
}


def giai_thich(ma, dong_bat_dau, dong_danh_dau, xac_suat, co_loi, nhap, ngon_ngu, mau=None):
    """dong_danh_dau: số dòng tuyệt đối (tính từ 1). Trả về dict theo LUOC_DO hoặc None khi lỗi."""
    danh_so = "\n".join(f"{dong_bat_dau + i:4d} | {x}" for i, x in enumerate(ma.split("\n")))
    noi_dung = (f"Xác suất có lỗ hổng: {xac_suat:.2f} ({'vượt' if co_loi else 'dưới'} ngưỡng cảnh báo).\n"
                f"Các dòng được đánh dấu: {', '.join(map(str, dong_danh_dau))}.\n\nMã nguồn (số dòng ở bên trái):\n{danh_so}\n\nBản nháp của Qwen Instruct:\n{nhap or '(trống)'}")
    if mau:
        noi_dung += "\n\nBản vá thật của các hàm tương tự trong dữ liệu (dạng diff, chỉ để tham khảo cách vá):\n" + "\n\n".join(f"[{x['cve']}, {', '.join(x['cwe']) or 'không rõ CWE'}]\n{x['diff']}" for x in mau)
    try:
        r = client.beta.messages.create(
            model=MO_HINH,
            max_tokens=4000,
            system=HUONG_DAN.replace("{ngon_ngu}", ngon_ngu),
            messages=[{"role": "user", "content": noi_dung}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": LUOC_DO}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError:
        log.warning("bi gioi han luot goi")
        return None
    except anthropic.APIStatusError as e:
        log.warning("loi API %s", e.status_code)
        return None
    except anthropic.APIConnectionError:
        log.warning("khong ket noi duoc")
        return None
    if r.stop_reason == "refusal":
        log.warning("bi tu choi: %s", r.stop_details.category if r.stop_details else None)
        return None
    van_ban = next((b.text for b in r.content if b.type == "text"), None)
    if van_ban is None:
        return None
    kq = json.loads(van_ban)
    so_dong = set(range(dong_bat_dau, dong_bat_dau + len(ma.split("\n"))))
    kq["dong"] = [x for x in kq["dong"] if x["dong"] in so_dong]
    for x in kq["dong"]:
        x["den_dong"] = min(max(x.get("den_dong") or x["dong"], x["dong"]), max(so_dong), x["dong"] + 15)
    return kq


HUONG_DAN_SUA = """Bạn kiểm tra một bản sửa mã nguồn {ngon_ngu} nhằm vá lỗ hổng bảo mật. Bạn nhận mã trước khi sửa và sau khi sửa. Đánh giá bản sau đã xử lý lỗ hổng của bản trước chưa: "da_xu_ly" khi lỗ hổng chính đã được vá đúng cách và không sinh lỗi mới, "xu_ly_mot_phan" khi mới vá một phần hoặc còn lỗ hổng khác rõ ràng, "chua_xu_ly" khi bản sửa không giải quyết được vấn đề. nhan_xet: một hoặc hai câu tiếng Việt nói thay đổi nào đã vá lỗ hổng gì. con_lai: điều còn cần làm, chuỗi rỗng nếu không còn. Mọi trường viết bằng tiếng Việt, chỉ giữ nguyên tên biến, tên hàm, mã CWE. Không nhắc tới tên mô hình, nhà cung cấp hay việc bạn là trợ lý AI."""

LUOC_DO_SUA = {
    "type": "object",
    "properties": {
        "ket_luan": {"type": "string", "enum": ["da_xu_ly", "xu_ly_mot_phan", "chua_xu_ly"]},
        "nhan_xet": {"type": "string"},
        "con_lai": {"type": "string"},
    },
    "required": ["ket_luan", "nhan_xet", "con_lai"],
    "additionalProperties": False,
}


def kiem_tra_sua(truoc, sau, ngon_ngu):
    """So bản trước và sau khi sửa; trả về dict theo LUOC_DO_SUA hoặc None khi lỗi."""
    try:
        r = client.beta.messages.create(
            model=MO_HINH,
            max_tokens=1500,
            system=HUONG_DAN_SUA.replace("{ngon_ngu}", ngon_ngu),
            messages=[{"role": "user", "content": f"Mã trước khi sửa:\n{truoc[:20000]}\n\nMã sau khi sửa:\n{sau[:20000]}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": LUOC_DO_SUA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        log.warning("kiem tra sua loi: %s", type(e).__name__)
        return None
    if r.stop_reason == "refusal":
        return None
    van_ban = next((b.text for b in r.content if b.type == "text"), None)
    return json.loads(van_ban) if van_ban else None
