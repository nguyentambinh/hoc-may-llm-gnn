// Nguyễn Tâm Bình - 2324801030003
import 'dart:convert';

import 'package:http/http.dart' as http;

const _goc = String.fromEnvironment('API');

Uri _dc(String duongDan) => _goc.isEmpty ? Uri.base.resolve(duongDan) : Uri.parse('$_goc$duongDan');

class LoiPhanTich implements Exception {
  final String loiNhan;
  LoiPhanTich(this.loiNhan);
  @override
  String toString() => loiNhan;
}

class MucGiaiThich {
  final int dong, denDong;
  final String nguon, mucDo, lyDo, cwe, goiYSua, maSua;
  MucGiaiThich.tuJson(Map<String, dynamic> j) : dong = j['dong'], denDong = j['den_dong'] ?? j['dong'], nguon = j['nguon'], mucDo = j['muc_do'], lyDo = j['ly_do'], cwe = j['cwe'] ?? '', goiYSua = j['goi_y_sua'] ?? '', maSua = (j['ma_sua'] ?? '').toString().trimRight();
  bool get coMaSua => maSua.trim().isNotEmpty && mucDo != 'an_toan';
}

class KetQuaHam {
  final String ten, ma;
  final int dongBatDau;
  final double xacSuat, nguong;
  final bool coLoi, biCat, giaiThichLoi;
  final int? dongCuoiDuocCham, dongCuoiChamDiem;
  final Map<int, double> diemDong;
  final List<int> dongDanhDau;
  final String? tomTat;
  final String maVa;
  final List<MucGiaiThich> muc;

  KetQuaHam.tuJson(Map<String, dynamic> j)
      : ten = j['ten'],
        ma = j['ma'],
        dongBatDau = j['dong_bat_dau'],
        xacSuat = (j['xac_suat'] as num).toDouble(),
        nguong = (j['nguong'] as num).toDouble(),
        coLoi = j['co_loi'],
        biCat = j['bi_cat'] ?? false,
        giaiThichLoi = j['giai_thich_loi'] ?? false,
        dongCuoiDuocCham = j['dong_cuoi_duoc_cham'],
        dongCuoiChamDiem = j['dong_cuoi_cham_diem'],
        diemDong = (j['diem_dong'] as Map<String, dynamic>).map((k, v) => MapEntry(int.parse(k), (v as num).toDouble())),
        dongDanhDau = List<int>.from(j['dong_danh_dau']),
        tomTat = j['giai_thich']?['tom_tat'],
        maVa = (j['giai_thich']?['ma_va'] ?? '').toString().trimRight(),
        muc = [for (final m in (j['giai_thich']?['dong'] as List? ?? const [])) MucGiaiThich.tuJson(m)];

  bool get canXemLai => coLoi && tomTat != null && !muc.any((m) => m.mucDo == 'cao' || m.mucDo == 'trung_binh');

  int get dongKetThuc => dongBatDau + ma.split('\n').length - 1;
  int? soMuc(int dong) {
    final i = dongDanhDau.indexOf(dong);
    return i < 0 ? null : i + 1;
  }
}

class ViDu {
  final String ten, moTa, ma, ngonNgu;
  ViDu.tuJson(Map<String, dynamic> j) : ten = j['ten'], moTa = j['mo_ta'], ma = j['ma'], ngonNgu = j['ngon_ngu'];
}

Future<List<ViDu>> layViDu() async {
  try {
    final r = await http.get(_dc('api/vi-du'));
    if (r.statusCode != 200) return [];
    return [for (final x in jsonDecode(utf8.decode(r.bodyBytes)) as List) ViDu.tuJson(x)];
  } catch (_) {
    return [];
  }
}

class KiemTraSua {
  final String ketLuan, nhanXet, conLai;
  KiemTraSua.tuJson(Map<String, dynamic> j) : ketLuan = j['ket_luan'] ?? '', nhanXet = j['nhan_xet'] ?? '', conLai = j['con_lai'] ?? '';
}

class KetQuaPhanTich {
  final List<KetQuaHam> ham;
  final String tenNgonNgu;
  final KiemTraSua? kiemTraSua;
  KetQuaPhanTich(this.ham, this.tenNgonNgu, this.kiemTraSua);
}

const ngonNguHoTro = {'tu_dong': 'Tự nhận ngôn ngữ', 'c': 'C', 'c_sharp': 'C#', 'java': 'Java', 'javascript': 'JavaScript', 'python': 'Python'};

Future<KetQuaPhanTich> phanTich(String ma, String ngonNgu, {String? maTruoc}) async {
  http.Response r;
  try {
    r = await http.post(_dc('api/phan-tich'), headers: {'Content-Type': 'application/json'}, body: jsonEncode({'ma': ma, 'ngon_ngu': ngonNgu, 'ma_truoc': ?maTruoc})).timeout(const Duration(minutes: 4));
  } catch (_) {
    throw LoiPhanTich('Không kết nối được máy chủ phân tích. Kiểm tra mạng rồi bấm Phân tích lại.');
  }
  final than = utf8.decode(r.bodyBytes);
  if (r.statusCode == 200) {
    final j = jsonDecode(than);
    return KetQuaPhanTich([for (final h in (j['ham'] as List)) KetQuaHam.tuJson(h)], j['ten_ngon_ngu'] ?? '', j['kiem_tra_sua'] == null ? null : KiemTraSua.tuJson(j['kiem_tra_sua']));
  }
  String? chiTiet;
  try {
    final d = jsonDecode(than)['detail'];
    if (d is String) chiTiet = d;
  } catch (_) {}
  if (r.statusCode == 422) chiTiet = 'Hãy dán mã nguồn trước khi phân tích.';
  throw LoiPhanTich(chiTiet ?? 'Máy chủ đang bận hoặc gặp sự cố. Đợi một lát rồi thử lại.');
}
