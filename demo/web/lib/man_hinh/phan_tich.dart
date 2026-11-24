// Nguyễn Tâm Bình - 2324801030003
import 'package:flutter/material.dart';

import '../dich_vu/api.dart';
import '../giao_dien/mau.dart';
import 'ket_qua.dart';

class ManHinhPhanTich extends StatefulWidget {
  const ManHinhPhanTich({super.key});
  @override
  State<ManHinhPhanTich> createState() => _ManHinhPhanTichState();
}

class _ManHinhPhanTichState extends State<ManHinhPhanTich> {
  final _ma = TextEditingController();
  final _cuonMa = ScrollController();
  String? _tenNgonNgu;
  KiemTraSua? _kiemSua;
  bool _vuaSua = false;
  String _maGoc = '';
  final Map<int, String> _daSua = {};
  List<KetQuaHam>? _kqTruoc;
  bool _dang = false, _sua = true;
  String? _loi;
  List<KetQuaHam>? _kq;
  List<DongMa> _dong = const [];
  KetQuaHam? _hamChon;
  int? _dongSang;
  List<ViDu> _viDu = const [];

  @override
  void initState() {
    super.initState();
    layViDu().then((v) => mounted ? setState(() => _viDu = v) : null);
  }

  Future<void> _phanTich([String? maTruoc]) async {
    if (maTruoc == null && _daSua.isNotEmpty && _ma.text == _ghepSua()) maTruoc = _maGoc;
    if (_ma.text.trim().isEmpty) {
      setState(() => _loi = 'Hãy dán mã nguồn hoặc chọn một ví dụ trước khi phân tích.');
      return;
    }
    setState(() {
      _dang = true;
      _loi = null;
    });
    try {
      final ra = await phanTich(_ma.text, 'tu_dong', maTruoc: maTruoc);
      final kq = ra.ham;
      setState(() {
        _tenNgonNgu = ra.tenNgonNgu;
        _kiemSua = ra.kiemTraSua;
        _kqTruoc = maTruoc != null ? (_kq ?? _kqTruoc) : null;
        _vuaSua = maTruoc != null;
        _kq = kq;
        _dong = ghepDong(_ma.text, kq);
        _maGoc = _ma.text;
        _daSua.clear();
        _sua = false;
        _hamChon = kq.where((h) => h.coLoi).firstOrNull ?? (kq.isEmpty ? null : kq.first);
        _dongSang = null;
      });
      if (_hamChon != null) WidgetsBinding.instance.addPostFrameCallback((_) => _cuonToi(_hamChon!.dongBatDau));
    } on LoiPhanTich catch (e) {
      setState(() => _loi = e.loiNhan);
    } finally {
      if (mounted) setState(() => _dang = false);
    }
  }

  String _ghepSua() {
    final ds = _maGoc.split('\n');
    for (final dong in _daSua.keys.toList()..sort((a, b) => b.compareTo(a))) {
      final den = _denDong(dong);
      if (dong >= 1 && den <= ds.length) ds.replaceRange(dong - 1, den, dongSua(ds[dong - 1], _daSua[dong]!));
    }
    return ds.join('\n');
  }

  void _apDungSua(int dong, String? maSua) {
    setState(() {
      if (maSua == null) {
        _daSua.remove(dong);
      } else {
        _daSua[dong] = maSua;
      }
      _ma.text = _ghepSua();
    });
  }

  int _denDong(int dong) => _goiY.where((m) => m.dong == dong).firstOrNull?.denDong ?? dong;

  List<MucGiaiThich> get _goiY => [for (final h in _kq ?? const <KetQuaHam>[]) ...h.muc.where((m) => m.coMaSua)];
  bool get _coBanVa => (_kq ?? const <KetQuaHam>[]).any((h) => h.maVa.trim().isNotEmpty) || _goiY.isNotEmpty;

  void _suaTatCa() {
    final truoc = _maGoc;
    final ds = _maGoc.split('\n');
    final ham = [...?_kq]..sort((a, b) => b.dongBatDau.compareTo(a.dongBatDau));
    for (final h in ham) {
      if (h.maVa.trim().isNotEmpty) {
        final dau = h.dongBatDau - 1, cuoi = h.dongKetThuc;
        if (dau >= 0 && cuoi <= ds.length) ds.replaceRange(dau, cuoi, dongSua(ds[dau], h.maVa));
      } else {
        for (final m in h.muc.where((m) => m.coMaSua).toList()..sort((a, b) => b.dong.compareTo(a.dong))) {
          final i = _daSua[m.dong] ?? m.maSua;
          if (m.dong >= 1 && m.denDong <= ds.length) ds.replaceRange(m.dong - 1, m.denDong, dongSua(ds[m.dong - 1], i));
        }
      }
    }
    _ma.text = ds.join('\n');
    _phanTich(truoc);
  }

  void _phanTichLai() => _phanTich(_maGoc);

  Widget _thanhSua() {
    final tong = _goiY.length;
    final da = _goiY.where((m) => _daSua.containsKey(m.dong)).length;
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 10),
      decoration: BoxDecoration(color: Mau.datNhat, border: Border.all(color: Mau.dat.withValues(alpha: 0.5)), borderRadius: BorderRadius.circular(4)),
      child: Wrap(spacing: 12, runSpacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
        Text('Đã áp dụng $da/$tong gợi ý sửa', style: Chu.nhan(co: 14, mau: Mau.dat)),
        OutlinedButton.icon(onPressed: _dang || da == 0 ? null : _phanTichLai, icon: const Icon(Icons.refresh, size: 18), label: const Text('Phân tích lại')),
        FilledButton.icon(onPressed: _dang ? null : _suaTatCa, style: FilledButton.styleFrom(backgroundColor: Mau.dat, foregroundColor: Mau.chuTrenMau, minimumSize: const Size(0, 40)), icon: const Icon(Icons.done_all, size: 18), label: const Text('Xác nhận sửa tất cả (vá cả hàm)')),
      ]),
    );
  }

  void _cuonToi(int dong) {
    if (!_cuonMa.hasClients) return;
    final dich = ((viTriDong(_dong, dong) - 3) * caoDong).clamp(0.0, _cuonMa.position.maxScrollExtent);
    _cuonMa.animateTo(dich, duration: const Duration(milliseconds: 220), curve: Curves.easeOutCubic);
  }

  bool get _vaTiep => _vuaSua && (_kiemSua?.ketLuan == 'xu_ly_mot_phan' || _kiemSua?.ketLuan == 'chua_xu_ly') && _coBanVa && _ma.text == _ghepSua();

  void _chonViDu(ViDu v) {
    setState(() {
      _ma.text = v.ma;
      _vuaSua = false;
      _kqTruoc = null;
      _daSua.clear();
      _sua = true;
      _kq = null;
      _loi = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        _thanhTren(),
        Expanded(
          child: LayoutBuilder(builder: (_, c) {
            final rong = c.maxWidth >= 1080;
            final trai = _khungTrai();
            final phai = _khungPhai();
            return rong
                ? Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [Expanded(flex: 58, child: trai), VerticalDivider(width: 1, color: Mau.ke), Expanded(flex: 42, child: phai)])
                : Column(children: [SizedBox(height: c.maxHeight * 0.55, child: trai), const Divider(), Expanded(child: phai)]);
          }),
        ),
      ]),
    );
  }

  Widget _thanhTren() {
    Widget tieuDe(bool rong) => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Biên bản kiểm tra lỗ hổng mã nguồn', maxLines: rong ? 1 : null, overflow: rong ? TextOverflow.ellipsis : null, style: TextStyle(fontSize: 21, fontWeight: FontWeight.w700, color: Mau.muc, letterSpacing: -0.2)),
          const SizedBox(height: 4),
          Text('Đồ án Học máy: Ứng dụng LLM và GNN vào phát hiện lỗ hổng bảo mật trong mã nguồn', maxLines: rong ? 1 : null, overflow: rong ? TextOverflow.ellipsis : null, style: TextStyle(fontSize: 13.5, color: Mau.muc2, height: 1.4)),
          Text('Nguyễn Tâm Bình · Nguyễn Hoàng Thiên · Vũ Đức Thắng', maxLines: rong ? 1 : null, overflow: rong ? TextOverflow.ellipsis : null, style: TextStyle(fontSize: 13.5, color: Mau.muc3, height: 1.4)),
        ]);
    final logo = Container(
      padding: EdgeInsets.all(Mau.toi ? 5 : 0),
      decoration: Mau.toi ? const BoxDecoration(color: Colors.white, shape: BoxShape.circle) : null,
      child: Image(image: const AssetImage('assets/logo.png'), height: Mau.toi ? 46 : 56, semanticLabel: 'Logo TBT'),
    );
    final dieuKhien = Wrap(spacing: 12, runSpacing: 10, crossAxisAlignment: WrapCrossAlignment.center, children: [
      SegmentedButton<ThemeMode>(
        segments: const [
          ButtonSegment(value: ThemeMode.system, icon: Icon(Icons.brightness_auto_outlined, size: 18), tooltip: 'Theo trình duyệt'),
          ButtonSegment(value: ThemeMode.light, icon: Icon(Icons.light_mode_outlined, size: 18), tooltip: 'Giao diện sáng'),
          ButtonSegment(value: ThemeMode.dark, icon: Icon(Icons.dark_mode_outlined, size: 18), tooltip: 'Giao diện tối'),
        ],
        selected: {cheDoGiaoDien.value},
        showSelectedIcon: false,
        style: SegmentedButton.styleFrom(selectedBackgroundColor: Mau.nut, selectedForegroundColor: Colors.white, foregroundColor: Mau.muc2, side: BorderSide(color: Mau.ke), minimumSize: const Size(44, 44)),
        onSelectionChanged: (s) => cheDoGiaoDien.value = s.first,
      ),
      FilledButton.icon(
        onPressed: _dang ? null : (_vaTiep ? _suaTatCa : _phanTich),
        icon: _dang ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)) : Icon(_vaTiep ? Icons.healing_outlined : Icons.fact_check_outlined, size: 20),
        label: Text(_dang ? 'Đang kiểm tra' : (_vaTiep ? 'Vá tiếp' : 'Phân tích')),
      ),
    ]);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      decoration: BoxDecoration(color: Mau.mat, border: Border(bottom: BorderSide(color: Mau.ke))),
      child: LayoutBuilder(
        builder: (_, c) => c.maxWidth >= 1000
            ? Row(children: [logo, const SizedBox(width: 16), Expanded(child: tieuDe(true)), const SizedBox(width: 24), dieuKhien])
            : Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Row(crossAxisAlignment: CrossAxisAlignment.start, children: [logo, const SizedBox(width: 12), Expanded(child: tieuDe(false))]), const SizedBox(height: 12), dieuKhien]),
      ),
    );
  }

  Widget _khungTrai() {
    return Container(
      color: Mau.mat,
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Container(
          height: 52,
          padding: const EdgeInsets.symmetric(horizontal: 20),
          decoration: BoxDecoration(border: Border(bottom: BorderSide(color: Mau.keNhat))),
          child: Row(children: [
            Expanded(child: Text(_sua ? 'Mã nguồn cần kiểm tra' : 'Mã nguồn đã kiểm tra', style: Chu.nhan(co: 14, mau: Mau.muc), overflow: TextOverflow.ellipsis)),
            if (!_sua) ...[
              if (MediaQuery.sizeOf(context).width >= 700) ...[Text('Vạch và điểm: mức mô hình chú ý', style: Chu.nhan(co: 13, mau: Mau.muc3, dam: FontWeight.w500)), const SizedBox(width: 16)],
              OutlinedButton.icon(onPressed: _dang ? null : () => setState(() => _sua = true), icon: const Icon(Icons.edit_outlined, size: 18), label: const Text('Sửa mã')),
            ],
            if (_sua && _ma.text.isNotEmpty) TextButton(onPressed: _dang ? null : () => setState(() => _ma.clear()), child: const Text('Xóa hết')),
          ]),
        ),
        Expanded(
          child: _sua
              ? Padding(
                  padding: const EdgeInsets.fromLTRB(20, 12, 20, 12),
                  child: TextField(
                    controller: _ma,
                    expands: true,
                    maxLines: null,
                    onChanged: (_) => setState(() {}),
                    style: Chu.ma().copyWith(height: 1.6),
                    decoration: InputDecoration(border: InputBorder.none, hintText: 'Dán một hoặc vài hàm C, C#, Java, JavaScript hoặc Python vào đây, hoặc chọn ví dụ bên phải.', hintStyle: TextStyle(fontFamily: Chu.sans, fontSize: 15, color: Mau.muc3)),
                  ),
                )
              : KhungMa(dong: _dong, cuon: _cuonMa, hamChon: _hamChon, dongSang: _dongSang, onChonHam: (h) => setState(() => _hamChon = h), onSua: _dang ? null : _apDungSua, daSua: _daSua),
        ),
      ]),
    );
  }

  Widget _khungPhai() {
    return ListView(padding: const EdgeInsets.fromLTRB(24, 20, 24, 28), children: [
      if (_loi != null) _thongBaoLoi(_loi!),
      if (_dang) ..._dangCho() else if (_kq == null) ..._trong() else ..._ketQua(),
    ]);
  }

  Widget _thongBaoLoi(String s) => Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(color: Mau.dauNhat, border: Border.all(color: Mau.dau), borderRadius: BorderRadius.circular(4)),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [Icon(Icons.error_outline, color: Mau.dau, size: 20), const SizedBox(width: 10), Expanded(child: Text(s, style: TextStyle(fontSize: 14.5, color: Mau.muc, height: 1.45)))]),
      );

  List<Widget> _dangCho() => [
        Text('Đang kiểm tra từng hàm và viết giải thích, thường mất 20 giây đến 1 phút; hàm dài có thể lâu hơn.', style: TextStyle(fontSize: 15, color: Mau.muc2, height: 1.5)),
        const SizedBox(height: 16),
        for (final h in [168.0, 120.0]) Container(height: h, margin: const EdgeInsets.only(bottom: 14), decoration: BoxDecoration(color: Mau.keNhat, borderRadius: BorderRadius.circular(4))),
      ];

  List<Widget> _trong() => [
        Text('Chưa có biên bản', style: TextStyle(fontSize: 19, fontWeight: FontWeight.w700, color: Mau.muc)),
        const SizedBox(height: 8),
        Text('Dán mã C, C#, Java, JavaScript hoặc Python bên trái rồi bấm Phân tích; hệ thống tự nhận ngôn ngữ. Mô hình Qwen2.5-Coder-1.5B tinh chỉnh LoRA trên bộ R2Vul chấm từng hàm, ngưỡng cảnh báo chọn riêng cho từng ngôn ngữ trên tập valid. Mỗi hàm sẽ được đóng dấu kết luận kèm xác suất có lỗ hổng; ba dòng mô hình chú ý nhiều nhất mang thẻ Mục, mỗi mục có lời giải thích cùng số.', style: TextStyle(fontSize: 15, height: 1.55, color: Mau.muc2)),
        const SizedBox(height: 22),
        if (_viDu.isNotEmpty) ...[
          Text('Ví dụ lấy từ tập valid và test của R2Vul', style: Chu.nhan(co: 13, mau: Mau.muc)),
          const SizedBox(height: 10),
          for (final v in _viDu)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: OutlinedButton(
                onPressed: () => _chonViDu(v),
                style: OutlinedButton.styleFrom(alignment: Alignment.centerLeft, padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12)),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(v.ten, style: Chu.ma(co: 14, dam: FontWeight.w700)),
                  const SizedBox(height: 4),
                  Text(v.moTa, style: TextStyle(fontSize: 13.5, color: Mau.muc2, height: 1.4, fontWeight: FontWeight.w400)),
                ]),
              ),
            ),
        ],
      ];

  Widget _khungKiemSua() {
    final k = _kiemSua;
    final (nhan, mau, nen) = switch (k?.ketLuan) {
      'da_xu_ly' => ('Đã xử lý lỗ hổng', Mau.dat, Mau.datNhat),
      'xu_ly_mot_phan' => ('Mới xử lý một phần', Mau.hoPhach, Mau.hoPhachNhat),
      'chua_xu_ly' => ('Chưa xử lý được lỗ hổng', Mau.dau, Mau.dauNhat),
      _ => ('Chưa kiểm tra được bản sửa lúc này', Mau.muc2, Mau.nen),
    };
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: BoxDecoration(color: nen, border: Border(left: BorderSide(color: mau, width: 4)), borderRadius: BorderRadius.circular(4)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('KIỂM TRA BẢN SỬA', style: Chu.nhan(co: 12.5, mau: Mau.muc2, dam: FontWeight.w700).copyWith(letterSpacing: 1.6)),
        const SizedBox(height: 6),
        Row(children: [Icon(k?.ketLuan == 'da_xu_ly' ? Icons.verified_outlined : Icons.error_outline, color: mau, size: 20), const SizedBox(width: 8), Expanded(child: Text(nhan, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: mau)))]),
        if (k != null && k.nhanXet.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 8), child: Text(k.nhanXet, style: TextStyle(fontSize: 14.5, height: 1.5, color: Mau.muc))),
        if (k != null && k.conLai.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 6), child: Text.rich(TextSpan(style: TextStyle(fontSize: 14, height: 1.45, color: Mau.muc2), children: [const TextSpan(text: 'Còn cần làm: ', style: TextStyle(fontWeight: FontWeight.w600)), TextSpan(text: k.conLai)]))),
        Padding(
          padding: const EdgeInsets.only(top: 10),
          child: Text('Đây là nhận định của lớp giải thích khi so mã trước và sau khi sửa. Con dấu bên dưới vẫn là kết luận của mô hình phân loại: mô hình chấm theo kiểu hàm nên có thể vẫn cảnh báo dù lỗ hổng đã được vá.', style: TextStyle(fontSize: 13, height: 1.5, color: Mau.muc3)),
        ),
      ]),
    );
  }

  List<Widget> _ketQua() {
    final kq = _kq!;
    final daVa = _vuaSua && _kiemSua?.ketLuan == 'da_xu_ly';
    final soDaVa = daVa ? kq.where((h) => h.coLoi && (_kqTruoc?.where((x) => x.ten == h.ten).firstOrNull?.coLoi ?? false)).length : 0;
    final soXemLai = daVa ? 0 : kq.where((h) => h.canXemLai).length;
    final soCanhBao = kq.where((h) => h.coLoi).length - soDaVa - soXemLai;
    final soDongCao = kq.fold<int>(0, (t, h) => t + h.muc.where((m) => m.mucDo == 'cao').length);
    final soDongVua = kq.fold<int>(0, (t, h) => t + h.muc.where((m) => m.mucDo == 'trung_binh').length);
    return [
      Text.rich(TextSpan(style: TextStyle(fontSize: 15, color: Mau.muc2), children: [
        TextSpan(text: '${kq.length}', style: Chu.ma(co: 15, dam: FontWeight.w700)),
        TextSpan(text: ' hàm', style: TextStyle(fontWeight: FontWeight.w700, color: Mau.muc)),
        const TextSpan(text: ' đã kiểm tra · '),
        TextSpan(text: '$soCanhBao', style: Chu.ma(co: 15, mau: soCanhBao > 0 ? Mau.dau : Mau.dat, dam: FontWeight.w700)),
        TextSpan(text: ' hàm bị cảnh báo', style: TextStyle(fontWeight: FontWeight.w700, color: soCanhBao > 0 ? Mau.dau : Mau.dat)),
        if (soXemLai > 0) TextSpan(text: ' · $soXemLai hàm cần xem lại', style: TextStyle(fontWeight: FontWeight.w700, color: Mau.hoPhach)),
        if (soDaVa > 0) TextSpan(text: ' · $soDaVa hàm đã vá', style: TextStyle(fontWeight: FontWeight.w700, color: Mau.dat)),
        if (soDongCao > 0) TextSpan(text: ' · $soDongCao dòng mức cao', style: TextStyle(color: Mau.dau)),
        if (soDongVua > 0) TextSpan(text: ' · $soDongVua dòng mức trung bình', style: TextStyle(color: Mau.hoPhach)),
        if ((_tenNgonNgu ?? '').isNotEmpty) TextSpan(text: ' · ngôn ngữ $_tenNgonNgu'),
      ])),
      const SizedBox(height: 8),
      Text('Thẻ Mục đánh dấu nơi mô hình chú ý nhiều nhất, không khẳng định chắc chắn đó là dòng lỗi. Phần giải thích là gợi ý để người rà soát kiểm tra lại.', style: TextStyle(fontSize: 13.5, height: 1.5, color: Mau.muc2)),
      const SizedBox(height: 16),
      if (_vuaSua) _khungKiemSua(),
      if (_coBanVa && !(_vuaSua && _kiemSua?.ketLuan == 'da_xu_ly')) _thanhSua(),
      for (final h in kq)
        Padding(
          padding: const EdgeInsets.only(bottom: 16),
          child: PhanHam(
            h: h,
            truoc: _kqTruoc?.where((x) => x.ten == h.ten).firstOrNull,
            daVa: _vuaSua && _kiemSua?.ketLuan == 'da_xu_ly' && (_kqTruoc?.where((x) => x.ten == h.ten).firstOrNull?.coLoi ?? false),
            chon: h == _hamChon,
            onChon: () {
              setState(() => _hamChon = h);
              _cuonToi(h.dongBatDau);
            },
            onSua: _dang ? null : _apDungSua,
            daSua: _daSua,
            onChonDong: (d) {
              setState(() {
                _hamChon = h;
                _dongSang = d;
              });
              _cuonToi(d);
            },
          ),
        ),
    ];
  }
}
