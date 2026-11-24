// Nguyễn Tâm Bình - 2324801030003
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../dich_vu/api.dart';
import '../giao_dien/mau.dart';

const caoDong = 30.0;
const _rongRanh = 252.0;
const _rongRanhHep = 104.0;
const _rongKyTu = 9.0;

String soVn(double x, [int le = 2]) => x.toStringAsFixed(le).replaceAll('.', ',');

Color mauMuc(String? mucDo, bool coLoi) => switch (mucDo) {
      'cao' => Mau.dau,
      'trung_binh' => Mau.hoPhach,
      'thap' => Mau.muc2,
      'an_toan' => Mau.dat,
      _ => coLoi ? Mau.dau : Mau.muc2,
    };

Color nenMuc(String? mucDo) => switch (mucDo) {
      'cao' => Mau.dauNhat,
      'trung_binh' => Mau.hoPhachNhat,
      _ => Colors.transparent,
    };

class DongMa {
  final int so;
  final String noiDung;
  final KetQuaHam? ham;
  final double? diem;
  final int? soMuc;
  final MucGiaiThich? giaiThich;
  final bool khongCham;
  final bool goiY, dauGoiY;
  const DongMa(this.so, this.noiDung, {this.ham, this.diem, this.soMuc, this.giaiThich, this.khongCham = false, this.goiY = false, this.dauGoiY = false});
  bool get boSung => soMuc == null && giaiThich?.nguon == 'bo_sung';
}

String thutLe(String dong) => RegExp(r'^[ \t]*').firstMatch(dong)!.group(0)!;

List<String> dongSua(String goc, String maSua) {
  final ds = maSua.split('\n');
  final le = thutLe(goc);
  if (le.isEmpty || thutLe(ds.first).isNotEmpty) return ds;
  return [for (final d in ds) d.trim().isEmpty ? d : '$le$d'];
}

List<DongMa> ghepDong(String ma, List<KetQuaHam> ds) {
  final dong = ma.split('\n');
  final kq = <DongMa>[];
  for (var i = 0; i < dong.length; i++) {
    final so = i + 1;
    final h = ds.where((x) => so >= x.dongBatDau && so <= x.dongKetThuc).firstOrNull;
    if (h == null) {
      kq.add(DongMa(so, dong[i]));
      continue;
    }
    final g = h.muc.where((m) => m.dong == so).firstOrNull;
    kq.add(DongMa(so, dong[i], ham: h, diem: h.diemDong[so], soMuc: h.soMuc(so), giaiThich: g, khongCham: h.dongCuoiDuocCham != null && so > h.dongCuoiDuocCham!));
    final ket = h.muc.where((m) => m.coMaSua && m.denDong == so).firstOrNull;
    if (ket != null) {
      final moi = dongSua(dong[ket.dong - 1], ket.maSua);
      for (var k = 0; k < moi.length; k++) {
        kq.add(DongMa(ket.dong, moi[k], ham: h, giaiThich: ket, goiY: true, dauGoiY: k == 0));
      }
    }
  }
  return kq;
}

int viTriDong(List<DongMa> ds, int so) {
  final i = ds.indexWhere((d) => d.so == so && !d.goiY);
  return i < 0 ? so - 1 : i;
}

class KhungMa extends StatefulWidget {
  final List<DongMa> dong;
  final ScrollController cuon;
  final KetQuaHam? hamChon;
  final int? dongSang;
  final ValueChanged<KetQuaHam> onChonHam;
  final void Function(int dong, String? maSua)? onSua;
  final Map<int, String> daSua;
  const KhungMa({super.key, required this.dong, required this.cuon, required this.hamChon, required this.dongSang, required this.onChonHam, this.onSua, this.daSua = const {}});
  @override
  State<KhungMa> createState() => _KhungMaState();
}

class _KhungMaState extends State<KhungMa> {
  final _ranh = ScrollController();
  final _ngang = ScrollController();

  @override
  void initState() {
    super.initState();
    widget.cuon.addListener(_dongBo);
  }

  void _dongBo() {
    if (_ranh.hasClients && widget.cuon.hasClients) _ranh.jumpTo(widget.cuon.offset.clamp(0.0, _ranh.position.maxScrollExtent));
  }

  @override
  void dispose() {
    widget.cuon.removeListener(_dongBo);
    _ranh.dispose();
    _ngang.dispose();
    super.dispose();
  }

  bool _chon(DongMa d) => d.ham != null && d.ham == widget.hamChon;

  Color _nen(int so, DongMa d) {
    if (d.goiY) return Mau.datNhat;
    if (widget.dongSang == so) return Mau.chon;
    if (d.soMuc != null || d.boSung) {
      final n = nenMuc(d.giaiThich?.mucDo ?? (d.ham!.coLoi ? 'cao' : null));
      if (n != Colors.transparent) return n;
    }
    return _chon(d) ? Mau.nen : Mau.mat;
  }

  @override
  Widget build(BuildContext context) {
    final dai = widget.dong.fold<int>(0, (m, d) => math.max(m, d.noiDung.replaceAll('\t', '    ').length));
    return LayoutBuilder(builder: (_, c) {
      final hep = c.maxWidth < 640;
      final rongRanh = hep ? _rongRanhHep : _rongRanh;
      final rongMa = math.max(c.maxWidth - rongRanh, dai * _rongKyTu + 48);
      return Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        SizedBox(
          width: rongRanh,
          child: ScrollConfiguration(
            behavior: ScrollConfiguration.of(context).copyWith(scrollbars: false),
            child: ListView.builder(
              controller: _ranh,
              physics: const NeverScrollableScrollPhysics(),
              itemExtent: caoDong,
              padding: const EdgeInsets.symmetric(vertical: 12),
              itemCount: widget.dong.length,
              itemBuilder: (_, i) {
                final d = widget.dong[i];
                if (d.goiY) return _RanhGoiY(d: d, hep: hep, onSua: widget.onSua, daApDung: widget.daSua.containsKey(d.so));
                return _Ranh(so: d.so, d: d, nen: _nen(d.so, d), hep: hep, onChon: widget.onChonHam);
              },
            ),
          ),
        ),
        Expanded(
          child: Scrollbar(
            controller: _ngang,
            thumbVisibility: true,
            notificationPredicate: (n) => n.depth == 0,
            child: SingleChildScrollView(
              controller: _ngang,
              scrollDirection: Axis.horizontal,
              child: SizedBox(
                width: rongMa,
                child: Scrollbar(
                  controller: widget.cuon,
                  thumbVisibility: true,
                  child: ListView.builder(
                    controller: widget.cuon,
                    itemExtent: caoDong,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    itemCount: widget.dong.length,
                    itemBuilder: (_, i) {
                      final d = widget.dong[i];
                      if (d.goiY) {
                        final da = widget.daSua.containsKey(d.so);
                        return Container(
                          padding: const EdgeInsets.only(left: 14),
                          alignment: Alignment.centerLeft,
                          decoration: BoxDecoration(color: da ? Mau.dat.withValues(alpha: 0.22) : Mau.datNhat, border: Border(bottom: BorderSide(color: Mau.keNhat), left: BorderSide(color: Mau.dat, width: 3))),
                          child: Text(d.noiDung.replaceAll('\t', '    '), softWrap: false, style: Chu.ma(mau: Mau.dat, dam: FontWeight.w600)),
                        );
                      }
                      final boDi = !d.goiY && (d.ham?.muc.any((m) => m.coMaSua && widget.daSua.containsKey(m.dong) && d.so >= m.dong && d.so <= m.denDong) ?? false);
                      final noiBat = d.soMuc != null || d.boSung;
                      return GestureDetector(
                        onTap: d.ham == null ? null : () => widget.onChonHam(d.ham!),
                        child: Container(
                          padding: const EdgeInsets.only(left: 14),
                          alignment: Alignment.centerLeft,
                          decoration: BoxDecoration(color: _nen(d.so, d), border: Border(bottom: BorderSide(color: Mau.keNhat))),
                          child: Text(d.noiDung.replaceAll('\t', '    '), softWrap: false, style: boDi ? Chu.ma(mau: Mau.muc3).copyWith(decoration: TextDecoration.lineThrough, decorationColor: Mau.dau) : Chu.ma(mau: d.khongCham ? Mau.muc3 : Mau.muc, dam: noiBat ? FontWeight.w600 : FontWeight.w400)),
                        ),
                      );
                    },
                  ),
                ),
              ),
            ),
          ),
        ),
      ]);
    });
  }
}

class _Ranh extends StatelessWidget {
  final int so;
  final DongMa d;
  final Color nen;
  final bool hep;
  final ValueChanged<KetQuaHam> onChon;
  const _Ranh({required this.so, required this.d, required this.nen, required this.hep, required this.onChon});

  @override
  Widget build(BuildContext context) {
    final mau = mauMuc(d.giaiThich?.mucDo, d.ham?.coLoi ?? false);
    final noiBat = d.soMuc != null || d.boSung;
    return MouseRegion(
      cursor: d.ham != null ? SystemMouseCursors.click : MouseCursor.defer,
      child: GestureDetector(
        onTap: d.ham == null ? null : () => onChon(d.ham!),
        child: Container(
          decoration: BoxDecoration(color: nen, border: Border(bottom: BorderSide(color: Mau.keNhat), right: BorderSide(color: Mau.ke))),
          child: hep
              ? Row(children: [
                  SizedBox(width: 30, child: Text('$so', textAlign: TextAlign.right, style: Chu.ma(co: 13, mau: noiBat ? Mau.muc : Mau.muc3, dam: noiBat ? FontWeight.w700 : FontWeight.w400))),
                  const SizedBox(width: 6),
                  Container(width: 24, height: 8, alignment: Alignment.centerLeft, decoration: BoxDecoration(border: Border(left: BorderSide(color: Mau.ke), right: BorderSide(color: Mau.ke))), child: d.diem == null ? null : FractionallySizedBox(widthFactor: d.diem!.clamp(0.06, 1.0), child: Container(color: noiBat ? mau : Mau.muc2.withValues(alpha: 0.5)))),
                  const SizedBox(width: 6),
                  Expanded(child: Align(alignment: Alignment.centerLeft, child: d.soMuc != null ? TheMuc(nhan: '${d.soMuc}', mau: mau) : d.boSung ? TheMuc(nhan: '+', mau: mau, netDut: true) : null)),
                ])
              : Row(children: [
            SizedBox(width: 46, child: Text('$so', textAlign: TextAlign.right, style: Chu.ma(co: 14, mau: noiBat ? Mau.muc : Mau.muc3, dam: noiBat ? FontWeight.w700 : FontWeight.w400))),
            const SizedBox(width: 10),
            Container(
              width: 40,
              height: 10,
              alignment: Alignment.centerLeft,
              decoration: BoxDecoration(border: Border(left: BorderSide(color: Mau.ke), right: BorderSide(color: Mau.ke))),
              child: d.diem == null ? null : FractionallySizedBox(widthFactor: d.diem!.clamp(0.04, 1.0), child: Container(color: noiBat ? mau : Mau.muc2.withValues(alpha: 0.5))),
            ),
            const SizedBox(width: 8),
            SizedBox(width: 42, child: d.diem == null ? null : Text(soVn(d.diem!), textAlign: TextAlign.right, style: Chu.ma(co: 13, mau: noiBat ? Mau.muc : Mau.muc3, dam: noiBat ? FontWeight.w700 : FontWeight.w400))),
            const SizedBox(width: 10),
            Expanded(child: Align(alignment: Alignment.centerLeft, child: d.soMuc != null ? TheMuc(nhan: 'Mục ${d.soMuc}', mau: mau) : d.boSung ? TheMuc(nhan: 'Bổ sung', mau: mau, netDut: true) : null)),
          ]),
        ),
      ),
    );
  }
}

class _RanhGoiY extends StatelessWidget {
  final DongMa d;
  final bool hep;
  final void Function(int dong, String? maSua)? onSua;
  final bool daApDung;
  const _RanhGoiY({required this.d, required this.hep, required this.onSua, this.daApDung = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.only(left: hep ? 14 : 34, right: 8),
      decoration: BoxDecoration(color: Mau.datNhat, border: Border(bottom: BorderSide(color: Mau.keNhat), right: BorderSide(color: Mau.ke))),
      child: Row(children: [
        Text('+', style: Chu.ma(co: 15, mau: Mau.dat, dam: FontWeight.w700)),
        const SizedBox(width: 8),
        if (d.dauGoiY && !hep) Text(daApDung ? 'Đã sửa' : 'Gợi ý sửa', style: Chu.nhan(co: 12.5, mau: Mau.dat)),
        const Spacer(),
        if (d.dauGoiY && onSua != null)
          Tooltip(
            message: daApDung ? 'Bỏ sửa dòng ${d.so}' : 'Thay dòng ${d.so} bằng gợi ý này',
            child: TextButton(
              onPressed: () => onSua!(d.so, daApDung ? null : d.giaiThich!.maSua),
              style: TextButton.styleFrom(foregroundColor: Mau.chuTrenMau, backgroundColor: Mau.dat, minimumSize: const Size(0, 24), padding: const EdgeInsets.symmetric(horizontal: 8), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(3)), textStyle: const TextStyle(fontFamily: Chu.sans, fontSize: 12.5, fontWeight: FontWeight.w600)),
              child: Text(daApDung ? 'Hoàn tác' : (hep ? 'Sửa' : 'Xác nhận sửa')),
            ),
          ),
      ]),
    );
  }
}

class TheMuc extends StatelessWidget {
  final String nhan;
  final Color mau;
  final bool netDut;
  const TheMuc({super.key, required this.nhan, required this.mau, this.netDut = false});
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
        decoration: BoxDecoration(color: netDut ? Mau.mat : mau, border: Border.all(color: mau, width: 1.3), borderRadius: BorderRadius.circular(3)),
        child: Text(nhan, style: Chu.nhan(co: 13, mau: netDut ? mau : Mau.chuTrenMau, dam: FontWeight.w700)),
      );
}

class ConDau extends StatelessWidget {
  final bool coLoi;
  final double xacSuat;
  final bool daVa, canXemLai;
  const ConDau({super.key, required this.coLoi, required this.xacSuat, this.daVa = false, this.canXemLai = false});
  @override
  Widget build(BuildContext context) {
    final m = daVa || !coLoi ? Mau.dat : (canXemLai ? Mau.hoPhach : Mau.dau);
    final dau = Container(
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(border: Border.all(color: m, width: 2.2), borderRadius: BorderRadius.circular(4)),
      child: Container(
        width: 118,
        padding: const EdgeInsets.symmetric(vertical: 6),
        decoration: BoxDecoration(border: Border.all(color: m, width: 1), borderRadius: BorderRadius.circular(2)),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Text(daVa ? 'ĐÃ VÁ' : (!coLoi ? 'ĐẠT' : (canXemLai ? 'CẦN XEM LẠI' : 'CẢNH BÁO')), style: Chu.nhan(co: 13, mau: m, dam: FontWeight.w700).copyWith(letterSpacing: 2.2)),
          const SizedBox(height: 2),
          daVa ? Icon(Icons.verified_outlined, color: m, size: 30) : Text(soVn(xacSuat), style: Chu.ma(co: 26, mau: m, dam: FontWeight.w700)),
        ]),
      ),
    );
    final tat = MediaQuery.of(context).disableAnimations;
    return Semantics(
      label: '${daVa ? 'Đã vá' : !coLoi ? 'Đạt' : canXemLai ? 'Cần xem lại' : 'Cảnh báo'}, xác suất lỗi ${soVn(xacSuat)}',
      child: TweenAnimationBuilder<double>(
        tween: Tween(begin: tat ? 1 : 0, end: 1),
        duration: const Duration(milliseconds: 260),
        curve: Curves.easeOutExpo,
        child: dau,
        builder: (_, t, con) => Opacity(opacity: (0.35 + 0.65 * t).clamp(0.0, 1.0), child: Transform.rotate(angle: -0.045, child: Transform.scale(scale: 1.18 - 0.18 * t, child: con))),
      ),
    );
  }
}

class PhanHam extends StatelessWidget {
  final KetQuaHam h;
  final bool chon;
  final VoidCallback onChon;
  final ValueChanged<int> onChonDong;
  final void Function(int dong, String? maSua)? onSua;
  final Map<int, String> daSua;
  final KetQuaHam? truoc;
  final bool daVa;
  const PhanHam({super.key, required this.h, required this.chon, required this.onChon, required this.onChonDong, this.onSua, this.daSua = const {}, this.truoc, this.daVa = false});

  String _dongGoc(int dong, [int? den]) {
    final ds = h.ma.split('\n');
    final i = dong - h.dongBatDau, j = (den ?? dong) - h.dongBatDau;
    if (i < 0 || i >= ds.length) return '';
    return ds.sublist(i, (j < i ? i : j) + 1 > ds.length ? ds.length : (j < i ? i : j) + 1).join('\n');
  }

  @override
  Widget build(BuildContext context) {
    final boSung = h.muc.where((m) => m.nguon == 'bo_sung' && h.soMuc(m.dong) == null).toList();
    final coGiaiThich = h.tomTat != null;
    return Container(
      decoration: BoxDecoration(color: Mau.mat, border: Border.all(color: chon ? Mau.muc2 : Mau.ke), borderRadius: BorderRadius.circular(4)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        InkWell(
          onTap: onChon,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(18, 16, 18, 14),
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(h.ten, style: Chu.ma(co: 17, dam: FontWeight.w700), overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 6),
                  Text.rich(TextSpan(style: Chu.nhan(co: 13, mau: Mau.muc3, dam: FontWeight.w500), children: [const TextSpan(text: 'Dòng '), TextSpan(text: '${h.dongBatDau}–${h.dongKetThuc}', style: Chu.ma(co: 13, mau: Mau.muc3))])),
                  const SizedBox(height: 10),
                  daVa
                      ? Text.rich(TextSpan(style: TextStyle(fontSize: 14, color: Mau.muc3), children: [
                          const TextSpan(text: 'Điểm mô hình '),
                          TextSpan(text: soVn(h.xacSuat), style: Chu.ma(co: 14, mau: Mau.muc3, dam: FontWeight.w700)),
                          TextSpan(text: ' (ngưỡng ${soVn(h.nguong)}), chỉ để tham khảo sau khi vá'),
                        ]))
                      : Text.rich(TextSpan(style: TextStyle(fontSize: 14, color: Mau.muc2), children: [
                          const TextSpan(text: 'Xác suất có lỗ hổng '),
                          TextSpan(text: soVn(h.xacSuat), style: Chu.ma(co: 14, dam: FontWeight.w700)),
                          const TextSpan(text: ', ngưỡng cảnh báo '),
                          TextSpan(text: soVn(h.nguong), style: Chu.ma(co: 14)),
                        ])),
                  if (truoc != null)
                    Container(
                      margin: const EdgeInsets.only(top: 8),
                      padding: const EdgeInsets.fromLTRB(10, 6, 10, 6),
                      decoration: BoxDecoration(color: Mau.nen, border: Border.all(color: Mau.ke), borderRadius: BorderRadius.circular(3)),
                      child: Text.rich(TextSpan(style: TextStyle(fontSize: 14, color: Mau.muc2), children: [
                        const TextSpan(text: 'Trước khi vá '),
                        TextSpan(text: soVn(truoc!.xacSuat), style: Chu.ma(co: 14, mau: truoc!.coLoi ? Mau.dau : Mau.dat, dam: FontWeight.w700)),
                        const TextSpan(text: '  →  sau khi vá '),
                        TextSpan(text: soVn(h.xacSuat), style: Chu.ma(co: 14, mau: h.coLoi ? Mau.dau : Mau.dat, dam: FontWeight.w700)),
                        TextSpan(text: '  (ngưỡng ${soVn(h.nguong)}, ${h.xacSuat <= truoc!.xacSuat ? 'giảm' : 'tăng'} ${soVn((h.xacSuat - truoc!.xacSuat).abs())})'),
                      ])),
                    ),
                  if (h.canXemLai && !daVa)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text('Mô hình chấm trên ngưỡng nhưng phần giải thích không thấy lỗ hổng rõ ràng trong hàm. Nhiều khả năng đây là cảnh báo nhầm, vì mô hình chấm theo kiểu hàm (xử lý yêu cầu, truy vấn dữ liệu…) chứ chưa nhận ra hàm đã được viết an toàn; nên người rà soát kiểm tra lại.', style: TextStyle(fontSize: 13, height: 1.45, color: Mau.hoPhach)),
                    ),
                  if (daVa && h.coLoi)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text('Kết luận ĐÃ VÁ dựa trên kiểm tra bản sửa; mô hình phân loại vẫn chấm trên ngưỡng vì nó nhận diện kiểu hàm chứ chưa nhận ra bản vá.', style: TextStyle(fontSize: 13, height: 1.45, color: Mau.hoPhach)),
                    ),
                  if (h.biCat) Padding(padding: const EdgeInsets.only(top: 6), child: Text('Hàm dài hơn 2.048 token: mô hình chỉ chấm tới dòng ${h.dongCuoiDuocCham}.', style: TextStyle(fontSize: 13, color: Mau.hoPhach))),
                  if (!h.biCat && h.dongCuoiChamDiem != null) Padding(padding: const EdgeInsets.only(top: 6), child: Text('Xác suất tính trên cả hàm; điểm chú ý từng dòng chỉ tính tới dòng ${h.dongCuoiChamDiem}.', style: TextStyle(fontSize: 13, color: Mau.hoPhach))),
                ]),
              ),
              const SizedBox(width: 12),
              ConDau(key: ValueKey('${h.ten}${h.dongBatDau}${h.xacSuat}$daVa${h.canXemLai}'), coLoi: h.coLoi, xacSuat: h.xacSuat, daVa: daVa, canXemLai: h.canXemLai && !daVa),
            ]),
          ),
        ),
        const Divider(),
        Padding(
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            if (coGiaiThich && !h.coLoi)
              Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Text('Mô hình chấm ${soVn(h.xacSuat)}, dưới ngưỡng ${soVn(h.nguong)} nên đóng dấu ĐẠT. Phần dưới là phân tích gợi ý độc lập với mô hình và vẫn có thể nêu nguy cơ.', style: TextStyle(fontSize: 14, height: 1.5, color: Mau.muc2, fontWeight: FontWeight.w600)),
              ),
            if (coGiaiThich) Padding(padding: const EdgeInsets.only(bottom: 12), child: Text(h.tomTat!, style: TextStyle(fontSize: 15, height: 1.5, color: Mau.muc))),
            if (h.maVa.trim().isNotEmpty && !daVa)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Theme(
                  data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                  child: ExpansionTile(
                    tilePadding: EdgeInsets.zero,
                    childrenPadding: EdgeInsets.zero,
                    leading: Icon(Icons.healing_outlined, color: Mau.dat, size: 20),
                    title: Text('Bản vá cả hàm (dùng khi bấm Xác nhận sửa tất cả)', style: Chu.nhan(co: 14, mau: Mau.dat)),
                    children: [
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.fromLTRB(10, 8, 10, 8),
                        decoration: BoxDecoration(color: Mau.datNhat, border: Border(left: BorderSide(color: Mau.dat, width: 3)), borderRadius: BorderRadius.circular(2)),
                        child: SingleChildScrollView(
                          scrollDirection: Axis.horizontal,
                          child: Text(h.maVa.replaceAll('\t', '    '), style: Chu.ma(co: 13, mau: Mau.muc).copyWith(height: 1.5)),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            if (!coGiaiThich && h.giaiThichLoi) Padding(padding: EdgeInsets.only(bottom: 12), child: Text('Chưa lấy được phần giải thích lúc này. Các dòng mô hình chú ý vẫn được đánh dấu bên dưới.', style: TextStyle(fontSize: 14, color: Mau.hoPhach))),
            if (!coGiaiThich && !h.giaiThichLoi) Padding(padding: EdgeInsets.only(bottom: 12), child: Text('Hàm dưới ngưỡng cảnh báo. Ba dòng mô hình chú ý nhiều nhất:', style: TextStyle(fontSize: 14, color: Mau.muc2))),
            for (final d in h.dongDanhDau) _MucView(soMuc: '${h.soMuc(d)}', dong: d, diem: h.diemDong[d], m: h.muc.where((x) => x.dong == d).firstOrNull, coLoi: h.coLoi, onChon: () => onChonDong(d), onSua: onSua, maGoc: _dongGoc(d, h.muc.where((x) => x.dong == d).firstOrNull?.denDong ?? d), daApDung: daSua.containsKey(d)),
            for (final m in boSung) _MucView(soMuc: '+', dong: m.dong, diem: h.diemDong[m.dong], m: m, coLoi: h.coLoi, boSung: true, onChon: () => onChonDong(m.dong), onSua: onSua, maGoc: _dongGoc(m.dong, m.denDong), daApDung: daSua.containsKey(m.dong)),
          ]),
        ),
      ]),
    );
  }
}

class _MucView extends StatelessWidget {
  final String soMuc;
  final int dong;
  final double? diem;
  final MucGiaiThich? m;
  final bool coLoi, boSung;
  final VoidCallback onChon;
  final void Function(int dong, String? maSua)? onSua;
  final String maGoc;
  final bool daApDung;
  const _MucView({this.daApDung = false, required this.soMuc, required this.dong, required this.diem, required this.m, required this.coLoi, this.boSung = false, required this.onChon, this.onSua, this.maGoc = ''});

  static const _tenMuc = {'cao': 'Mức cao', 'trung_binh': 'Mức trung bình', 'thap': 'Mức thấp', 'an_toan': 'Có vẻ an toàn'};

  @override
  Widget build(BuildContext context) {
    final mau = mauMuc(m?.mucDo, coLoi);
    return InkWell(
      onTap: onChon,
      borderRadius: BorderRadius.circular(4),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Container(
            width: 32,
            height: 32,
            alignment: Alignment.center,
            decoration: BoxDecoration(color: boSung ? Mau.mat : mau, border: Border.all(color: mau, width: 1.4), borderRadius: BorderRadius.circular(3)),
            child: Text(soMuc, style: Chu.ma(co: 15, mau: boSung ? mau : Mau.chuTrenMau, dam: FontWeight.w700)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Wrap(spacing: 10, runSpacing: 4, crossAxisAlignment: WrapCrossAlignment.center, children: [
                Text('Dòng $dong', style: Chu.ma(co: 14, dam: FontWeight.w700)),
                if (diem != null) Text('điểm ${soVn(diem!)}', style: Chu.ma(co: 13, mau: Mau.muc2)),
                if (boSung) Text('Phân tích bổ sung', style: Chu.nhan(co: 13, mau: Mau.muc2)),
                if (m != null) Text(_tenMuc[m!.mucDo] ?? m!.mucDo, style: Chu.nhan(co: 13, mau: mau)),
                if (m != null && m!.cwe.isNotEmpty) Text(m!.cwe, style: TextStyle(fontSize: 13.5, color: Mau.muc2, fontWeight: FontWeight.w500)),
              ]),
              if (m != null) ...[
                const SizedBox(height: 4),
                Text(m!.lyDo, style: TextStyle(fontSize: 14.5, height: 1.5, color: Mau.muc)),
                if (m!.goiYSua.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 4), child: Text.rich(TextSpan(style: TextStyle(fontSize: 14, height: 1.45, color: Mau.muc2), children: [const TextSpan(text: 'Cách sửa: ', style: TextStyle(fontWeight: FontWeight.w600)), TextSpan(text: m!.goiYSua)]))),
                if (m!.coMaSua) ...[
                  const SizedBox(height: 8),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.fromLTRB(10, 8, 10, 8),
                    decoration: BoxDecoration(color: Mau.datNhat, border: Border(left: BorderSide(color: Mau.dat, width: 3)), borderRadius: BorderRadius.circular(2)),
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        for (final g in maGoc.split('\n')) Text('- ${g.trim()}', style: Chu.ma(co: 13, mau: Mau.dau).copyWith(height: 1.5, decoration: TextDecoration.lineThrough)),
                        for (final d in dongSua(maGoc.split('\n').first, m!.maSua)) Text('+ ${d.replaceAll('\t', '    ')}', style: Chu.ma(co: 13, mau: Mau.dat, dam: FontWeight.w600).copyWith(height: 1.5)),
                      ]),
                    ),
                  ),
                  if (onSua != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: daApDung
                          ? Wrap(spacing: 12, crossAxisAlignment: WrapCrossAlignment.center, children: [
                              Row(mainAxisSize: MainAxisSize.min, children: [Icon(Icons.check_circle, size: 18, color: Mau.dat), const SizedBox(width: 6), Text('Đã áp dụng', style: Chu.nhan(co: 14, mau: Mau.dat))]),
                              OutlinedButton(onPressed: () => onSua!(dong, null), child: const Text('Hoàn tác')),
                            ])
                          : FilledButton.icon(
                              onPressed: () => onSua!(dong, m!.maSua),
                              style: FilledButton.styleFrom(backgroundColor: Mau.dat, foregroundColor: Mau.chuTrenMau, minimumSize: const Size(0, 38)),
                              icon: const Icon(Icons.check, size: 18),
                              label: const Text('Xác nhận sửa'),
                            ),
                    ),
                ],
              ],
            ]),
          ),
        ]),
      ),
    );
  }
}
