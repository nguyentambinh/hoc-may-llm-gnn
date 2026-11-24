// Nguyễn Tâm Bình - 2324801030003
import 'package:flutter/material.dart';

import 'giao_dien/mau.dart';
import 'man_hinh/phan_tich.dart';

void main() => runApp(const UngDung());

class UngDung extends StatefulWidget {
  const UngDung({super.key});
  @override
  State<UngDung> createState() => _UngDungState();
}

class _UngDungState extends State<UngDung> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    cheDoGiaoDien.addListener(_doi);
    _capNhat();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    cheDoGiaoDien.removeListener(_doi);
    super.dispose();
  }

  @override
  void didChangePlatformBrightness() => _doi();

  void _capNhat() {
    final m = cheDoGiaoDien.value;
    Mau.toi = m == ThemeMode.dark || (m == ThemeMode.system && WidgetsBinding.instance.platformDispatcher.platformBrightness == Brightness.dark);
  }

  void _doi() {
    final cu = Mau.toi;
    _capNhat();
    if (cu == Mau.toi) return;
    setState(() {});
    WidgetsBinding.instance.addPostFrameCallback((_) {
      void veLai(Element e) {
        e.markNeedsBuild();
        e.visitChildren(veLai);
      }
      if (mounted) (context as Element).visitChildren(veLai);
    });
  }

  @override
  Widget build(BuildContext context) => MaterialApp(title: 'Biên bản kiểm tra lỗ hổng', debugShowCheckedModeBanner: false, theme: chuDe(), home: const ManHinhPhanTich());
}
