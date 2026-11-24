// Nguyễn Tâm Bình - 2324801030003
import 'package:flutter/material.dart';

final cheDoGiaoDien = ValueNotifier<ThemeMode>(ThemeMode.system);

class Mau {
  static bool toi = false;
  static Color _c(int sang, int maToi) => Color(toi ? maToi : sang);

  static Color get nen => _c(0xFFF2F5F9, 0xFF0E1424);
  static Color get mat => _c(0xFFFFFFFF, 0xFF161E31);
  static Color get muc => _c(0xFF16244B, 0xFFE4E9F4);
  static Color get muc2 => _c(0xFF475681, 0xFFB1BAD1);
  static Color get muc3 => _c(0xFF586487, 0xFF8F9AB6);
  static Color get ke => _c(0xFFC5CEDF, 0xFF2F3B57);
  static Color get keNhat => _c(0xFFE4E9F2, 0xFF222B42);
  static Color get dau => _c(0xFFBF1F2C, 0xFFF0646D);
  static Color get dauNhat => _c(0xFFFBECEC, 0xFF3A1E26);
  static Color get dat => _c(0xFF1D6B43, 0xFF4CC38A);
  static Color get datNhat => _c(0xFFE8F3EC, 0xFF143026);
  static Color get hoPhach => _c(0xFF9A5600, 0xFFF0A54A);
  static Color get hoPhachNhat => _c(0xFFFBF1E3, 0xFF352816);
  static Color get chon => _c(0xFFDCE4F4, 0xFF27365A);
  static Color get nut => _c(0xFF16244B, 0xFF4F6BD8);
  static Color get chuTrenMau => _c(0xFFFFFFFF, 0xFF0E1424);
}

class Chu {
  static const sans = 'BeVietnamPro';
  static const mono = 'JetBrainsMono';
  static const so = [FontFeature.tabularFigures()];

  static TextStyle ma({double co = 15, Color? mau, FontWeight dam = FontWeight.w400}) => TextStyle(fontFamily: mono, fontSize: co, height: 1.0, color: mau ?? Mau.muc, fontWeight: dam, fontFeatures: so);
  static TextStyle nhan({double co = 13, Color? mau, FontWeight dam = FontWeight.w600}) => TextStyle(fontFamily: sans, fontSize: co, color: mau ?? Mau.muc2, fontWeight: dam, letterSpacing: 0.4);
}

ThemeData chuDe() {
  final goc = ThemeData(useMaterial3: true, brightness: Mau.toi ? Brightness.dark : Brightness.light, fontFamily: Chu.sans, colorScheme: ColorScheme.fromSeed(seedColor: Mau.nut, brightness: Mau.toi ? Brightness.dark : Brightness.light, surface: Mau.mat, primary: Mau.nut, onPrimary: Colors.white, onSurface: Mau.muc, error: Mau.dau));
  return goc.copyWith(
    scaffoldBackgroundColor: Mau.nen,
    textTheme: goc.textTheme.apply(bodyColor: Mau.muc, displayColor: Mau.muc),
    dividerTheme: DividerThemeData(color: Mau.ke, thickness: 1, space: 1),
    textSelectionTheme: TextSelectionThemeData(cursorColor: Mau.muc, selectionColor: Mau.chon, selectionHandleColor: Mau.muc),
    scrollbarTheme: ScrollbarThemeData(thumbColor: WidgetStateProperty.all(Mau.ke), thickness: WidgetStateProperty.all(8), radius: const Radius.circular(2)),
    filledButtonTheme: FilledButtonThemeData(style: FilledButton.styleFrom(backgroundColor: Mau.nut, foregroundColor: Colors.white, disabledBackgroundColor: Mau.ke, minimumSize: const Size(136, 44), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)), textStyle: const TextStyle(fontFamily: Chu.sans, fontSize: 15, fontWeight: FontWeight.w600))),
    outlinedButtonTheme: OutlinedButtonThemeData(style: OutlinedButton.styleFrom(foregroundColor: Mau.muc, side: BorderSide(color: Mau.ke), minimumSize: const Size(0, 40), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)), textStyle: const TextStyle(fontFamily: Chu.sans, fontSize: 14, fontWeight: FontWeight.w500))),
    textButtonTheme: TextButtonThemeData(style: TextButton.styleFrom(foregroundColor: Mau.muc)),
    inputDecorationTheme: InputDecorationTheme(labelStyle: TextStyle(color: Mau.muc2), hintStyle: TextStyle(color: Mau.muc3)),
    dropdownMenuTheme: DropdownMenuThemeData(textStyle: TextStyle(color: Mau.muc)),
    canvasColor: Mau.mat,
    tooltipTheme: TooltipThemeData(decoration: BoxDecoration(color: Mau.muc, borderRadius: BorderRadius.circular(3)), textStyle: TextStyle(color: Mau.mat, fontSize: 13)),
    focusColor: Mau.chon,
  );
}
