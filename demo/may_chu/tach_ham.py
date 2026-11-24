# Nguyễn Tâm Bình - 2324801030003
import tree_sitter_c
import tree_sitter_c_sharp
import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language, Parser

NGON_NGU = {
    "c": Language(tree_sitter_c.language()),
    "cpp": Language(tree_sitter_cpp.language()),
    "c_sharp": Language(tree_sitter_c_sharp.language()),
    "java": Language(tree_sitter_java.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "python": Language(tree_sitter_python.language()),
}
TEN_NGON_NGU = {"c": "C", "c_sharp": "C#", "java": "Java", "javascript": "JavaScript", "python": "Python"}
LOAI_HAM = {
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "c_sharp": {"method_declaration", "constructor_declaration", "local_function_statement", "operator_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "javascript": {"function_declaration", "generator_function_declaration", "method_definition"},
    "python": {"function_definition"},
}
DAU_HIEU = {
    "python": ("def ", "self", "elif ", "import ", "None", "True", "False"),
    "java": ("public ", "private ", "protected ", "throws ", "String ", "@Override", "new "),
    "c_sharp": ("public ", "private ", "internal ", "string ", "var ", "async Task", "=> ", "using "),
    "javascript": ("function", "const ", "let ", "=> ", "===", "require(", "undefined"),
    "c": ("#include", "->", "malloc", "sizeof", "NULL", "unsigned ", "struct "),
}


def _ten_c(nut, nguon):
    d = nut.child_by_field_name("declarator")
    while d is not None and d.type not in ("identifier", "field_identifier", "qualified_identifier", "destructor_name", "operator_name"):
        d = d.child_by_field_name("declarator") or (d.named_children[0] if d.named_children else None)
    return nguon[d.start_byte:d.end_byte].decode("utf-8", "replace") if d is not None else "(không rõ tên)"


def _ten(nut, nguon, ngon_ngu):
    if ngon_ngu in ("c", "cpp"):
        return _ten_c(nut, nguon)
    ten = nut.child_by_field_name("name")
    if ten is None and nut.parent is not None and nut.parent.type == "variable_declarator":
        ten = nut.parent.child_by_field_name("name")
    return nguon[ten.start_byte:ten.end_byte].decode("utf-8", "replace") if ten is not None else "(hàm ẩn danh)"


def _la_ham(nut, ngon_ngu):
    if nut.type in LOAI_HAM[ngon_ngu]:
        return True
    return ngon_ngu == "javascript" and nut.type in ("arrow_function", "function_expression", "function") and nut.parent is not None and nut.parent.type == "variable_declarator"


def _tach(nguon, ngon_ngu):
    cay = Parser(NGON_NGU[ngon_ngu]).parse(nguon)
    ham, ngan_xep = [], [cay.root_node]
    while ngan_xep:
        nut = ngan_xep.pop()
        if _la_ham(nut, ngon_ngu):
            ham.append(nut)
            continue
        ngan_xep.extend(reversed(nut.children))
    return ham, cay.root_node.has_error


def _ban_bao(nguon, ngon_ngu):

    if ngon_ngu in ("java", "c_sharp"):
        return b"class W {\n" + nguon + b"\n}", 1
    return nguon, 0


def doan_ngon_ngu(ma):

    diem = {}
    for nn in TEN_NGON_NGU:
        nguon, _ = _ban_bao(ma.encode("utf-8"), nn)
        ham, loi = _tach(nguon, nn)
        diem[nn] = (0 if loi else 2) + (1 if ham else 0) + 0.1 * sum(ma.count(t) for t in DAU_HIEU[nn])
    return max(diem, key=diem.get)


def tach_ham(ma, ngon_ngu):

    nguon = ma.encode("utf-8")
    goi, lech = _ban_bao(nguon, ngon_ngu)
    ham, loi = _tach(goi, ngon_ngu)
    if ngon_ngu == "c" and (not ham or loi):
        ham2, _ = _tach(nguon, "cpp")
        if len(ham2) >= len(ham):
            ham, goi, lech, ngon_ngu = ham2, nguon, 0, "cpp"
    if lech and not ham:
        ham, _ = _tach(nguon, ngon_ngu)
        goi, lech = nguon, 0
    if not ham:
        return [{"ten": "(đoạn mã)", "dong_bat_dau": 1, "ma": ma}]
    kq = []
    for nut in ham:
        dau = goi.rfind(b"\n", 0, nut.start_byte) + 1
        kq.append({"ten": _ten(nut, goi, ngon_ngu), "dong_bat_dau": nut.start_point[0] + 1 - lech, "ma": goi[dau:nut.end_byte].decode("utf-8", "replace")})
    return kq
