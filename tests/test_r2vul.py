# Nguyễn Tâm Bình - 2324801030003
import contextlib
import csv
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

from vuldet.data import r2vul
from vuldet.eval.languages import LANGUAGES, METRICS, append_language_result, language_rows
from vuldet.eval.metrics import binary_metrics
from vuldet.graphs import cpg_to_pyg, joern_export as joern
from vuldet.kg.build_kg import build
from vuldet.kg.export_gexf import export
from vuldet.train import baselines, lora_result
from vuldet.utils.checkpoint import SPLITS, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("tests").addHandler(logging.NullHandler())
logging.getLogger("tests").propagate = False


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def method(name="f", count=3, file="/tmp/r2vul_train_0.java"):
    return {"file": file, "name": name, "nodes": [[i, "CALL", "f()", 1] for i in range(count)], "edges": []}


class JoernTests(unittest.TestCase):
    def test_wrap_languages(self):
        for lang in ("java", "c_sharp"):
            with self.subTest(lang=lang):
                self.assertEqual(joern.wrap_source("void f() {}", lang), "class W {\nvoid f() {}\n}")
        self.assertEqual(joern.wrap_source("int f() {}", "c"), "int f() {}")
        self.assertEqual(joern.wrap_source("    def f():\n        return 1\n", "python"), "def f():\n    return 1\n")
        with self.assertRaises(ValueError):
            joern.wrap_source("fn f() {}", "rust")

    def test_javascript_function_forms(self):
        for code in ("function f() {}", "  async function f() {}", "x = function() {}", "x: function() {}", "x = async function() {}", "const x = (a) => a", "let x = async a => a", "x: a => a", "(a) => a", "async (a) => a", "exports.f = function() {}"):
            with self.subTest(code=code):
                self.assertEqual(joern.wrap_source(code, "javascript"), code)
        for code in ("f(a) { return a; }", "async f(a) { return a; }", "get value() { return 1; }"):
            with self.subTest(code=code):
                self.assertEqual(joern.wrap_source(code, "javascript"), "class W {\n" + code + "\n}")

    def test_frontend_commands(self):
        settings = {"c2cpg_defines": ["A=", "F(x)=x"]}
        for lang, (name, _) in joern.FRONTENDS.items():
            with self.subTest(lang=lang):
                cmd = joern.frontend_command(Path("src"), Path("out.bin"), settings, lang)
                self.assertEqual(cmd[0], "c2cpg.sh" if lang == "c" else f"/opt/joern-cli/frontends/{name}/bin/{name}")
                self.assertIn("-J-Xmx3g", cmd)
                self.assertEqual(cmd[2:5], ["src", "--output", "out.bin"])
                self.assertEqual(cmd[5:], ["--define", "A=", "--define", "F(x)=x"] if lang == "c" else [])

    def test_select_filters_and_largest_method(self):
        skipped = ("<global>", "<module>", "<init>", "<clinit>", "<lambda>", "<lambda>42", "<metaClassAdapter>")
        for lang in LANGUAGES:
            names = skipped + (("W", "W.<init>", "W..ctor") if lang in ("java", "c_sharp", "javascript") else ())
            items = [method(name, 30) for name in names] + [method("small", 2), method("large", 7), method("empty", 0), method("outside", 50, "/tmp/other.java")]
            with self.subTest(lang=lang):
                self.assertEqual(joern.select_methods(items, {"r2vul_train_0"}, lang)["r2vul_train_0"]["nodes"], method(count=7)["nodes"])

    def test_windows_filename_and_tie(self):
        item = method("f", 4, "C:\\temp\\r2vul_train_0.cs")
        self.assertEqual(joern.select_methods([item, method("g", 4)], {"r2vul_train_0"}, "c_sharp")["r2vul_train_0"]["nodes"], item["nodes"])

    def test_grouping_cpp_recovery_and_frontend_failure(self):
        batch = [{"id": lang, "lang": lang, "func": "f"} for lang in LANGUAGES]
        calls = []
        def fake(dataset, settings, logger, temporary, group, extension, lang, attempts=2):
            calls.append((lang, extension, [item["id"] for item in group]))
            self.assertTrue(all(item["lang"] == lang for item in group))
            if lang == "java":
                raise subprocess.TimeoutExpired("java", 1)
            return {} if lang == "c" and extension == "c" else {item["id"]: {"id": item["id"]} for item in group}
        with patch.object(joern, "export_pass", side_effect=fake):
            result, failed, cpp = joern.export_languages({}, logging.getLogger("tests"), Path("tmp"), batch)
        self.assertEqual(set(result), set(LANGUAGES) - {"java"})
        self.assertEqual(failed, [])
        self.assertEqual(cpp, 1)
        self.assertIn(("c", "cpp", ["c"]), calls)

    def test_split_drops_only_broken_function(self):
        group = [{"id": f"r2vul_train_{k}", "lang": "c_sharp", "func": "f"} for k in range(5)]
        def fake(dataset, settings, logger, temporary, items, extension, lang, attempts=2):
            if any(item["id"] == "r2vul_train_3" for item in items):
                raise subprocess.CalledProcessError(1, "joern")
            return {item["id"]: {"id": item["id"]} for item in items}
        with patch.object(joern, "export_pass", side_effect=fake):
            found, bad = joern.export_split({}, logging.getLogger("tests"), Path("tmp"), group, "cs", "c_sharp")
        self.assertEqual(bad, ["r2vul_train_3"])
        self.assertEqual(set(found), {f"r2vul_train_{k}" for k in (0, 1, 2, 4)})

    def test_export_pass_wrap_and_retry(self):
        with tempfile.TemporaryDirectory() as folder, working_directory(folder):
            Path("results/logs").mkdir(parents=True)
            calls = []
            def fake(command, timeout, stream):
                calls.append(command)
                self.assertGreater(timeout, 0)
                if len(calls) == 1:
                    raise subprocess.CalledProcessError(1, command)
                if command[0] == "joern":
                    dump = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("outFile=")))
                    write_jsonl(dump, [method()])
            with patch.object(joern, "run_command", side_effect=fake):
                result = joern.export_pass("r2vul", {"timeout": 600}, logging.getLogger("tests"), Path("tmp"), [{"id": "r2vul_train_0", "func": "void f() {}"}], "java", "java")
            self.assertIn("r2vul_train_0", result)
            self.assertEqual(Path("tmp/java/r2vul_train_0.java").read_text(), "class W {\nvoid f() {}\n}")
            self.assertEqual(len(calls), 3)
            self.assertIn("src/vuldet/graphs/dump_cpg.sc", calls[-1])

    def test_multilingual_checkpoint_and_retry_missing(self):
        with tempfile.TemporaryDirectory() as folder, working_directory(folder):
            settings = {"graphs": "graphs", "interim": "interim", "tmp": "tmp", "batch_size": 2, "c2cpg_defines": []}
            rows = [{"id": f"r2vul_train_{i}", "func": "f", "lang": lang} for i, lang in enumerate(("java", "python", "c"))]
            for split in SPLITS:
                write_jsonl(Path(f"interim/{split}.jsonl"), rows if split == "train" else [])
            def fake(settings, logger, temporary, batch):
                return {item["id"]: {"id": item["id"], "nodes": [[1, "METHOD", "f", 1]], "edges": []} for item in batch if item["lang"] != "python"}, [], 0
            with patch.object(joern, "config", return_value=settings), patch.object(joern, "fingerprint", return_value="same"), patch.object(joern, "get_logger", return_value=logging.getLogger("tests")), patch.object(joern, "export_languages", side_effect=fake) as exporter:
                joern.export_multilingual()
                self.assertEqual(exporter.call_count, 2)
                joern.export_multilingual()
                self.assertEqual(exporter.call_count, 2)
                self.assertEqual(Path("graphs/missing.txt").read_text().splitlines(), ["r2vul_train_1"])
                exporter.side_effect = lambda settings, logger, temporary, batch: ({item["id"]: {"id": item["id"], "nodes": [], "edges": []} for item in batch}, [], 0)
                joern.export_multilingual(True)
                self.assertEqual(exporter.call_count, 3)
                self.assertEqual([item["id"] for item in exporter.call_args.args[-1]], ["r2vul_train_1"])
                self.assertEqual([item["id"] for item in read_jsonl("graphs/raw/batch_00000.jsonl")], ["r2vul_train_0", "r2vul_train_1"])
                self.assertEqual(Path("graphs/missing.txt").read_text(), "")


class DataTests(unittest.TestCase):
    def fixture(self, folder, file_format=False):
        source, raw = Path(folder) / "source", Path(folder) / "raw"
        for split in SPLITS:
            originals = [{"lang": lang, "function": f"code {lang}", "vulnerable": label, **{key: None for key in r2vul.META_FIELDS}, "cve_id": "CVE-1", "cwe_id": ["CWE-79"], "origin": "vuln" if label else "unchanged", "repo": "repo", "commit_URL": "url", "parent_commit_sha": "sha", "date": "date", "file": "file"} for lang in LANGUAGES for label in (0, 1)]
            originals += [{**originals[0], "file": "second"}]
            rows = [{"func": row["function"], "target": row["vulnerable"], "lang": row["lang"], "cve_id": row["cve_id"], "origin": row["origin"]} for row in originals]
            write_jsonl(source / f"{split}.jsonl", rows)
            path = raw / ("validation" if split == "valid" else split) / "data.arrow"
            path.parent.mkdir(parents=True)
            table = pa.Table.from_pylist(originals)
            with pa.OSFile(str(path), "wb") as sink, (pa.ipc.new_file if file_format else pa.ipc.new_stream)(sink, table.schema) as writer:
                writer.write_table(table)
        return source, raw

    def test_arrow_stream_order_duplicate_metadata_and_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            source, raw = self.fixture(folder)
            output, stdout = Path(folder) / "output", io.StringIO()
            with contextlib.redirect_stdout(stdout):
                counts = r2vul.prepare(source, raw, output)
            for split in SPLITS:
                rows = list(read_jsonl(output / f"{split}.jsonl"))
                self.assertEqual([row["id"] for row in rows], [f"r2vul_{split}_{k}" for k in range(11)])
                self.assertEqual(rows[-1]["meta"]["file"], "second")
                self.assertEqual(rows[0]["meta"]["file"], "file")
                self.assertEqual(set(rows[0]["meta"]), set(r2vul.META_FIELDS))
                self.assertTrue(all(row["split"] == split and isinstance(row["meta"]["cwe_id"], list) for row in rows))
                self.assertEqual(sum(value for (s, _, _), value in counts.items() if s == split), 11)
            self.assertIn("valid\tc_sharp\t1\t1\t2", stdout.getvalue())

    def test_arrow_file_format(self):
        with tempfile.TemporaryDirectory() as folder:
            source, raw = self.fixture(folder, True)
            self.assertEqual(len(list(r2vul.split_records(source, raw, "valid"))), 11)

    def test_missing_or_extra_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source, raw = self.fixture(folder)
            rows = list(read_jsonl(source / "train.jsonl"))
            write_jsonl(source / "train.jsonl", rows[:-1])
            with self.assertRaisesRegex(ValueError, "thiếu"):
                list(r2vul.split_records(source, raw, "train"))
            rows[0]["func"] = "unknown code"
            write_jsonl(source / "train.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Không khớp"):
                list(r2vul.split_records(source, raw, "train"))

    def test_node_vocabulary_multilingual(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.jsonl"
            write_jsonl(path, [{"nodes": [[1, "NEW_FRONTEND_LABEL", "", 1], [2, "KEY_VALUE_PAIR", "", 1]]}])
            labels = cpg_to_pyg.node_labels("r2vul", [path])
            self.assertEqual(labels[:40], cpg_to_pyg.NODE_LABELS)
            self.assertIn("NEW_FRONTEND_LABEL", labels)
            self.assertIn("FINDING", labels)


class LanguageTests(unittest.TestCase):
    def sample(self):
        return np.tile([0, 1], 5), np.tile([0.1, 0.4], 5), np.repeat(LANGUAGES, 2)

    def test_threshold_on_valid_and_probability_half(self):
        y, scores, langs = self.sample()
        rows = language_rows(y, scores, langs, y, scores, langs, fixed_threshold=0.5)
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(row["f1"], 1.0)
            self.assertEqual(row["accuracy_05"], 0.5)
            self.assertEqual(row["f1_05"], 0.0)
            self.assertEqual(row["auroc"], 1.0)
            self.assertEqual(row["auprc"], 1.0)
        self.assertEqual(rows[0]["threshold"], 0.4)
        changed = language_rows(y, scores, langs, 1 - y, scores, langs, fixed_threshold=0.5)
        self.assertEqual([row["threshold"] for row in rows], [row["threshold"] for row in changed])

    def test_logit_zero_equals_probability_half(self):
        y, probabilities, langs = self.sample()
        logits = np.log(probabilities / (1 - probabilities))
        probability_rows = language_rows(y, probabilities, langs, y, probabilities, langs, fixed_threshold=0.5)
        logit_rows = language_rows(y, logits, langs, y, logits, langs)
        for first, second in zip(probability_rows, logit_rows):
            self.assertEqual([first[key] for key in METRICS], [second[key] for key in METRICS])

    def test_macro_is_unweighted_five_languages(self):
        y, scores, langs = self.sample()
        test_y = np.r_[y, [0] * 10]
        test_scores = np.r_[scores, [0.8] * 10]
        test_langs = np.r_[langs, ["c"] * 10]
        rows = language_rows(y, scores, langs, test_y, test_scores, test_langs)
        self.assertAlmostEqual(rows[5]["accuracy"], np.mean([row["accuracy"] for row in rows[:5]]))
        self.assertNotAlmostEqual(rows[5]["accuracy"], binary_metrics(test_y, test_scores, rows[0]["threshold"])["accuracy"])

    def test_missing_language_or_single_class_is_explicit(self):
        y, scores, langs = self.sample()
        rows = language_rows(y[:-1], scores[:-1], langs[:-1], y[:-2], scores[:-2], langs[:-2])
        self.assertTrue(np.isnan(rows[4]["f1"]))
        self.assertTrue(np.isnan(rows[5]["f1"]))
        self.assertEqual(rows[10]["nguon_nguong"], "valid_chung")

    def test_invalid_input_rejected(self):
        y, scores, langs = self.sample()
        for values in ((y, scores[:-1], langs), (y, scores * np.nan, langs), (y, scores, ["rust"] * 10), (y + 2, scores, langs)):
            with self.subTest(values=str(values)[:40]), self.assertRaises(ValueError):
                language_rows(*values, y, scores, langs)

    def test_csv_write_replace_and_alignment(self):
        y, scores, langs = self.sample()
        with tempfile.TemporaryDirectory() as folder:
            interim, output = Path(folder) / "interim", Path(folder) / "table.csv"
            ids = {split: [f"r2vul_{split}_{i}" for i in range(10)] for split in ("valid", "test")}
            for split in ids:
                write_jsonl(interim / f"{split}.jsonl", [{"id": ident, "target": int(label), "lang": str(lang)} for ident, label, lang in zip(ids[split], y, langs)])
            args = (ids["valid"], y, scores, ids["test"], y, scores)
            for model in ("A", "B", "A"):
                append_language_result("r2vul", {"mo_hinh": model, "seed": 42}, *args, output=output, interim=interim)
            with output.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 24)
            self.assertEqual(sum(row["lang"] == "macro_5" for row in rows), 4)
            with self.assertRaisesRegex(ValueError, "khớp"):
                append_language_result("r2vul", {}, ids["valid"], 1 - y, scores, *args[3:], output=output, interim=interim)

    def test_lora_ids_and_language_validation(self):
        with tempfile.TemporaryDirectory() as folder, working_directory(folder):
            write_jsonl(Path("data/interim/r2vul/valid.jsonl"), [{"id": "r2vul_valid_0", "target": 1, "lang": "c_sharp"}])
            path = Path("pred.csv")
            path.write_text("id,lang,target,prob\nr2vul5_valid_0,c_sharp,1,0.8\n", encoding="utf-8")
            ids, y, logits, n = lora_result.read_predictions(path, "r2vul", "valid")
            self.assertEqual(ids.tolist(), ["r2vul_valid_0"])
            self.assertAlmostEqual(logits[0], np.log(4))
            self.assertEqual(n, 1)
            path.write_text("id,lang,target,prob\nr2vul5_valid_0,java,1,0.8\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ngôn ngữ"):
                lora_result.read_predictions(path, "r2vul", "valid")

    def test_r2vul_baselines_include_predictions_without_subsampling(self):
        rows = {split: ([f"r2vul_{split}_{i}" for i in range(10)], np.tile([0, 1], 5), []) for split in SPLITS}
        with patch.object(baselines, "load_interim", side_effect=lambda ds, split: rows[split]), patch.object(baselines, "graph_id_sets", return_value={split: set(rows[split][0]) for split in SPLITS}), patch.object(baselines, "append_result") as append:
            baselines.run("r2vul", 42)
        self.assertEqual(append.call_count, 6)
        self.assertTrue(all(call.args[0]["n_train"] == 10 and len(call.args[1][0]) == 10 for call in append.call_args_list))


class KnowledgeGraphTests(unittest.TestCase):
    def test_r2vul_metadata_api_and_gephi(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in SPLITS:
                write_jsonl(root / "interim" / f"{split}.jsonl", [{"id": f"r2vul_{split}_0", "split": split, "target": 1, "lang": "python", "meta": {"cwe_id": ["CWE-79"], "cve_id": "CVE-1", "repo": "repo", "commit_URL": "url", "parent_commit_sha": "sha"}}])
            write_jsonl(root / "raw/batch_00000.jsonl", [{"id": f"r2vul_{split}_0", "nodes": [[1, "CALL", "await obj.run(x)", 1]], "edges": []} for split in SPLITS])
            with contextlib.redirect_stdout(io.StringIO()):
                build(root / "interim", root / "raw", root / "kg", root / "stats.json", 1, "r2vul")
                export(root / "kg/r2vul_kg.pkl", root / "gephi")
            stats = json.loads((root / "stats.json").read_text())
            self.assertEqual(stats["nodes_by_kind"], {"function": 3, "language": 1, "cve": 1, "commit": 2, "project": 1, "cwe": 1, "api": 1})
            self.assertEqual(stats["edges_by_rel"]["calls"], 3)
            self.assertTrue((root / "gephi/kg_overview.gexf").exists())


class ShellTests(unittest.TestCase):
    def run_chain(self, failed=""):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("Máy không có Bash")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = (ROOT / "scripts/run_r2vul.sh").read_text(encoding="utf-8").replace("cd /work", f'cd "{root.as_posix()}"')
            shim = '''python() {
    if [[ "${1:-}" == "-" ]]; then cat >/dev/null; return 0; fi
    printf '%s\\n' "$*" >> calls.txt
    if [[ "$*" == *"joern_export"* ]]; then mkdir -p data/graphs/r2vul; printf 'missing\\n' > data/graphs/r2vul/missing.txt; fi
    if [[ -n "$FAIL_STEP" && "$*" == *"$FAIL_STEP"* ]]; then return 7; fi
    return 0
}
'''
            path = root / "run.sh"
            path.write_text(script.replace("set -Eeuo pipefail", "set -Eeuo pipefail\n" + shim), encoding="utf-8", newline="\n")
            syntax = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            result = subprocess.run([bash, str(path)], env={**os.environ, "FAIL_STEP": failed}, capture_output=True, text=True, encoding="utf-8")
            return result, (root / "results/logs/chuoi_r2vul.txt").read_text(encoding="utf-8"), (root / "calls.txt").read_text(encoding="utf-8")

    def test_chain_order_status_and_colab_reminder(self):
        result, status, calls = self.run_chain()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([line.split(" XONG ")[0] for line in status.splitlines()[:-1]], ["joern_export", "retry_missing", "cpg_to_pyg", "embed_unixcoder_func_cpu", "kg"])
        self.assertEqual(status.splitlines()[-1], "Cần chạy Colab: cấp nút + Qwen cấp hàm")
        self.assertIn("--retry-missing", calls)
        self.assertNotIn("--mode node", calls)
        self.assertNotIn("qwen", calls)

    def test_chain_failure_logs_and_stops(self):
        result, status, calls = self.run_chain("cpg_to_pyg")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(status.splitlines()[-1], "cpg_to_pyg LOI")
        self.assertNotIn("vuldet.llm.embed", calls)
        self.assertNotIn("Cần chạy Colab", status)




if __name__ == "__main__":
    unittest.main()
