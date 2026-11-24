# Nguyễn Tâm Bình - 2324801030003
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import textwrap
import time
from pathlib import Path

from vuldet.utils.checkpoint import SPLITS, batches, config, fingerprint, guard, read_jsonl, records, write_json
from vuldet.utils.log import get_logger

FRONTENDS = {"c": ("c2cpg", "c"), "java": ("javasrc2cpg", "java"), "c_sharp": ("csharpsrc2cpg", "cs"), "python": ("pysrc2cpg", "py"), "javascript": ("jssrc2cpg", "js")}
JS_FUNCTION = re.compile(r"^(?:async\s+)?function\b|^(?:(?:export\s+)?(?:const|let|var)\s+)?[\w$.[\]'\"]+\s*(?:=|:)\s*(?:(?:async\s+)?function\b|(?:async\s+)?(?:\([^)]*\)|[\w$]+)\s*=>)|^(?:async\s+)?(?:\([^)]*\)|[\w$]+)\s*=>", re.DOTALL)


def wrap_source(code, lang):
    if lang not in FRONTENDS:
        raise ValueError(f"Unsupported language: {lang}")
    if lang == "python":
        return textwrap.dedent(code)
    if lang in ("java", "c_sharp") or (lang == "javascript" and not JS_FUNCTION.match(code.lstrip())):
        return "class W {\n" + code + "\n}"
    return code


def frontend_command(source, cpg, settings, lang=None):
    if lang is None or lang == "c":
        return ["c2cpg.sh", "-J-Xmx3g", str(source), "--output", str(cpg), *define_arguments(settings["c2cpg_defines"])]
    frontend = FRONTENDS[lang][0]
    return [f"/opt/joern-cli/frontends/{frontend}/bin/{frontend}", "-J-Xmx3g", str(source), "--output", str(cpg)]


def select_methods(items, ids, lang=None):
    methods = {}
    for method in items:
        name = method.get("name", "")
        if lang is not None and (name in ("<global>", "<module>", "<init>", "<clinit>", "<metaClassAdapter>") or name.startswith("<lambda>") or (lang in ("java", "c_sharp", "javascript") and name in ("W", "W.<init>", "W..ctor"))):
            continue
        record_id = Path(method["file"].replace("\\", "/")).stem
        if record_id in ids and method.get("nodes") and (record_id not in methods or len(method["nodes"]) > len(methods[record_id]["nodes"])):
            methods[record_id] = {"id": record_id, "nodes": method["nodes"], "edges": method["edges"]}
    return methods


def define_arguments(defines):
    return [argument for define in defines for argument in ("--define", str(define))]


def run_command(command, timeout, stream):
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except BaseException:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait()
        raise


def export_pass(dataset, settings, logger, temporary, batch, extension, lang=None, attempts=2):
    if not batch:
        return {}
    source = temporary / extension
    source.mkdir(parents=True)
    ids = {item["id"] for item in batch}
    for item in batch:
        if Path(item["id"]).name != item["id"] or "/" in item["id"] or "\\" in item["id"]:
            raise ValueError("Invalid source id")
        (source / f"{item['id']}.{extension}").write_text(wrap_source(item["func"], lang) if lang is not None else item["func"], encoding="utf-8")
    cpg = temporary / f"{extension}.bin"
    dump = temporary / f"{extension}.jsonl"
    for attempt in range(attempts):
        deadline = time.monotonic() + float(settings["timeout"])
        try:
            with open(f"results/logs/joern_{dataset}_tool.log", "a", encoding="utf-8") as stream:
                run_command(frontend_command(source, cpg, settings, lang), max(0.1, deadline - time.monotonic()), stream)
                run_command(["joern", "-J-Xmx3g", "--script", "src/vuldet/graphs/dump_cpg.sc", "--param", f"cpgFile={cpg}", "--param", f"outFile={dump}"], max(0.1, deadline - time.monotonic()), stream)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.exception("%s attempt=%d failed", source.name, attempt + 1)
            if attempt == attempts - 1:
                raise
            cpg.unlink(missing_ok=True)
            dump.unlink(missing_ok=True)
    return select_methods(read_jsonl(dump), ids, lang)


def export_split(settings, logger, temporary, group, extension, lang, attempts=2):

    try:
        return export_pass("r2vul", settings, logger, temporary, group, extension, lang, attempts), []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if len(group) == 1:
            logger.warning("%s bỏ hàm làm hỏng CPG: %s", lang, group[0]["id"])
            return {}, [group[0]["id"]]
    mid = len(group) // 2
    left, bad_left = export_split(settings, logger, temporary / "a", group[:mid], extension, lang, 1)
    right, bad_right = export_split(settings, logger, temporary / "b", group[mid:], extension, lang, 1)
    return {**left, **right}, bad_left + bad_right


def export_languages(settings, logger, temporary, batch):
    methods, failed, recovered_cpp = {}, [], 0
    unknown = {item.get("lang") for item in batch} - FRONTENDS.keys()
    if unknown:
        raise ValueError(f"Unsupported languages: {unknown}")
    for lang, (_, extension) in FRONTENDS.items():
        group = [item for item in batch if item["lang"] == lang]
        if not group:
            continue
        found, _ = export_split(settings, logger, temporary / lang, group, extension, lang)
        methods.update(found)
        missing = [item for item in group if item["id"] not in methods]
        if lang == "c" and missing:
            recovered, _ = export_split(settings, logger, temporary / lang / "cpp_pass", missing, "cpp", lang)
            methods.update(recovered)
            recovered_cpp += len(recovered)
    return methods, failed, recovered_cpp


def export_multilingual(retry_missing=False):
    settings = config("r2vul")
    logger = get_logger("joern_r2vul")
    root = Path(settings["graphs"])
    output = root / "raw"
    output.mkdir(parents=True, exist_ok=True)
    signature = fingerprint([*(Path(settings["interim"]) / f"{split}.jsonl" for split in SPLITS), "src/vuldet/graphs/dump_cpg.sc", "src/vuldet/graphs/joern_export.py"], {key: settings[key] for key in ("batch_size", "c2cpg_defines")})
    guard(output / "input.json", signature)
    for index, batch in enumerate(batches(records(settings), int(settings["batch_size"]))):
        name = f"batch_{index:05d}"
        target, done = output / f"{name}.jsonl", output / f"{name}.done"
        status = json.loads(done.read_text(encoding="utf-8")) if done.exists() else {}
        if done.exists() and not status.get("failed") and (not retry_missing or not status.get("missing")):
            continue
        if retry_missing and not done.exists():
            continue
        methods = {item["id"]: item for item in read_jsonl(target)} if done.exists() else {}
        ids = {item["id"] for item in batch}
        if len(ids) != len(batch) or not methods.keys() <= ids:
            raise ValueError("Invalid ids in Joern batch")
        temporary = Path(settings["tmp"]) / "r2vul" / name
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        start = time.monotonic()
        try:
            pending = [item for item in batch if item["id"] not in methods]
            logger.info("%s started count=%d pending=%d", name, len(batch), len(pending))
            recovered, failed, recovered_cpp = export_languages(settings, logger, temporary, pending)
            methods.update(recovered)
            partial = target.with_suffix(".jsonl.tmp")
            with partial.open("w", encoding="utf-8") as stream:
                for item in batch:
                    if item["id"] in methods:
                        stream.write(json.dumps(methods[item["id"]], ensure_ascii=False) + "\n")
            partial.replace(target)
            write_json(done, {"missing": sorted(ids - methods.keys()), "count": len(methods), "cpp_pass": True, "recovered_cpp": status.get("recovered_cpp", 0) + recovered_cpp, "failed": bool(failed), "failed_languages": failed})
            logger.info("%s count=%d missing=%d seconds=%.2f", name, len(methods), len(ids) - len(methods), time.monotonic() - start)
        finally:
            shutil.rmtree(temporary)
            update_missing(root)
    update_missing(root)




def export(dataset, retry_missing=False):
    return export_multilingual(retry_missing)


def update_missing(root):
    temporary = root / "missing.txt.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for path in sorted((root / "raw").glob("batch_*.done")):
            for record_id in json.loads(path.read_text(encoding="utf-8"))["missing"]:
                stream.write(record_id + "\n")
    temporary.replace(root / "missing.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("r2vul",))
    parser.add_argument("--retry-missing", action="store_true")
    args = parser.parse_args()
    export(args.dataset, args.retry_missing)
