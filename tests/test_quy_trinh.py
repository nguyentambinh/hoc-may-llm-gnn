# Nguyễn Tâm Bình - 2324801030003
import ast
import importlib.util
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vuldet.eval import cai_thien_f1, phan_tich_gnn
from vuldet.graphs.cpg_to_pyg import make_graph

ROOT = Path(__file__).resolve().parents[1]


class QuyTrinhTests(unittest.TestCase):
    def test_dataset_choices(self):
        for path in (ROOT / 'src').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Call) and any(isinstance(arg, ast.Constant) and arg.value == '--dataset' for arg in node.args):
                    choices = next(kw.value for kw in node.keywords if kw.arg == 'choices')
                    self.assertEqual(ast.literal_eval(choices), ('r2vul',), str(path))

    def test_stacking_threshold_uses_validation(self):
        rng = np.random.default_rng(42)
        y = np.tile([0, 1], 30)
        valid = np.column_stack([y + rng.normal(0, 0.4, len(y)), y + rng.normal(0, 0.8, len(y))])
        test = valid[:20]
        reference = {'valid_y': y, 'test_y': y[:20], 'test_ids': np.arange(20)}
        oof, scores, c, auprc, weights = cai_thien_f1.stacking(reference, valid, test)
        self.assertEqual(oof.shape, y.shape)
        self.assertEqual(scores.shape, (20,))
        self.assertEqual(weights.shape, (2,))
        self.assertGreater(auprc, 0.7)
        first = cai_thien_f1.both_thresholds(reference, oof, scores)
        changed = cai_thien_f1.both_thresholds({**reference, 'test_y': 1 - y[:20]}, oof, scores)
        self.assertEqual({k: v['threshold'] for k, v in first.items()}, {k: v['threshold'] for k, v in changed.items()})

    def test_analysis_metrics_and_bootstrap(self):
        y = np.tile([0, 1], 10)
        probabilities = np.where(y, 0.8, 0.2)
        metrics = phan_tich_gnn.evaluate(y, probabilities, y, probabilities)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
        delta = phan_tich_gnn.bootstrap(y, probabilities, probabilities, rounds=20)
        self.assertEqual(delta['delta_auroc'], 0.0)
        self.assertEqual(delta['ci_thap'], 0.0)

    @unittest.skipUnless(importlib.util.find_spec('torch_geometric'), 'Thiếu torch_geometric: chưa kiểm tra dựng đối tượng PyG thật')
    def test_pyg_duplicate_nodes_and_edges(self):
        with sqlite3.connect(':memory:') as database:
            database.execute('CREATE TABLE codes (id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE)')
            item = {'id': 'r2vul_test_0', 'nodes': [[1, 'METHOD', 'f', 1], [1, 'METHOD', 'f', 1], [2, 'CALL', 'g()', 2]], 'edges': [[1, 2, 'AST'], [1, 99, 'CFG']]}
            graph = make_graph(item, 1, 'test', database, 500)
            self.assertEqual(graph.num_nodes, 2)
            self.assertEqual(graph.edge_index.tolist(), [[0], [1]])
            self.assertEqual(graph.edge_type.tolist(), [0])

    def run_models(self, fail=''):
        bash = shutil.which('bash')
        if not bash:
            self.skipTest('Không có Bash')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['data/embeddings/r2vul/func_test.done', 'data/embeddings/r2vul/node_codes.npy', 'results/lora/r2vul5/pred_valid.csv', 'results/lora/r2vul5/pred_test.csv', *[f'data/embeddings/r2vul/qwen15/func_{split}.npy' for split in ('train', 'valid', 'test')]]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture', encoding='utf-8')
            script = (ROOT / 'scripts/run_r2vul_mo_hinh.sh').read_text(encoding='utf-8').replace('cd /work', f'cd "{root.as_posix()}"')
            shim = '''python() {
    printf '%s\n' "$*" >> calls.txt
    if [[ -n "$FAIL_STEP" && "$*" == *"$FAIL_STEP"* ]]; then return 7; fi
}
nice() { shift 2; "$@"; }
'''
            script = script.replace('set -Eeuo pipefail', 'set -Eeuo pipefail\n' + shim)
            path = root / 'models.sh'
            path.write_text(script, encoding='utf-8', newline='\n')
            result = subprocess.run([bash, str(path)], env={**os.environ, 'FAIL_STEP': fail}, capture_output=True, text=True, encoding='utf-8')
            return result, (root / 'calls.txt').read_text(), (root / 'results/logs/chuoi_r2vul.txt').read_text()

    def test_models_run_all_groups(self):
        result, calls, status = self.run_models()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count('vuldet.train.gnn_sup'), 9)
        self.assertEqual(calls.count('vuldet.train.llm_only'), 3)
        self.assertIn('vuldet.train.lora_result', calls)
        self.assertIn('vuldet.train.ensemble', calls)
        self.assertTrue(calls.splitlines()[-1].startswith('-m vuldet.eval.cai_thien_f1'))
        self.assertIn('L11_L13_ket_hop XONG', status)

    def test_models_stop_on_failure(self):
        result, calls, status = self.run_models('vuldet.train.llm_only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('LOI vuldet.train.llm_only', status)
        self.assertNotIn('vuldet.train.ensemble', calls)


if __name__ == '__main__':
    unittest.main()
