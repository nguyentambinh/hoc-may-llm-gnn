# Nguyễn Tâm Bình - 2324801030003
set -Eeuo pipefail
cd /work
export PYTHONPATH=/work/src
export EMBED_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/logs
trang_thai=results/logs/chuoi_r2vul.txt
buoc=khoi_tao
trap 'printf "%s LOI\n" "$buoc" >> "$trang_thai"' ERR

xong() {
    printf '%s XONG %s\n' "$buoc" "$(date '+%Y-%m-%d %H:%M:%S%z')" >> "$trang_thai"
}

buoc=joern_export
python -m vuldet.graphs.joern_export --dataset r2vul 2>&1 | tee -a results/logs/run_r2vul.log
xong

buoc=retry_missing
if [[ -s data/graphs/r2vul/missing.txt ]]; then
    python -m vuldet.graphs.joern_export --dataset r2vul --retry-missing 2>&1 | tee -a results/logs/run_r2vul.log
fi
python - <<'PY'
import json
from pathlib import Path
failed = [path.name for path in Path('data/graphs/r2vul/raw').glob('batch_*.done') if json.loads(path.read_text()).get('failed')]
if failed:
    raise SystemExit(f'Joern vẫn lỗi công cụ sau retry: {failed}')
PY
xong

buoc=cpg_to_pyg
python -m vuldet.graphs.cpg_to_pyg --dataset r2vul 2>&1 | tee -a results/logs/run_r2vul.log
xong

buoc=embed_unixcoder_func_cpu
python -m vuldet.llm.embed --dataset r2vul --mode func --model unixcoder 2>&1 | tee -a results/logs/run_r2vul.log
xong

buoc=kg
python -m vuldet.kg.build_kg --dataset r2vul 2>&1 | tee -a results/logs/run_r2vul.log
python -m vuldet.kg.export_gexf --dataset r2vul 2>&1 | tee -a results/logs/run_r2vul.log
xong
printf '%s\n' 'Cần chạy Colab: cấp nút + Qwen cấp hàm' >> "$trang_thai"
