#!/bin/bash
# Nguyễn Tâm Bình - 2324801030003
set -Eeuo pipefail
cd /work
export PYTHONPATH=src PYTHONWARNINGS=ignore OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p results/logs
ST=results/logs/chuoi_r2vul.txt
LOG=results/logs/run_r2vul_mo_hinh.log
step() {
    echo "$(date '+%F %T') CHAY $*" >> $LOG
    if nice -n 19 python -m "$@" >> $LOG 2>&1; then return 0; fi
    printf 'LOI %s\n' "$*" >> $ST; return 1
}
xong() { printf '%s XONG %s\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S%z')" >> $ST; }
cho() { [ -s "$1" ] || { printf "THIEU %s\n" "$1" >&2; return 1; }; }

for s in 42 43 44; do step vuldet.train.gnn_sup --dataset r2vul --mode gnn --seed $s --no-report; done
step vuldet.train.baselines --dataset r2vul
xong L8_rgcn_3seed

cho data/embeddings/r2vul/func_test.done
step vuldet.train.llm_only --dataset r2vul --model unixcoder --no-report
for s in 42 43 44; do step vuldet.train.gnn_sup --dataset r2vul --mode llm_gnn --func-model unixcoder --seed $s --no-report; done
xong L1_L4_L9_unixcoder

cho data/embeddings/r2vul/node_codes.npy
for split in train valid test; do cho "data/embeddings/r2vul/qwen15/func_${split}.npy"; done
cho results/lora/r2vul5/pred_valid.csv
cho results/lora/r2vul5/pred_test.csv
step vuldet.train.llm_only --dataset r2vul --model qwen15 --no-report
step vuldet.train.llm_only --dataset r2vul --model unixcoder+qwen15 --no-report
for s in 42 43 44; do step vuldet.train.gnn_sup --dataset r2vul --mode llm_gnn --func-model qwen15 --seed $s --no-report; done
step vuldet.train.lora_result --dataset r2vul --no-report
xong L2_L3_L5_L6_L7_L10_qwen
step vuldet.train.ensemble --dataset r2vul
step vuldet.eval.cai_thien_f1 --dataset r2vul --no-report
xong L11_L13_ket_hop
