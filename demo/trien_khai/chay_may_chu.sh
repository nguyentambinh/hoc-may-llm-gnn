#!/bin/bash
# Nguyễn Tâm Bình - 2324801030003
set -e
KHOA=$(grep '^ANTHROPIC_API_KEY=' /root/.hocmay-secrets/anthropic.env | cut -d= -f2- | tr -d '\r\n ')
docker exec hocmay-vuldet pkill -f "uvicorn app:app" 2>/dev/null || true
docker exec -d -w /work/demo/may_chu -e HF_HOME=/work/.cache/hf -e HF_HUB_OFFLINE=1 -e ANTHROPIC_API_KEY="$KHOA" -e SO_LUONG=4 hocmay-vuldet sh -c "uvicorn app:app --host 0.0.0.0 --port 8088 >> /work/results/logs/demo_may_chu.log 2>&1"
for i in $(seq 1 60); do
  docker exec hocmay-vuldet python -c "import urllib.request,json;print(json.load(urllib.request.urlopen('http://127.0.0.1:8088/api/suc-khoe'))['san_sang'])" 2>/dev/null | grep -q True && { echo "may chu san sang"; exit 0; }
  sleep 5
done
echo "may chu chua san sang, xem results/logs/demo_may_chu.log"; exit 1
