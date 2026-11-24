#!/bin/bash
# Nguyễn Tâm Bình - 2324801030003
docker rm -f hocmay-demo-tunnel >/dev/null 2>&1
docker run -d --name hocmay-demo-tunnel --restart unless-stopped --network hocmay-vuldet_default --memory 64m --cpus 0.5 cloudflare/cloudflared:2024.12.2 tunnel --no-autoupdate --url http://hocmay-vuldet:8088 >/dev/null
for i in $(seq 1 30); do
  URL=$(docker logs hocmay-demo-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1)
  [ -n "$URL" ] && { echo "$URL"; exit 0; }
  sleep 2
done
echo "chua lay duoc duong dan"; exit 1
