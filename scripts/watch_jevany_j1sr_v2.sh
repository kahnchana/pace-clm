#!/usr/bin/env bash
set -euo pipefail
cd /fsx/home/kranasinghe/repo/pace-clm
OUT=outputs/jevany-j1sr-v2
expected=$(wc -l < "$OUT/mining/extra_manifest.jsonl")
deadline=$(( $(date +%s) + 108000 ))
while true; do
  total=0
  shards=0
  for shard in "$OUT"/mining/extra_rollout/predictions.rank*.jsonl; do
    if [[ -f "$shard" ]]; then
      shards=$((shards + 1))
      count=$(wc -l < "$shard")
      total=$((total + count))
    fi
  done
  if [[ "$shards" -eq 8 && "$total" -eq "$expected" ]]; then
    sleep 30
    break
  fi
  if [[ $(date +%s) -ge "$deadline" ]]; then
    echo "Mining rollout deadline exceeded: $total/$expected rows" >&2
    exit 1
  fi
  sleep 60
done
bash scripts/run_jevany_j1sr_v2_current_node.sh
