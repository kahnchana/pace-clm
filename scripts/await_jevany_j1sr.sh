#!/usr/bin/env bash
set -euo pipefail
ROOT=/fsx/home/kranasinghe/repo/pace-clm
OUT="$ROOT/outputs/jevany-j1sr"
MINING_PID=${1:?pass mining torchrun PID}
cd "$ROOT"
while kill -0 "$MINING_PID" 2>/dev/null; do sleep 30; done
count=0
for rank in 00 01 02 03 04 05 06 07; do
  shard="$OUT/mining/rollout/predictions.rank$rank.jsonl"
  if [[ -f "$shard" ]]; then
    count=$((count + $(wc -l < "$shard")))
  fi
done
if [[ "$count" != 1024 ]]; then
  echo "Mining ended with $count/1024 video predictions" >&2
  exit 1
fi
exec bash scripts/run_jevany_j1sr_current_node.sh
