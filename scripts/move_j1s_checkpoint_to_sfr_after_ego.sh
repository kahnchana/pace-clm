#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
DEST=/fsx/sfr/data/kanchana/pace-clm/outputs
LOG="$ROOT/outputs/artifact-migration"
SOURCE="$ROOT/outputs/jevany-j1s/full-seed42"
TARGET="$DEST/jevany-j1s/full-seed42"
COMPLETED="$ROOT/outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/scoring-completed"
while [[ ! -f "$COMPLETED" ]]; do sleep 30; done

mkdir -p "$TARGET" "$LOG"
[[ -d "$SOURCE" && ! -L "$SOURCE" && ! -e "$SOURCE.relocating" ]]
rsync -a --info=stats2 "$SOURCE/" "$TARGET/" > "$LOG/jevany-j1s_full-seed42.copy.log"
rsync -ani --checksum --delete "$SOURCE/" "$TARGET/" > "$LOG/jevany-j1s_full-seed42.verify.log"
[[ ! -s "$LOG/jevany-j1s_full-seed42.verify.log" ]]
mv "$SOURCE" "$SOURCE.relocating"
ln -s "$TARGET" "$SOURCE"
[[ -d "$SOURCE" ]]
rm -rf -- "$SOURCE.relocating"
echo "jevany-j1s/full-seed42 -> $TARGET" | tee -a "$LOG/completed.txt"
