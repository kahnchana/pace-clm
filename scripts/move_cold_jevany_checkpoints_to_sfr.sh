#!/usr/bin/env bash
set -euo pipefail

ROOT=/fsx/home/kranasinghe/repo/pace-clm
DEST=/fsx/sfr/data/kanchana/pace-clm/outputs
LOG="$ROOT/outputs/artifact-migration"
mkdir -p "$DEST" "$LOG"

for rel in \
  jevany-j1sr-v2/full-seed42 \
  jevany-j2/full-seed42 \
  jevany/full-j1-seed42; do
  src="$ROOT/outputs/$rel"
  dst="$DEST/$rel"
  label=${rel//\//_}
  if [[ -L "$src" ]]; then
    [[ "$(readlink "$src")" == "$dst" ]] || { echo "Unexpected link: $src" >&2; exit 1; }
    continue
  fi
  [[ -d "$src" && ! -e "$src.relocating" ]] || { echo "Unexpected source state: $src" >&2; exit 1; }
  mkdir -p "$dst"
  rsync -a --info=stats2 "$src/" "$dst/" > "$LOG/$label.copy.log"
  rsync -ani --checksum --delete "$src/" "$dst/" > "$LOG/$label.verify.log"
  [[ ! -s "$LOG/$label.verify.log" ]] || { echo "Copy verification failed: $rel" >&2; exit 1; }
  mv "$src" "$src.relocating"
  ln -s "$dst" "$src"
  [[ -d "$src" ]] || { echo "Link check failed: $src" >&2; exit 1; }
  rm -rf -- "$src.relocating"
  echo "$rel -> $dst" | tee -a "$LOG/completed.txt"
done
