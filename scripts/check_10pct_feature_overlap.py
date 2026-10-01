#!/usr/bin/env python3
"""Compare re-extracted shared positions with the prior frozen cache."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

p = argparse.ArgumentParser()
p.add_argument("--old", type=Path, required=True)
p.add_argument("--new", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
old = torch.load(a.old, map_location="cpu", weights_only=False)
raw = a.new.with_name(a.new.stem + ".raw.pt")
if not raw.exists():
    os.replace(a.new, raw)
new = torch.load(raw, map_location="cpu", weights_only=False)
lookup = {(r["video_id"], int(r["frame_index"])): i for i, r in enumerate(old["metadata"])}
pairs = [(lookup[(r["video_id"], int(r["frame_index"]))], i)
         for i, r in enumerate(new["metadata"]) if (r["video_id"], int(r["frame_index"])) in lookup]
if len(pairs) < 1000:
    raise RuntimeError(f"unexpectedly small overlap: {len(pairs)}")
x = old["features"][[i for i, _ in pairs]].float()
y = new["features"][[j for _, j in pairs]].float()
cosine = F.cosine_similarity(x, y, dim=-1)
delta = (x-y).abs()
bad = [k for k in range(len(pairs)) if float(cosine[k]) < .999]
report = {"overlap": len(pairs), "raw_min_cosine": float(cosine.min()),
          "raw_mean_cosine": float(cosine.mean()), "raw_max_absolute_delta": float(delta.max()),
          "raw_mean_absolute_delta": float(delta.mean()), "raw_mismatch_count": len(bad),
          "raw_mismatches": [{"video_id": old["metadata"][pairs[k][0]]["video_id"],
                              "frame_index": int(old["metadata"][pairs[k][0]]["frame_index"]),
                              "cosine": float(cosine[k])} for k in bad],
          "resolution": "reuse the prior frozen features at every shared position; retain raw union cache separately"}
for i, j in pairs:
    if int(old["targets"][i]) != int(new["targets"][j]):
        raise RuntimeError(f"shared-position label mismatch: {old['metadata'][i]}")
    new["features"][j] = old["features"][i].to(new["features"].dtype)
temporary = a.new.with_suffix(".pt.tmp")
torch.save(new, temporary)
os.replace(temporary, a.new)
report["harmonized_exact_matches"] = all(
    torch.equal(new["features"][j], old["features"][i].to(new["features"].dtype))
    for i, j in pairs
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report), flush=True)
if not report["harmonized_exact_matches"]:
    raise RuntimeError("shared-position harmonization failed")
