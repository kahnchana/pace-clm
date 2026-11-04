#!/usr/bin/env python3
"""Verify eight-rank NCCL and a gradient-sized GPU collective."""

import json
import os
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
results = []
for n in (1, 63_570_000):
    tensor = torch.full((n,), rank + 1, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    expected = world * (world + 1) // 2
    if not torch.all(tensor == expected):
        raise AssertionError("NCCL all-reduce returned the wrong sum")
    results.append({"elements": n, "seconds": time.perf_counter() - started})
gathered = [None] * world
dist.all_gather_object(gathered, {"rank": rank, "results": results})
if rank == 0:
    print(json.dumps({"world": world, "ranks": gathered}), flush=True)
dist.destroy_process_group()
