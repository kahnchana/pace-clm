#!/usr/bin/env python3
"""Relocate disposable experiment caches from the root disk to local NVMe."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    node = socket.gethostname()
    if node not in ('ip-10-3-120-167', 'ip-10-3-70-177'):
        raise RuntimeError('authorized nodes only')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU processes must be stopped before cache maintenance')
    mount = Path('/opt/dlami/nvme')
    if not mount.is_mount() or mount.stat().st_dev == Path('/').stat().st_dev:
        raise RuntimeError('local NVMe volume is not mounted')
    cache = mount / 'kranasinghe/pace-clm/node-cache'
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cache.stat().st_uid != os.getuid():
        raise RuntimeError('NVMe cache must be owned by the current user')
    audit = ROOT / 'outputs/storage-maintenance/20261115'
    audit.mkdir(parents=True, exist_ok=True)
    marker = audit / ('cache-cleanup-' + node + '.json')
    if marker.exists():
        raise RuntimeError('cache maintenance already recorded; inspect before repeating')
    def free():
        stat = os.statvfs('/')
        return stat.f_bavail * stat.f_frsize
    report = dict(node=node, utc=datetime.now(timezone.utc).isoformat(),
                  cache_root=str(cache), root_free_before=free(), entries=[])
    def record():
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(marker)
    warm = {'pace-estp128-train-triton', 'pace-estp128-eval-triton'}
    paths = sorted(p for p in Path('/tmp').glob('pace-*')
                   if 'triton' in p.name or p.name == 'pace-qwen-kernels')
    # Delete obsolete caches first, making root-disk space before any rename/link.
    paths.sort(key=lambda p: (p.name in warm, p.name))
    for source in paths:
        if source.is_symlink() or not source.is_dir():
            continue
        if source.stat().st_uid != os.getuid():
            raise RuntimeError('cache belongs to another user: ' + str(source))
        target = cache / 'tmp-compat' / source.name
        if target.exists():
            raise RuntimeError('destination already exists: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        size = int(subprocess.check_output(['du', '-sb', str(source)], text=True).split()[0])
        tick = time.monotonic()
        if source.name in warm:
            # Preserve current kernels; these caches can be regenerated, but reuse is faster.
            shutil.move(str(source), str(target))
            copied = int(subprocess.check_output(['du', '-sb', str(target)], text=True).split()[0])
            if copied != size:
                # Directory allocation differs across filesystems; require file bytes instead.
                print(json.dumps(dict(cache=source.name, logical_bytes_before=size,
                                      logical_bytes_after=copied)), flush=True)
            action = 'warm_cache_moved_to_nvme'
        else:
            shutil.rmtree(source)
            target.mkdir()
            action = 'obsolete_compiled_cache_removed'
        source.symlink_to(target, target_is_directory=True)
        if source.resolve() != target:
            raise RuntimeError('cache alias verification failed: ' + str(source))
        entry = dict(source=str(source), target=str(target), action=action,
                     previous_bytes=size, seconds=time.monotonic()-tick)
        report['entries'].append(entry)
        record()
        print(json.dumps(entry), flush=True)
    for name in ('tmp', 'triton', 'torchinductor', 'cuda', 'torch-extensions', 'matplotlib', 'numba'):
        (cache / name).mkdir(exist_ok=True)
    probe = cache / 'tmp' / 'cache-write-verification'
    probe.write_bytes(b'pace-clm cache write verified\n')
    if probe.read_bytes() != b'pace-clm cache write verified\n':
        raise RuntimeError('NVMe write verification failed')
    probe.unlink()
    report.update(root_free_after=free(), complete=True)
    record()
    print(json.dumps(dict(node=node, root_free_before=report['root_free_before'],
                         root_free_after=report['root_free_after'], complete=True)), flush=True)


if __name__ == '__main__':
    main()
