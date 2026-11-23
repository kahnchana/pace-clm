#!/usr/bin/env python3
"""Copy verified checkpoints to FSx, then replace original directories by links."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = args.inventory.resolve().parent
    inventory = json.loads(args.inventory.read_text())
    keep = {Path(p) for p in inventory['keep']}
    sources = [Path(p) for p in inventory['move']]
    if args.limit is not None:
        sources = sources[:args.limit]
    destination = args.destination.resolve()
    if destination != Path('/fsx/sfr/data/kanchana/pace-clm/checkpoints'):
        raise ValueError('unexpected archive destination')
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < inventory['bytes'] * 1.05:
        raise ValueError('insufficient archive space')
    # A second launch must observe the same protected files, never silently reset them.
    protected_path = audit / 'protected-checkpoints.json'
    if protected_path.exists():
        protected = json.loads(protected_path.read_text())
    else:
        protected = {str(p): {str(f.relative_to(p)): sha(f)
                              for f in sorted(p.rglob('*')) if f.is_file()}
                     for p in keep}
        protected_path.write_text(json.dumps(protected, indent=2) + '\n')
    for p in keep:
        if p.is_symlink():
            raise ValueError('protected checkpoint was replaced: ' + str(p))
    lock = threading.Lock()
    journal = (audit / 'checkpoint-moves.jsonl').open('a', buffering=1)
    progress = audit / 'checkpoint-move-progress.json'
    started = time.monotonic()
    totals = {'verified': 0, 'moved': 0, 'already_moved': 0, 'bytes': 0, 'errors': []}

    def migrate(source):
        if source in keep or not source.is_relative_to(root / 'outputs'):
            raise ValueError('unapproved source: ' + str(source))
        target = destination / source.relative_to(root)
        if source.is_symlink():
            if source.resolve() != target or not target.is_dir():
                raise ValueError('unexpected existing source link: ' + str(source))
            return dict(source=str(source), target=str(target), state='already_moved', bytes=0)
        if not source.is_dir():
            raise ValueError('source checkpoint missing: ' + str(source))
        stage = target.with_name('.' + target.name + '.migrating')
        backup = source.with_name('.' + source.name + '.archive-source')
        link = source.with_name('.' + source.name + '.archive-link')
        if target.exists() or stage.exists() or backup.exists() or link.is_symlink():
            raise ValueError('archive path needs recovery inspection: ' + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        hashes = {}
        total = 0

        def copy_file(src, dst):
            nonlocal total
            src, dst = Path(src), Path(dst)
            if src.is_symlink():
                raise ValueError('inventory unexpectedly contains file link: ' + str(src))
            before = src.stat()
            digest = hashlib.sha256()
            with src.open('rb') as inp, dst.open('xb') as out:
                for block in iter(lambda: inp.read(16 << 20), b''):
                    digest.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            after = src.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('source changed during copy: ' + str(src))
            shutil.copystat(src, dst)
            hashes[str(src.relative_to(source))] = digest.hexdigest()
            total += before.st_size
            return str(dst)

        shutil.copytree(source, stage, copy_function=copy_file)
        copied = {str(f.relative_to(stage)): f for f in stage.rglob('*') if f.is_file()}
        if set(copied) != set(hashes):
            raise ValueError('copied file list mismatch: ' + str(source))
        for name, expected in hashes.items():
            if sha(copied[name]) != expected:
                raise ValueError('archive checksum mismatch: ' + str(copied[name]))
        # Record source fidelity even for legacy checkpoints without commit manifests.
        commit_matches = None
        if (stage / 'commit.json').exists():
            commit = json.loads((stage / 'commit.json').read_text())
            commit_matches = all(hashes.get(name) == digest for name, digest in commit['files'].items())
            if not commit_matches:
                raise ValueError('original commit manifest mismatch: ' + str(source))
        record = dict(source=str(source), target=str(target), bytes=total,
                      sha256=hashes, commit_verified=commit_matches,
                      utc=datetime.now(timezone.utc).isoformat(), state='verified_copy')
        manifests = audit / 'checkpoint-manifests'
        manifests.mkdir(exist_ok=True)
        key = hashlib.sha256(str(source).encode()).hexdigest()
        manifest_path = manifests / (key + '.json')
        manifest_path.write_text(json.dumps(record, indent=2) + '\n')
        stage.rename(target)
        os.symlink(str(target), link, target_is_directory=True)
        source.rename(backup)
        try:
            link.replace(source)
        except BaseException:
            backup.rename(source)
            raise
        if source.resolve() != target or not all((source / n).is_file() for n in hashes):
            raise ValueError('archive link verification failed: ' + str(source))
        # Only delete the redundant original after every destination byte was verified.
        shutil.rmtree(backup)
        record['state'] = 'moved'
        manifest_path.write_text(json.dumps(record, indent=2) + '\n')
        return record

    def update(record):
        with lock:
            journal.write(json.dumps(record) + '\n')
            if record['state'] == 'error':
                totals['errors'].append(record)
            else:
                totals['verified'] += 1
                totals[record['state']] += 1
                totals['bytes'] += record['bytes']
            state = dict(**totals, total=len(sources), seconds=time.monotonic()-started)
            temporary = progress.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(progress)
            if totals['verified'] % 10 == 0 or record['state'] == 'error':
                print(json.dumps({k: v for k, v in state.items() if k != 'errors'}), flush=True)
            if record['state'] == 'error':
                print(json.dumps(record), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(migrate, source): source for source in sources}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                record = dict(state='error', source=str(futures[future]), error=f'{type(exc).__name__}: {exc}')
            update(record)
    for path, hashes in protected.items():
        p = Path(path)
        if p.is_symlink() or any(sha(p / n) != digest for n, digest in hashes.items()):
            raise ValueError('protected checkpoint changed: ' + path)
    print(json.dumps(totals), flush=True)
    if totals['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
