"""Auditable data and frozen-feature helpers for the November SFT campaign."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import threading

ROOT = Path(__file__).resolve().parents[1]
HOST = 'ip-10-3-70-177'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if frozen and path.exists():
        if json.loads(path.read_text()) != json.loads(serialized):
            raise ValueError(f'frozen artifact differs: {path}')
        return
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(serialized)
    temporary.replace(path)


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def position(row):
    # Dataset-qualified path avoids collisions between source video basenames.
    return (row['video_path'], row['question'], int(row['frame_index']))


def stratum(row):
    path = Path(row['video_path'])
    parts = path.parts
    source = parts[parts.index('videos') + 1] if 'videos' in parts else path.parent.name
    return (row['label'], (len(row['turns']) - 1) // 8, bool(row.get('memory')), source)


def install_training_contexts():
    """Replay exact recorded causal messages, replacing image slots only."""
    import train_jevany_streaming as training
    if getattr(training.decision_messages, '_campaign', False):
        return
    original = training.decision_messages

    def messages(row, images, *, response_blind=False):
        if 'controller_messages' not in row:
            return original(row, images, response_blind=response_blind)
        if response_blind or row.get('split') != 'train':
            raise ValueError('recorded contexts are restricted to original-history training')
        iterator = iter(images)
        result = []
        count = 0
        for message in row['controller_messages']:
            parts = []
            for part in message['content']:
                if part['type'] == 'image':
                    parts.append({'type': 'image', 'image': next(iterator)})
                    count += 1
                else:
                    parts.append(dict(part))
            result.append({'role': message['role'], 'content': parts})
        if count != len(images) or result[-1]['role'] != 'user':
            raise ValueError('recorded context/image alignment mismatch')
        return result

    messages._campaign = True
    training.decision_messages = messages


def install_feature_cache(runtime, frame_pools, directory):
    """Optional persistent CPU feature cache; never cache trainable language states."""
    import torch
    from pace_clm.qwen_controller_runtime import MODEL_NAME, MODEL_REVISION
    namespace = dict(model=MODEL_NAME, revision=MODEL_REVISION,
                     processor=runtime.processor.image_processor.to_dict(),
                     resolution=runtime.frame_resolution, tokens=runtime.frame_num_tokens,
                     torch=torch.__version__, dtype='bfloat16', version=1)
    namespace_key = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest()
    directory = Path(directory) / namespace_key
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / 'namespace.json', namespace, frozen=True)
    rank = int(os.environ.get('RANK', 0))
    db = sqlite3.connect(directory / f'rank{rank:02d}.sqlite', check_same_thread=False)
    db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, tensor BLOB NOT NULL)')
    lock = threading.Lock()
    stats = {'hits': 0, 'misses': 0, 'writes': 0}
    original_encode = runtime.encode_frames_features

    def encode(frames, *, chunk=8):
        for frame in frames:
            cached = getattr(frame, '_campaign_cached_tensor', None)
            if cached is not None:
                frame.image_embeds = cached.to(runtime.device)
                del frame._campaign_cached_tensor
            if frame.image_embeds is not None:
                frame.image_embeds = frame.image_embeds.to(runtime.device)
        original_encode(frames, chunk=chunk)
        for frame in frames:
            key = getattr(frame, '_campaign_feature_key', None)
            if key is None or getattr(frame, '_campaign_feature_saved', False):
                continue
            buffer = io.BytesIO()
            torch.save(frame.image_embeds.detach().cpu(), buffer)
            with lock:
                db.execute('INSERT OR IGNORE INTO features VALUES (?,?)', (key, buffer.getvalue()))
                db.commit()
            frame._campaign_feature_saved = True
            stats['writes'] += 1

    runtime.encode_frames_features = encode
    for pool in frame_pools:
        if pool is None:
            continue
        original_prepare = pool.prepare

        def prepare(row, timings=None, prefetched=None, *, cpu_only=False, original=original_prepare):
            result = original(row, timings, prefetched, cpu_only=True)
            inputs, vision, options, decision = result
            stat = Path(row['video_path']).stat()
            for frame, turn in zip(vision, row['turns'], strict=True):
                if getattr(frame, '_campaign_feature_key', None) is not None:
                    continue
                key = hashlib.sha256(json.dumps([row['video_path'], stat.st_size, stat.st_mtime_ns,
                                                int(turn['frame_index'])]).encode()).hexdigest()
                frame._campaign_feature_key = key
                with lock:
                    found = db.execute('SELECT tensor FROM features WHERE key=?', (key,)).fetchone()
                if found:
                    value = torch.load(io.BytesIO(found[0]), map_location='cpu', weights_only=True)
                    if value.ndim != 2 or value.shape[0] != frame.num_tokens or not torch.isfinite(value).all():
                        raise ValueError('invalid cached vision feature')
                    frame._campaign_cached_tensor = value
                    frame._campaign_feature_saved = True
                    stats['hits'] += 1
                else:
                    stats['misses'] += 1
            if not cpu_only:
                encode(vision)
                inputs = {key: value.to(runtime.device) if hasattr(value, 'to') else value
                          for key, value in inputs.items()}
            return inputs, vision, options, decision

        pool.prepare = prepare
    runtime._campaign_feature_stats = stats
