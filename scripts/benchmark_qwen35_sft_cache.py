#!/usr/bin/env python3
"""Eight-GPU live/cold-cache/warm-cache parity and preparation benchmark."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from qwen35_sft_campaign import install_feature_cache, rows, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('requires eight GPUs')
    from qwen35_diagnostic_kernels import pin_safe_l2norm
    pin_safe_l2norm()
    from pace_clm.qwen_controller_runtime import load_2b_controller
    from pace_clm.jevany_streaming import PointerHead, controller_logits
    from train_jevany_streaming import EpisodeFrames
    runtime, _ = load_2b_controller(checkpoint=str(args.checkpoint / 'controller_adapter'))
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(args.checkpoint / 'pointer_head.pt', weights_only=True))
    head.eval()
    source = list(rows(args.manifest))
    # One row from each context tier per rank, with fixed exact rows for all passes.
    selected = [source[tier * 128 + rank] for tier in range(8)]
    original_encode = runtime.encode_frames_features
    reference_logits, reference_features = [], []
    result = dict(rank=rank, examples=len(selected), passes={})
    with torch.no_grad():
        for mode in ('live', 'cold_cache', 'warm_cache'):
            runtime.encode_frames_features = original_encode
            pool = EpisodeFrames(runtime, response_blind=False, cache_frames=0)
            if mode != 'live':
                install_feature_cache(runtime, (pool,), args.output / 'cache')
            torch.cuda.synchronize()
            started = time.monotonic()
            features_exact = logits_exact = actions_equal = True
            max_logit_difference = 0.
            for index, row in enumerate(selected):
                # Eliminate in-memory frame reuse between rows in every pass.
                pool.frames.clear()
                inputs, vision, options, decision = pool.prepare(row)
                logits = controller_logits(runtime, head, inputs, vision, options, decision, train_decoder=False)
                features = torch.cat([frame.image_embeds for frame in vision]).cpu()
                values = logits.cpu()
                if mode == 'live':
                    reference_logits.append(values)
                    reference_features.append(features)
                else:
                    features_exact &= torch.equal(features, reference_features[index])
                    logits_exact &= torch.equal(values, reference_logits[index])
                    actions_equal &= int(values.argmax()) == int(reference_logits[index].argmax())
                    max_logit_difference = max(max_logit_difference, float((values - reference_logits[index]).abs().max()))
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            result['passes'][mode] = dict(seconds=elapsed, examples_per_second=len(selected) / elapsed,
                features_bit_identical=features_exact, logits_bit_identical=logits_exact,
                actions_equal=actions_equal, max_logit_difference=max_logit_difference,
                cache_stats=dict(getattr(runtime, '_campaign_feature_stats', {})) if mode != 'live' else {})
    write(args.output / f'rank{rank:02d}.json', result)


if __name__ == '__main__':
    main()
