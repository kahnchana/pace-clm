"""Persist the exact frozen vision observations used by each behavior-policy rank."""
from pathlib import Path

import torch


class FrozenVisionBank:
    def __init__(self):
        self.entries = {}
        self.replay = False

    @staticmethod
    def pairs(rows, samples):
        for row, sample in zip(rows, samples, strict=True):
            vision = sample[1]
            indices = [int(turn['frame_index']) for turn in row['turns']]
            if len(indices) != len(vision):
                raise ValueError('vision observation geometry differs')
            for index, frame in zip(indices, vision, strict=True):
                yield (row['video_path'], index), frame

    def restore(self, rows, samples, device):
        for key, frame in self.pairs(rows, samples):
            saved = self.entries.get(key)
            if saved is None:
                if self.replay:
                    raise ValueError(f'behavior vision feature missing: {key}')
                continue
            if frame.num_tokens != saved['num_tokens'] or not torch.equal(frame.grid.cpu(), saved['grid']):
                raise ValueError('behavior vision grid/token count changed')
            frame.image_embeds = saved['image_embeds'].to(device=device)

    def capture(self, rows, samples):
        for key, frame in self.pairs(rows, samples):
            if frame.image_embeds is None:
                raise ValueError('cannot save an unencoded behavior observation')
            if key not in self.entries:
                self.entries[key] = {'num_tokens': frame.num_tokens,
                                     'grid': frame.grid.detach().cpu().clone(),
                                     'image_embeds': frame.image_embeds.detach().clone()}

    def save(self, path):
        path = Path(path)
        cpu = {key: {**value, 'image_embeds': value['image_embeds'].cpu().contiguous().clone()}
               for key, value in self.entries.items()}
        temporary = path.with_suffix('.tmp')
        torch.save(cpu, temporary)
        temporary.replace(path)

    def load(self, path, device):
        cpu = torch.load(path, map_location='cpu', weights_only=True)
        self.entries = {key: {**value, 'image_embeds': value['image_embeds'].to(device=device)}
                        for key, value in cpu.items()}
        self.replay = True
