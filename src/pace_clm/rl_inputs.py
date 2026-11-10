"""Avoid copying raw image patches when RL uses frozen, cached vision features."""
from contextlib import contextmanager
from types import MethodType

import torch


def cached_vision_turn(runtime,messages,vision):
    text=runtime.processor.apply_chat_template(messages,tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    parts=text.split(runtime.processor.image_token)
    if len(parts)-1!=len(vision):raise ValueError('causal image placeholder count differs')
    rendered=[parts[0]]
    for frame,tail in zip(vision,parts[1:],strict=True):
        rendered.extend((runtime.processor.image_token*frame.num_tokens,tail))
    encoded=runtime.tokenizer(''.join(rendered),add_special_tokens=False,return_tensors='pt')
    ids=encoded['input_ids']
    return {'input_ids':ids,'attention_mask':encoded['attention_mask'],
            'mm_token_type_ids':(ids==runtime.processor.image_token_id).to(ids.dtype),
            # Non-None enters the image scatter path. The guarded feature cache
            # supplies every real frame embedding; the vision tower ignores this.
            'pixel_values':vision[0].pixel_values[:1],
            'image_grid_thw':torch.cat([frame.grid for frame in vision],dim=0)}


def install_cached_vision_inputs(runtime):
    if getattr(runtime,'rl_cached_vision_inputs',False):return
    original=runtime._cached_image_features
    @contextmanager
    def cached(vision):
        if not vision or any(frame.image_embeds is None for frame in vision):
            raise RuntimeError('lightweight RL inputs require all causal frozen vision features')
        with original(vision):yield
    runtime.assemble_turn=MethodType(cached_vision_turn,runtime)
    runtime._cached_image_features=cached
    runtime.rl_cached_vision_inputs=True
