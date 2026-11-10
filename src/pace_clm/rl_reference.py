"""Load an exact frozen SFT reference alongside a BF16 base model."""
from pathlib import Path

import torch


def load_exact_reference(model, checkpoint, *, adapter_name='reference'):
    """Undo PEFT's BF16 intermediate rounding when adding a second adapter."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    model.load_adapter(str(checkpoint), adapter_name=adapter_name, is_trainable=False)
    # PEFT promotes the new adapter after its first load. Reload now so the
    # original FP32 checkpoint values go directly into FP32 parameters.
    saved = load_file(str(checkpoint/'adapter_model.safetensors'))
    set_peft_model_state_dict(model, saved, adapter_name=adapter_name)
    actual = get_peft_model_state_dict(model, adapter_name=adapter_name)
    if set(actual) != set(saved):
        raise RuntimeError('reference adapter keys differ from the SFT checkpoint')
    for name, value in actual.items():
        if value.dtype != saved[name].dtype or not torch.equal(value.detach().cpu(), saved[name]):
            raise RuntimeError(f'reference adapter differs from SFT: {name}')
    return len(actual)
