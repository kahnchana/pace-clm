import torch
from torch import nn

from pace_clm.rl_reference import load_exact_reference


def test_second_adapter_retains_original_fp32_values(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8).to(torch.bfloat16)

        def forward(self, inputs):
            return self.linear(inputs)

    source = get_peft_model(Model(), LoraConfig(r=2, target_modules=['linear']))
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if 'lora_' in name:
                parameter.copy_(torch.linspace(-.17321, .19753, parameter.numel()).reshape_as(parameter))
    source.save_pretrained(tmp_path)
    target = PeftModel.from_pretrained(Model(), tmp_path, is_trainable=False)
    assert load_exact_reference(target, tmp_path) == 2
    expected = get_peft_model_state_dict(source)
    actual = get_peft_model_state_dict(target, adapter_name='reference')
    assert all(torch.equal(actual[name], value) for name, value in expected.items())
