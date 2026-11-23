#!/usr/bin/env python3
"""CPU-only committed SFT checkpoint/optimizer/RNG round-trip verification."""
import argparse
import random
from pathlib import Path
import torch
from safetensors.torch import load_file
from prepare_pace_estp_sft import read,write,sha


def verify(run):
    path=sorted((run/'train').glob('step-*/commit.json'))[-1]
    commit=read(path);checkpoint=path.parent
    for name,digest in commit['files'].items():
        assert sha(checkpoint/name)==digest,name
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['step']==commit['step'] and state['world']==8
    assert state['seen_per_rank']*8==state['step']*32
    assert state['source_presentations']=={'pace':state['step']*16,'estp_it':state['step']*16}
    saved=state['optimizer'];groups=[]
    for group in saved['param_groups']:
        params=[torch.nn.Parameter(torch.zeros_like(saved['state'][i]['exp_avg'])) for i in group['params']]
        groups.append({'params':params,'lr':group['lr']})
    optimizer=torch.optim.AdamW(groups);optimizer.load_state_dict(saved)
    actual=optimizer.state_dict()
    assert actual['param_groups']==saved['param_groups']
    for i,values in saved['state'].items():
        for name,value in values.items():
            if torch.is_tensor(value):assert torch.equal(value,actual['state'][i][name]),(i,name)
            else:assert value==actual['state'][i][name]
    for rank in range(8):
        rng=torch.load(checkpoint/f'rng.rank{rank}.pt',map_location='cpu',weights_only=False)
        generator=torch.Generator();generator.set_state(rng['torch']);a=torch.rand(8,generator=generator)
        generator.set_state(rng['torch']);assert torch.equal(a,torch.rand(8,generator=generator))
        python=random.Random();python.setstate(rng['python']);value=python.random()
        python.setstate(rng['python']);assert value==python.random()
        assert rng['cuda'].dtype==torch.uint8 and rng['cuda'].numel()>0
    adapter=load_file(str(checkpoint/'controller_adapter/adapter_model.safetensors'))
    assert sum(v.numel() for v in adapter.values())==62521344
    assert all(bool(torch.isfinite(v).all()) for v in adapter.values())
    head=torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True)
    assert sum(v.numel() for v in head.values())==1049088
    assert all(bool(torch.isfinite(v).all()) for v in head.values())
    result=dict(passed=True,checkpoint=str(checkpoint),step=state['step'],optimizer_roundtrip_exact=True,
        all_eight_rng_files_verified=True,finite_adapter_and_head=True,extra_optimizer_updates=0,
        source_presentations=state['source_presentations'])
    write(run/'checkpoint-restore-verification.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    verify(p.parse_args().run.resolve())
