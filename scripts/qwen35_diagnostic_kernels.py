"""Restrict the existing FLA normalization kernel's launch tuning for diagnostics."""
from __future__ import annotations
import importlib

def pin_safe_l2norm():
    import triton
    module=importlib.import_module('fla.modules.l2norm')
    pinned={}
    for name in ['l2norm_fwd_kernel','l2norm_bwd_kernel','l2norm_fwd_kernel1','l2norm_bwd_kernel1']:
        kernel=getattr(module,name)
        options={} if name.endswith('1') else {'BT':32}
        kernel.configs=[triton.Config(options,num_warps=4)]
        kernel.cache.clear()
        pinned[name]=dict(options,num_warps=4)
    return pinned
