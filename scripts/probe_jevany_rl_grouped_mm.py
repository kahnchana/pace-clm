import torch,json,time
assert torch.cuda.is_available()
torch.manual_seed(42)
results=[]
for counts in ([0,1,0,7,3,0,2,5],[31,0,127,2,1,64,0,7]):
 offsets=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
 x=torch.randn(sum(counts),128,device='cuda',dtype=torch.bfloat16)
 w=torch.randn(8,64,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)
 pieces=[];start=0
 for i,n in enumerate(counts):pieces.append(x[start:start+n]@w[i]);start+=n
 expected=torch.cat(pieces)
 actual=torch.nn.functional.grouped_mm(x,w,offs=offsets)
 difference=(actual.float()-expected.float()).abs()
 results.append({'rows':sum(counts),'max_absolute_error':float(difference.max()),'max_scaled_error':float((difference/(1+expected.abs().float())).max())})
 assert torch.allclose(actual.float(),expected.float(),atol=.125,rtol=.01)
print(json.dumps({'torch':torch.__version__,'native_grouped_mm':results}),flush=True)
