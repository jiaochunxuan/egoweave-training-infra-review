import statistics, time, torch, torch_musa
import torch.nn.functional as f
torch.musa.set_device(0)
d=torch.device('musa:0')
for name in ('_scaled_dot_product_attention_flash_musa', '_scaled_dot_product_attention_math_musa'):
    op=getattr(torch.ops.aten,name,None)
    print('schema',name,getattr(op,'_schemas',None),flush=True)
b,s=4,800
torch.manual_seed(42)
q=torch.randn((b,8,s,256),device=d,dtype=torch.bfloat16,requires_grad=True)
k=torch.randn((b,1,s,256),device=d,dtype=torch.bfloat16,requires_grad=True)
v=torch.randn_like(k,requires_grad=True)
mask=torch.zeros((b,1,s,s),device=d,dtype=torch.bfloat16)
mask[:,:,:-32,-32:]=torch.finfo(torch.bfloat16).min
mask[:,:,:,640:768]=torch.finfo(torch.bfloat16).min
def eager(q,k,v):
    return ((q @ k.repeat_interleave(8,1).transpose(-1,-2))*0.0625+mask).softmax(-1,dtype=torch.float32).to(q.dtype) @ v.repeat_interleave(8,1)
def sdpa_gqa(q,k,v):
    return f.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=.0625,enable_gqa=True)
def sdpa_repeat(q,k,v):
    return f.scaled_dot_product_attention(q,k.repeat_interleave(8,1),v.repeat_interleave(8,1),attn_mask=mask,scale=.0625)
for name, fn in [('eager',eager),('sdpa_gqa',sdpa_gqa),('sdpa_repeat',sdpa_repeat)]:
    try:
        times=[]
        for i in range(5):
            torch.musa.synchronize()
            start=time.monotonic()
            y=fn(q,k,v)
            grads=torch.autograd.grad(y.float().square().mean(),(q,k,v))
            torch.musa.synchronize()
            times.append(time.monotonic()-start)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            torch.autograd.grad(fn(q,k,v).float().square().mean(),(q,k,v))
        names=[e.key for e in prof.key_averages() if 'attention' in e.key]
        if name=='eager': ref,refgrad=y.detach(),tuple(t.detach() for t in grads)
        print(name,'median_sec',statistics.median(times[1:]),'kernels',names,
              'max_output_error',(y-ref).abs().max().item(),'grad_norm_ratios',
              [a.float().norm().item()/b.float().norm().item() for a,b in zip(grads,refgrad)],flush=True)
    except Exception as exc:
        print(name,'FAILED',repr(exc),flush=True)
