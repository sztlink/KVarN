"""Capture a REAL GLM-4.7-Flash MLA latent, then measure quant fidelity of
(a) KVarN's scheme (Hadamard+Sinkhorn+per-channel RTN) vs (b) PLAIN int4
(per-token min/max) vs (c) per-channel int4 — on attention scores + output.
Tests whether int4 is 'inherently' lossy here or KVarN's scheme is the problem."""
import os, sys, torch, torch.nn.functional as F
M = os.environ["MLA_MODEL"]
from vllm import LLM, SamplingParams
import vllm.v1.attention.backends.mla.triton_mla as tm

# monkeypatch to capture the raw normed latent on first store call
_cap = {}
_orig = tm.TritonMLAImpl.do_kv_cache_update
def _patched(self, kv_c_normed, k_pe, *a, **k):
    if "lat" not in _cap and 150 < kv_c_normed.shape[0] < 500:
        torch.save(kv_c_normed.detach().float().cpu(), "/tmp/glm_latent.pt"); _cap["lat"]=1; print("[probe] CAPTURED", kv_c_normed.shape, "absmax", float(kv_c_normed.float().abs().max()), "std", float(kv_c_normed.float().std()), flush=True)
    return _orig(self, kv_c_normed, k_pe, *a, **k)
tm.TritonMLAImpl.do_kv_cache_update = _patched

llm = LLM(model=M, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.80,
          trust_remote_code=True, enforce_eager=True, enable_prefix_caching=False, kv_cache_dtype="kvarn_mla_k4_g128", block_size=128,
          enable_chunked_prefill=False, kernel_config={"moe_backend":"triton"})
# a real, content-rich prompt (>128 tokens so it fills a tile)
prompt = ("Solve step by step: " + " ".join(f"term{i}={(i*7)%13}" for i in range(80))
          + " Now reason about the sum.")
llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0))
import time as _t
for _ in range(60):
    if os.path.exists("/tmp/glm_latent.pt"): break
    _t.sleep(1)
lat = torch.load("/tmp/glm_latent.pt").cuda(); R = lat.shape[-1]
T = (lat.shape[0] // 128) * 128
lat = lat[:T]
print(f"[probe] captured latent T={T} R={R} per-ch std range "
      f"[{lat.std(0).min():.3f},{lat.std(0).max():.3f}] absmax={lat.abs().max():.3f}")

from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize
def had(n,d):
    H=torch.ones(1,1)
    while H.shape[0]<n: H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return (H/n**0.5).to(d)
H=had(R,lat.device); G=128

def kvarn_deq(tile,bits):           # KVarN scheme -> dequant in ORIGINAL frame
    rot=(tile@H).t().contiguous()
    bal,sc,sr=variance_normalize(rot)
    qmax=(1<<bits)-1; lo=bal.amin(1,keepdim=True); hi=bal.amax(1,keepdim=True)
    scale=((hi-lo)/qmax).clamp_min(1e-8)
    q=torch.clamp(torch.round((bal-lo)/scale),0,qmax)
    deq_rot=((q*scale+lo)*sc*sr).t()            # [g,R] rotated
    return deq_rot@H.t()                          # back to original frame

def plain_int4_pertoken(tile,bits):  # per-token min/max (each row independent)
    qmax=(1<<bits)-1; lo=tile.amin(1,keepdim=True); hi=tile.amax(1,keepdim=True)
    scale=((hi-lo)/qmax).clamp_min(1e-8)
    q=torch.clamp(torch.round((tile-lo)/scale),0,qmax)
    return q*scale+lo

def plain_int4_perchan(tile,bits):   # per-channel min/max over the tile
    qmax=(1<<bits)-1; lo=tile.amin(0,keepdim=True); hi=tile.amax(0,keepdim=True)
    scale=((hi-lo)/qmax).clamp_min(1e-8)
    q=torch.clamp(torch.round((tile-lo)/scale),0,qmax)
    return q*scale+lo

torch.manual_seed(0)
q=torch.randn(16,R,device=lat.device)            # 16 query heads
score_ref=q@lat.t(); o_ref=F.softmax(score_ref/R**0.5,-1)@lat
for name,fn in [("KVarN(Had+Sink+RTN)",kvarn_deq),("plain int4 per-token",plain_int4_pertoken),
                ("plain int4 per-chan",plain_int4_perchan)]:
    for bits in (4,):
        deq=torch.empty_like(lat)
        for s in range(0,T,G): deq[s:s+G]=fn(lat[s:s+G],bits)
        rel=( (deq-lat).norm()/lat.norm() ).item()
        sc=F.cosine_similarity((q@lat.t()).flatten(),(q@deq.t()).flatten(),0).item()
        o=F.softmax((q@deq.t())/R**0.5,-1)@deq
        ocos=F.cosine_similarity(o_ref.flatten(),o.flatten(),0).item()
        print(f"  {name:24} {bits}b: lat_rel_err={rel:.4f} score_cos={sc:.5f} attn_out_cos={ocos:.5f}")
print("PROBE_DONE")
