"""Profile one steady-state decode burst of KVarN-MLA on GLM-4.7-Flash.
Ranks CUDA kernels by total GPU time to find the real bottleneck (grouped decode
kernel vs query-rotation/un-rotation matmuls vs MoE vs everything else).
Run with KVARN_MLA_GRAPH=1 (or MLA_KV=auto for the FP16 reference profile)."""
import os
import torch
from vllm import LLM, SamplingParams

MODEL = os.environ["MLA_MODEL"]
KV = os.environ.get("MLA_KV", "kvarn_mla_k4_g128")
kw = dict(model=MODEL, trust_remote_code=True, dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("MLA_GMU", "0.80")),
          enforce_eager=os.environ.get("MLA_EAGER","1")=="1", enable_prefix_caching=False, enable_chunked_prefill=False,
          kernel_config={"moe_backend": "triton"})
if KV != "auto":
    kw["kv_cache_dtype"] = KV; kw["block_size"] = 128
llm = LLM(**kw)

B = int(os.environ.get("MLA_B", "16"))
prompts = ["Tell me a long detailed story about computing history."] * B
# warmup (capture graphs, JIT)
llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True))

sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    llm.generate(prompts, sp)

ka = prof.key_averages()
rows = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)
total = sum(e.self_device_time_total for e in ka) or 1.0
print(f"\n===== TOP CUDA kernels (KV={KV}) — self GPU time =====")
print(f"{'kernel':70} {'ms':>10} {'%':>6}")
for e in rows[:25]:
    ms = e.self_device_time_total / 1000.0
    nm = e.key[:68]
    print(f"{nm:70} {ms:10.2f} {100*e.self_device_time_total/total:6.1f}")
# Group KVarN-relevant kernels
def grp(name):
    n = name.lower()
    if "grouped_stage1" in n or "splitk_stage2" in n or "tile_decode" in n: return "KVARN_DECODE"
    if "scatter_store" in n: return "KVARN_STORE"
    if "flush" in n or "sinkhorn" in n or "variance" in n: return "KVARN_FLUSH"
    if "gemm" in n or "matmul" in n or "cutlass" in n or "ampere" in n or "cublas" in n or " mm" in n: return "GEMM(incl rotation+proj)"
    if "moe" in n or "expert" in n: return "MOE"
    return "other"
agg = {}
for e in ka:
    agg.setdefault(grp(e.key), 0.0)
    agg[grp(e.key)] += e.self_device_time_total
print("\n===== grouped =====")
for k, v in sorted(agg.items(), key=lambda x: -x[1]):
    print(f"{k:28} {v/1000:9.2f} ms  {100*v/total:5.1f}%")
