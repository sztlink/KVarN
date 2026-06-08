"""Profile the vLLM WORKER process (where GPU work runs) via vLLM's built-in
torch profiler, then parse the trace for the top GPU kernels by total time.
Localizes the KVarN-MLA decode gap that micro-bench + builder-sync ruled out.
Set VLLM_TORCH_PROFILER_DIR before running.
"""
import glob
import gzip
import json
import os
import torch
from vllm import LLM, SamplingParams

MODEL = os.environ["MLA_MODEL"]
KV = os.environ.get("MLA_KV", "kvarn_mla_k4_g128")
TRACE_DIR = os.environ["VLLM_TORCH_PROFILER_DIR"]
kw = dict(model=MODEL, trust_remote_code=True, dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("MLA_GMU", "0.80")),
          enforce_eager=False, enable_prefix_caching=False, enable_chunked_prefill=False,
          kernel_config={"moe_backend": "triton"},
          profiler_config={"profiler":"torch","torch_profiler_dir":TRACE_DIR})
if KV != "auto":
    kw["kv_cache_dtype"] = KV; kw["block_size"] = 128
llm = LLM(**kw)
B = int(os.environ.get("MLA_B", "16"))
prompts = ["Tell me a long detailed story about computing history."] * B
llm.generate(prompts, SamplingParams(max_tokens=48, temperature=0.0, ignore_eos=True))  # warmup
llm.start_profile()
llm.generate(prompts, SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True))
llm.stop_profile()
del llm

# parse the newest trace
files = sorted(glob.glob(os.path.join(TRACE_DIR, "*.json*")), key=os.path.getmtime)
if not files:
    print("NO TRACE FILES in", TRACE_DIR); raise SystemExit
f = files[-1]
op = gzip.open if f.endswith(".gz") else open
with op(f, "rt") as fh:
    data = json.load(fh)
ev = data.get("traceEvents", data) if isinstance(data, dict) else data
# GPU kernel events: cat == 'kernel'
agg = {}
total = 0.0
for e in ev:
    if e.get("cat") == "kernel" and "dur" in e:
        nm = e["name"]
        agg[nm] = agg.get(nm, 0.0) + e["dur"]
        total += e["dur"]
total = total or 1.0
print(f"\n===== TOP GPU kernels (KV={KV}), trace={os.path.basename(f)} =====")
for nm, us in sorted(agg.items(), key=lambda x: -x[1])[:25]:
    short = nm[:74]
    print(f"{short:76} {us/1000:9.2f} ms {100*us/total:5.1f}%")
def grp(n):
    n = n.lower()
    if "grouped_stage1" in n or "splitk_stage2" in n or "tile_decode" in n: return "KVARN_DECODE"
    if "scatter" in n: return "KVARN_STORE"
    if "sinkhorn" in n or "variance" in n or "flush" in n: return "KVARN_FLUSH"
    if any(k in n for k in ["gemm","cutlass","cublas","ampere","sm90","sm80","wgmma","matmul","gett"]): return "GEMM(proj+rotation)"
    if "moe" in n or "expert" in n or "grouped_gemm" in n: return "MOE"
    if "rms" in n or "norm" in n: return "NORM"
    if "rope" in n or "rotary" in n: return "ROPE"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "cast" in n or "fused" in n: return "ELEMENTWISE/COPY"
    return "other"
ag = {}
for nm, us in agg.items():
    ag.setdefault(grp(nm), 0.0); ag[grp(nm)] += us
print("\n===== grouped =====")
for k, v in sorted(ag.items(), key=lambda x: -x[1]):
    print(f"{k:24} {v/1000:9.2f} ms {100*v/total:5.1f}%")

# --- extra: wall span vs GPU-busy vs CPU-op time (find the non-kernel gap) ---
cpu = [e for e in ev if e.get("cat") in ("cpu_op","user_annotation") and "dur" in e]
allts = [e for e in ev if "ts" in e and "dur" in e]
if allts:
    t0 = min(e["ts"] for e in allts); t1 = max(e["ts"]+e["dur"] for e in allts)
    kb = sum(e["dur"] for e in ks)
    print(f"\n[timeline] wall_span={ (t1-t0)/1000:.1f}ms  gpu_kernel_busy={kb/1000:.1f}ms  "
          f"gpu_util={100*kb/(t1-t0):.0f}%")
    from collections import Counter
    c = Counter()
    for e in cpu: c[e["name"][:46]] += e["dur"]
    print("[top CPU ops by total dur]")
    for nm,us in c.most_common(12): print(f"   {us/1000:8.1f}ms  {nm}")
