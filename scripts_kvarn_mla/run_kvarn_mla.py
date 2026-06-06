"""Run / benchmark KVarN-MLA vs FP16 on an MLA model (V2-Lite / GLM-4.7-Flash).

Confirms correctness (coherent greedy output) and measures decode burst tok/s +
KV capacity (num_gpu_blocks). Uses the documented sm_120 recipe.

Env knobs:
  MLA_MODEL   path to model snapshot (default: V2-Lite cache)
  MLA_KV      "auto" (FP16) | "kvarn_mla_k4_g128"
  MLA_EAGER   "1" (default) eager | "0" cuda graphs
  MLA_MODE    "smoke" (short greedy, default) | "burst" (decode tok/s)
"""
import json
import os
import sys
import time

import torch
from vllm import LLM, SamplingParams

MODEL = os.environ.get(
    "MLA_MODEL",
    "/mnt/nvme1/huggingface/models--deepseek-ai--DeepSeek-V2-Lite/"
    "snapshots/604d5664dddd88a0433dbae533b7fe9472482de0")
KV = os.environ.get("MLA_KV", "kvarn_mla_k4_g128")
EAGER = os.environ.get("MLA_EAGER", "1") == "1"
MODE = os.environ.get("MLA_MODE", "smoke")

kw = dict(
    model=MODEL, trust_remote_code=True, dtype="bfloat16",
    max_model_len=int(os.environ.get("MLA_MAXLEN","4096")), gpu_memory_utilization=float(os.environ.get("MLA_GMU", "0.55")),
    enforce_eager=EAGER, enable_prefix_caching=False,
    tensor_parallel_size=int(os.environ.get("MLA_TP","1")),
    enable_chunked_prefill=False,
    kernel_config={"moe_backend": "triton"},
)
if KV != "auto":
    kw["kv_cache_dtype"] = KV
    kw["block_size"] = 128

llm = LLM(**kw)

# Report KV capacity (num_gpu_blocks * block_size).
try:
    nb = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    bs = llm.llm_engine.vllm_config.cache_config.block_size
    print(f"[mla] KV capacity: num_gpu_blocks={nb} block_size={bs} "
          f"= {nb * bs:,} tokens")
except Exception as e:
    print(f"[mla] capacity probe failed: {e}")

if MODE == "smoke":
    sp = SamplingParams(max_tokens=80, temperature=0.0)
    prompt = "The capital of France is Paris. The currency of France is"
    out = llm.generate([prompt], sp)[0].outputs[0]
    print("RESULT " + json.dumps({"text": out.text,
                                  "token_ids": list(out.token_ids)[:40]}))
else:  # burst
    B = int(os.environ.get("MLA_B", "32"))
    OUT = int(os.environ.get("MLA_OUT", "2048"))
    prompt = "Tell me a long detailed story about the history of computing."
    sp = SamplingParams(max_tokens=OUT, temperature=0.0, ignore_eos=True)
    _ = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))  # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    outs = llm.generate([prompt] * B, sp)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"RESULT_BURST {json.dumps({'kv': KV, 'eager': EAGER, 'B': B, 'out_tokens': tot, 'sec': round(dt,2), 'tok_s': round(tot/dt,1)})}")
