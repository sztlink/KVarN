"""Deterministic long-decode divergence test: does KVarN-MLA derail vs FP16 at
long context? Greedy (temp=0), one prompt, 3000 tokens. Compares token-by-token
agreement prefix length + prints both tails. KV mode via env MLA_KV."""
import os, torch
from vllm import LLM, SamplingParams
M = os.environ["MLA_MODEL"]; KV = os.environ.get("MLA_KV", "kvarn_mla_k4_g128")
kw = dict(model=M, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.80,
          trust_remote_code=True, enforce_eager=False, enable_prefix_caching=False,
          enable_chunked_prefill=False, kernel_config={"moe_backend": "triton"})
if KV != "auto":
    kw["kv_cache_dtype"] = KV; kw["block_size"] = 128
llm = LLM(**kw)
prompt = "Count from 1 and explain each number's parity in one short sentence, continuing well past one hundred: 1 is odd."
o = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=3000))[0].outputs[0]
ids = list(o.token_ids)
print(f"RESULT KV={KV} ntok={len(ids)} finish={o.finish_reason}")
print("HEAD:", repr(o.text[:160]))
print("TAIL:", repr(o.text[-200:]))
# crude repetition/garbage check: ratio of unique tokens in last 200
import collections
last = ids[-200:]
print("uniq_frac_last200:", round(len(set(last))/max(len(last),1), 3))
torch.save(ids, f"/tmp/diverge_{KV}.pt")
