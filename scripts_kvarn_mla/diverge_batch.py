"""High-batch + cross-batch divergence test. Runs FP16 then KVarN in ONE process,
TWO sequential generate() calls (like 2 seeds), B distinct greedy prompts, and
compares KVarN vs FP16 token-id agreement per sequence per round. Catches
multi-sequence pool corruption + cross-batch state-reset bugs that B=1 misses."""
import os, torch
from vllm import LLM, SamplingParams
M = os.environ["MLA_MODEL"]; KV = os.environ["MLA_KV"]; B = int(os.environ.get("DB_B","8"))
MT = int(os.environ.get("DB_MAXTOK","4096"))
kw = dict(model=M, dtype="bfloat16", max_model_len=MT+512, gpu_memory_utilization=0.80,
          trust_remote_code=True, enforce_eager=os.environ.get("DB_EAGER","0")=="1", enable_prefix_caching=False,
          enable_chunked_prefill=False, kernel_config={"moe_backend":"triton"})
if KV != "auto":
    kw["kv_cache_dtype"]=KV; kw["block_size"]=128
llm = LLM(**kw)
prompts = [f"Starting from {1+10*i}, count upward and state each number's parity, one per sentence: {1+10*i} is "
           + ("odd" if (1+10*i)%2 else "even") + "." for i in range(B)]
sp = SamplingParams(temperature=0.0, max_tokens=MT)
allids = []
for rnd in range(2):                       # two batches in one process (cross-batch test)
    outs = llm.generate(prompts, sp)
    ids = [list(o.outputs[0].token_ids) for o in outs]
    allids.append(ids)
    print(f"RESULT KV={KV} round={rnd} lens={[len(x) for x in ids]}")
torch.save(allids, f"/tmp/db_{KV.replace('/','_')}.pt")
