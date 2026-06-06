"""Run ONE generate in a fresh process (no cross-generate state pollution).
DB_MODE=b8 -> 8-prompt batch; DB_MODE=p<i> -> just prompt i alone (B=1).
Saves token-ids to /tmp/fresh_<tag>.pt for cross-process comparison."""
import os, torch
from vllm import LLM, SamplingParams
M=os.environ["MLA_MODEL"]; KV=os.environ.get("MLA_KV","kvarn_mla_k4_g128")
MODE=os.environ["DB_MODE"]; MT=int(os.environ.get("DB_MAXTOK","500"))
kw=dict(model=M,dtype="bfloat16",max_model_len=MT+512,gpu_memory_utilization=0.80,
        trust_remote_code=True,enforce_eager=True,enable_prefix_caching=False,
        enable_chunked_prefill=False,kernel_config={"moe_backend":"triton"})
if KV!="auto": kw["kv_cache_dtype"]=KV; kw["block_size"]=128
llm=LLM(**kw)
allp=[f"Starting from {1+10*i}, count upward and state each number's parity, one per sentence: {1+10*i} is "
      +("odd" if (1+10*i)%2 else "even")+"." for i in range(8)]
sp=SamplingParams(temperature=0.0,max_tokens=MT)
if MODE=="b8":
    outs=llm.generate(allp,sp); ids=[list(o.outputs[0].token_ids) for o in outs]
else:
    i=int(MODE[1:]); outs=llm.generate([allp[i]],sp); ids=[list(outs[0].outputs[0].token_ids)]
torch.save(ids, f"/tmp/fresh_{MODE}.pt")
print(f"FRESH_DONE mode={MODE} lens={[len(x) for x in ids]}")
