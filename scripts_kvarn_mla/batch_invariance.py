"""Decisive bug test: KVarN output must be BATCH-INVARIANT (a sequence's per-token
latent quant is independent of other seqs in the batch). Generate 8 prompts as a
B=8 batch AND each alone (B=1), greedy; compare each prompt's KVarN output between
the two. Any difference = batch-dependent corruption = BUG (not lossy quant)."""
import os, torch
from vllm import LLM, SamplingParams
M=os.environ["MLA_MODEL"]; KV=os.environ.get("MLA_KV","kvarn_mla_k4_g128")
kw=dict(model=M,dtype="bfloat16",max_model_len=2048,gpu_memory_utilization=0.80,
        trust_remote_code=True,enforce_eager=os.environ.get("DB_EAGER","1")=="1",
        enable_prefix_caching=False,enable_chunked_prefill=False,kernel_config={"moe_backend":"triton"})
if KV!="auto": kw["kv_cache_dtype"]=KV; kw["block_size"]=128
llm=LLM(**kw)
prompts=[f"Starting from {1+10*i}, count upward and state each number's parity, one per sentence: {1+10*i} is "
         +("odd" if (1+10*i)%2 else "even")+"." for i in range(8)]
sp=SamplingParams(temperature=0.0,max_tokens=800)
# B=8 batch
b8=[list(o.outputs[0].token_ids) for o in llm.generate(prompts,sp)]
# each alone (B=1)
b1=[list(llm.generate([p],sp)[0].outputs[0].token_ids) for p in prompts]
print(f"\n=== BATCH-INVARIANCE KV={KV} (B=8 vs B=1, same prompt) ===")
bad=0
for i in range(8):
    a,b=b1[i],b8[i]; n=min(len(a),len(b))
    div=next((j for j in range(n) if a[j]!=b[j]), n if len(a)==len(b) else n)
    ok = a==b
    if not ok: bad+=1
    print(f"  prompt{i}: B1len={len(a)} B8len={len(b)} match={ok} firstdiff={div}")
print(f"VERDICT: {'BATCH-DEPENDENT BUG' if bad else 'batch-invariant (no bug; any FP16 diff = lossy quant)'} ({bad}/8 differ)")
