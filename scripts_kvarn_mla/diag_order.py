"""Order/cross-generate diagnostic: run 2 seeds in one process, dump per-seed
score AND a sample raw output, to see if the 2nd generate is corrupted."""
import os, json, re, time, torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
M=os.environ["MLA_MODEL"]; KV=os.environ.get("MLA_KV","kvarn_mla_k4_g128")
MAXTOK=int(os.environ.get("AIME_MAXTOK","12288")); NPROB=int(os.environ.get("AIME_NPROB","12"))
SEEDS=json.loads(os.environ.get("AIME_SEEDS","[1337,42]"))
kw=dict(model=M,dtype="bfloat16",max_model_len=MAXTOK+1024,gpu_memory_utilization=0.80,
        max_num_seqs=64,trust_remote_code=True,enable_prefix_caching=False,
        enforce_eager=os.environ.get("MLA_EAGER","0")=="1",enable_chunked_prefill=False,
        kernel_config={"moe_backend":"triton"})
if KV!="auto": kw["kv_cache_dtype"]=KV; kw["block_size"]=128
llm=LLM(**kw); tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True)
PROB=os.environ.get("AIME_PROB","/home/pbich/projects/reasoning_bench_runpod/data/aime2025/problems.json")
problems=json.load(open(PROB))[:NPROB]
def build(p):
    u=("You are an expert competition mathematician. Solve the following AIME problem step-by-step.\n\n"
       "Problem:\n"+p+"\n\nReason carefully. The final answer is an integer from 000 to 999. "
       "End your response with 'ANSWER: \\boxed{XYZ}'.")
    return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
def extract(t):
    t=re.sub(r"<think>.*?</think>","",t,flags=re.DOTALL)
    m=re.findall(r"\\boxed\{\s*(\d+)\s*\}",t)
    if m: return m[-1]
    m=re.findall(r"ANSWER:\s*(\d{1,3})",t,re.I); return m[-1] if m else None
prompts=[build(p["problem"]) for p in problems]
for gi,seed in enumerate(SEEDS):
    sp=SamplingParams(n=1,temperature=0.6,top_p=0.95,max_tokens=MAXTOK,seed=seed)
    t0=time.time(); outs=llm.generate(prompts,sp); dt=time.time()-t0
    corr=0; olens=[]
    for p,o in zip(problems,outs):
        txt=o.outputs[0].text; olens.append(len(o.outputs[0].token_ids))
        pred=extract(txt)
        if pred is not None and str(int(pred))==str(int(p["answer"])): corr+=1
    print(f"[diag] gen#{gi} seed={seed}: {corr}/{len(problems)} olens(min/med/max)="
          f"{min(olens)}/{sorted(olens)[len(olens)//2]}/{max(olens)} ({dt:.0f}s)",flush=True)
    s=outs[0].outputs[0].text
    print(f"[diag] gen#{gi} seed={seed} SAMPLE prob0 (first 400 + last 300 chars):\n"
          f"  HEAD: {s[:400]!r}\n  TAIL: {s[-300:]!r}",flush=True)
print("DIAG_DONE")
