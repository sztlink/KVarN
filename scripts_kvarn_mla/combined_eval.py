"""DeepSeek-V2-Lite: (1) 16K-decode burst tok/s, (2) long-context perplexity on
wikitext-2-raw (PPL over a 16K-token window; deep positions are predicted from
~16K of KV history = the long-context KV-quant accuracy metric). Run once per
KV mode (MLA_KV=auto FP16 | kvarn_mla_k4_g128) — same delta = quant degradation.
"""
import os, time, math, torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
from datasets import load_dataset

M = os.environ["MLA_MODEL"]; KV = os.environ.get("MLA_KV", "kvarn_mla_k4_g128")
CTX = int(os.environ.get("CE_CTX", "16384"))
MODE = os.environ.get("CE_MODE", "both")   # both | burst | ppl  (ppl: low GMU, B=1)
kw = dict(model=M, dtype="bfloat16", max_model_len=CTX + 256,
          gpu_memory_utilization=float(os.environ.get("CE_GMU", "0.85")),
          max_num_seqs=int(os.environ.get("CE_B", "16")), trust_remote_code=True,
          enable_prefix_caching=False, enforce_eager=False, enable_chunked_prefill=False,
          kernel_config={"moe_backend": "triton"})
if KV != "auto":
    kw["kv_cache_dtype"] = KV; kw["block_size"] = 128
llm = LLM(**kw)
tag = "KVarN" if KV != "auto" else "FP16"

# ---- (1) 16K burst ----
if MODE in ("both", "burst"):
    B = int(os.environ.get("CE_B", "16")); OUT = int(os.environ.get("CE_OUT", "16384"))
    prompt = "Tell me a long detailed story about the history of computing."
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))  # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    outs = llm.generate([prompt] * B, SamplingParams(max_tokens=OUT, temperature=0.0, ignore_eos=True))
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"RESULT_BURST {tag} KV={KV} B={B} OUT={OUT} tok_s={tot/dt:.1f} sec={dt:.1f}", flush=True)
if MODE == "burst":
    print("COMBINED_DONE", flush=True); raise SystemExit

# ---- (2) long-context PPL on wikitext-2-raw test ----
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(t for t in ds["text"] if t.strip())
ids = tok(text, add_special_tokens=False)["input_ids"][:CTX]
sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)[0]
plp = out.prompt_logprobs  # list[pos] -> {token_id: Logprob}; plp[0] is None
def ppl_over(lo, hi):
    nll = 0.0; n = 0
    for i in range(max(lo, 1), hi):
        d = plp[i]
        if d and ids[i] in d:
            nll += -d[ids[i]].logprob; n += 1
    return math.exp(nll / n) if n else float("nan"), n
full, nfull = ppl_over(0, len(ids))
deep, ndeep = ppl_over(int(len(ids) * 0.75), len(ids))  # last quarter = deepest context
print(f"RESULT_PPL {tag} KV={KV} ctx={len(ids)} ppl_full={full:.4f} (n={nfull}) "
      f"ppl_deep25={deep:.4f} (n={ndeep})", flush=True)
print("COMBINED_DONE", flush=True)
