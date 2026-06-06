"""AIME25 accuracy: FP16-KV vs KVarN-MLA on GLM-4.7-Flash. 30 problems, avg@SEEDS.
Both run the same bf16 model; KV cache is bf16 (auto) vs 4-bit KVarN-MLA graph path.
Run once per backend via env MLA_KV (+ KVARN_MLA_GRAPH). Prints per-seed + avg acc.
"""
import json
import os
import re
import time

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

M = os.environ["MLA_MODEL"]
KV = os.environ.get("MLA_KV", "kvarn_mla_k4_g128")
PROB = os.environ.get("AIME_PROB",
                      "/home/pbich/projects/reasoning_bench_runpod/data/aime2025/problems.json")
MAXTOK = int(os.environ.get("AIME_MAXTOK", "12288"))
SEEDS = json.loads(os.environ.get("AIME_SEEDS", "[42, 1337, 7]"))

kw = dict(model=M, dtype="bfloat16", max_model_len=MAXTOK + 1024,
          gpu_memory_utilization=float(os.environ.get("AIME_GMU","0.80")), max_num_seqs=int(os.environ.get('AIME_MAXSEQS','64')), trust_remote_code=True,
          enable_prefix_caching=False, enforce_eager=False, enable_chunked_prefill=False,
          kernel_config={"moe_backend": "triton"})
if KV != "auto":
    kw["kv_cache_dtype"] = KV
    kw["block_size"] = 128
llm = LLM(**kw)
cc = llm.llm_engine.vllm_config.cache_config
print(f"[aime] KV={KV} cap={cc.num_gpu_blocks * cc.block_size:,} tok")
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
problems = json.load(open(PROB))
import os as _o
_n=int(_o.environ.get('AIME_NPROB','0'))
if _n: problems=problems[:_n]


def build(p):
    user = ("You are an expert competition mathematician. Solve the following AIME "
            "problem step-by-step.\n\nProblem:\n" + p + "\n\n"
            "Reason carefully, ensure all calculations are precise. The final answer "
            "is an integer from 000 to 999. End your response with 'ANSWER: \\boxed{XYZ}'.")
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user


def extract(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.findall(r"\\boxed\{\s*(\d+)\s*\}", text)
    if m:
        return m[-1]
    m = re.findall(r"ANSWER:\s*(\d{1,3})", text, re.I)
    if m:
        return m[-1]
    nums = re.findall(r"\b(\d{1,3})\b", text)
    return nums[-1] if nums else None


def norm(s):
    try:
        return str(int(s))
    except Exception:
        return str(s).strip()


prompts = [build(p["problem"]) for p in problems]
accs = []
for seed in SEEDS:
    sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=MAXTOK, seed=seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    correct = 0
    for p, o in zip(problems, outs):
        pred = extract(o.outputs[0].text)
        if pred is not None and norm(pred) == norm(p["answer"]):
            correct += 1
    acc = 100.0 * correct / len(problems)
    accs.append(acc)
    print(f"[aime] KV={KV} seed={seed}: {correct}/{len(problems)} = {acc:.1f}%  ({dt:.0f}s)")
print(f"RESULT_AIME KV={KV} avg@{len(SEEDS)}={sum(accs)/len(accs):.1f}% seeds={accs}")
