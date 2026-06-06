import os, json, re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
M = os.environ["MLA_MODEL"]
llm = LLM(model=M, dtype="bfloat16", max_model_len=26000, gpu_memory_utilization=0.88,
          trust_remote_code=True, enforce_eager=False, enable_prefix_caching=False,
          enable_chunked_prefill=False, kernel_config={"moe_backend": "triton"})
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
probs = json.load(open("/home/pbich/projects/reasoning_bench_runpod/data/aime2025/problems.json"))[:3]
for p in probs:
    msgs = [{"role": "user", "content": "Solve this AIME problem. End with ANSWER: \\boxed{XYZ}.\n\n" + p["problem"]}]
    pr = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    o = llm.generate([pr], SamplingParams(temperature=0.6, top_p=0.95, max_tokens=24576))[0].outputs[0]
    boxed = re.findall(r"\\boxed\{\s*(\d+)\s*\}", o.text)
    print(f"DIAG GOLD={p['answer']} ntok={len(o.token_ids)} finish={o.finish_reason} boxed={boxed}")
    print("  TAIL:", repr(o.text[-160:]))
print("PROMPT_HEAD:", repr(pr[:240]))
