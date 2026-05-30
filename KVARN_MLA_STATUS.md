# KVarN-MLA build status (local mla-experiment branch — DO NOT push)

Goal: extend KVarN to quantize the DeepSeek MLA compressed latent (kv_lora_rank
dims) + keep the decoupled RoPE part fp16. Dev model: DeepSeek-V2-Lite
(kv_lora_rank=512, qk_rope_head_dim=64, 27 layers, 64-expert MoE).

## Run prerequisites on this box (sm_120 / Blackwell)
- `VLLM_ATTENTION_BACKEND=TRITON_MLA`  (FlashInfer/FlashMLA fail arch check)
- `VLLM_DISABLE_FLASHINFER_ROPE=1`     (env-gate added to deepseek_scaling_rope.py;
  FlashInfer rope JIT fails on sm_120 -> use native PyTorch rope)
- `kernel_config={'moe_backend':'triton'}`  (MoE kernels)
- `VLLM_USE_FLASHINFER_SAMPLER=0`, `trust_remote_code=True`, `HF_HUB_OFFLINE=1`

## Validated (real, tested)
- ACCURACY (probe): round-trip rotate->sinkhorn->RTN->dequant->unrotate on the
  512 latent in MLACommonImpl.forward_impl (env KVARN_MLA_BITS). bits=4 AND
  bits=2 keep V2-Lite coherent+correct. -> method works on MLA latents.
  Files: kvarn/mla_probe.py + forward_impl hook (mla_attention.py ~line 618).
- CKPT1 savings: packed 512-latent k4 = 402 B/token vs 1152 fp16 = **2.87x**
  (k2 ~4.3x). RoPE-fp16 (128B) is the floor. round-trip cos 0.9997.
- CKPT2 Triton dequant kernel (D=512, V-style, channel-major pack): cos 1.0 vs
  PyTorch ref, max_abs 1.4e-6, runs on sm_120. (/tmp/triton_kvarn_mla_test.py)
  Note: KVarN's existing `_kvarn_dequant_blocks_kernel` is D-parameterized and
  also usable at D=512.

## Design (key decisions)
- Latent stored ROTATED (c_KV @ H, H = orthonormal Hadamard-512). Rotation
  invariance: scores q_a.c = (H q_a).(H c); output = H^T (sum w_i (H c_i)).
  => FUSED path: rotate absorbed-query by H once/step, kernel uses rotated
  latent directly (per-token cost = dequant only, no per-token matmul),
  un-rotate the output once/step (or fold H^T into W_UV). MATERIALIZE path:
  just dequant+unrotate to fp16, feed stock attention (no query/output change).
- MLA quirk: V == K == the latent. The grouped MLA kernel loads K once and
  (IS_MLA) reuses it as V -> only ONE dequant load to add.
- Injection slot: triton_decode_attention.py `_fwd_kernel_stage1_grouped`,
  the K load ~line 364 (next to the existing fp8 dequant path ~line 371-372,
  which is the precedent: load quantized + multiply by scale).
- RoPE (qk_rope_head_dim=64) kept fp16, loaded as today (kpe ~line 376).

## Remaining build
### 3a MATERIALIZE (working + measurable first)
1. cache_dtype `kvarn_mla_k4_g128` (+k2); register cache.py + torch_utils.
2. Packed MLA spec: head_size -> packed bytes (256 latent + scales + 128 rope).
   page_size from packed bytes (mirror TQ/MLA spec; see attention.py MLA spec).
3. KVarNMLABackend(MLACommonBackend)/Impl(MLACommonImpl): override
   get_kv_cache_shape (packed), do_kv_cache_update (rotate+sinkhorn+RTN+pack
   store), forward_impl decode (gather attended blocks -> ckpt2 dequant kernel
   -> unrotate -> fp16 scratch in MLA layout -> stock TritonMLA attention).
   Hard part: paged gather + block-table remap into the fp16 scratch.
4. Backend selection: route kvarn_mla_* + use_mla -> KVarNMLABackend.
   (MLA selection is NOT the cuda.py list; it's the use_mla path.)
5. Validate vs fp16 baseline on V2-Lite; measure num_gpu_blocks (savings) + tok/s.

### 3b FULLY FUSED (peak speed, after 3a works)
- Modify `_fwd_kernel_stage1_grouped`: replace fp16 K-load with packed-load +
  in-loop dequant (ckpt2 logic) using rotated latent; add query rotation by H
  before the kernel + output un-rotation after (or fold into W_UV at
  process_weights_after_loading). No fp16 materialization.
- Validate numerics vs 3a; measure peak tok/s.

## Honest status
Method + savings + dequant kernel = proven. 3a/3b = substantial multi-session
vLLM-MLA integration + kernel surgery (paged cache, block tables, spec,
backend, selection, then fused attention). Resume from here.

## Overnight findings (morning summary)

VALIDATED PRIMITIVES (all real, tested on sm_120):
- Per-token Hadamard+RTN quant of the 512-latent (mla_quant.py): record=388 B
  vs 1152 fp16 = **2.97x**, latent cos 0.9921, RoPE exact. Streaming (no pool),
  and CUDA-graph-safe (matmul/min-max/round/bitops only — no Sinkhorn host-sync).
- Triton dequant kernel D=512 (ckpt2): cos 1.0 vs ref.
- Accuracy probe (Sinkhorn path): coherent at 4/2-bit (enforce_eager only —
  Sinkhorn's iterative ops break CUDA-graph capture; per-token path avoids this).
- BASELINE V2-Lite MLA burst (FP16, cudagraph): **3080 tok/s, cap 1,858,368 tok**.

KEY REGIME FINDING (why a burst SPEED win isn't visible here):
- MLA's latent cache is already tiny (1152 B/token), so on small V2-Lite the KV
  capacity is enormous (1.86M tok) — the burst (~147k tok) is nowhere near
  memory-bound. KVarN-MLA's ~3x savings raises capacity to ~5.5M but that does
  NOT raise throughput when not capacity-bound; materialize would only add
  dequant overhead. Same lesson as standard KVarN: capacity->throughput only in
  the memory-bound regime (large MLA model / very long context). Those (V3/V4,
  600B+) don't fit on 2x sm_120, so a burst speed WIN can't be demonstrated on
  this box. What we CAN show here: end-to-end correctness + real capacity (savings).

REMAINING BUILD (materialize backend) is genuine multi-session vLLM-MLA
integration (cache spec + packed layout + store hook + paged decode + selection)
— do_kv_cache_update is defined on the MLA impl base; the decode reads the paged
cache via the attention kernel (gather/dequant or fused). Not reliably
landable unattended overnight, and would not show a speed win on V2-Lite anyway.

RECOMMENDATION: the honest result is "KVarN-MLA works; ~3x latent compression;
speed win needs the fused kernel + a memory-bound regime (big MLA model)". For a
real speed number we'd want a large MLA model on enough GPUs, or accept the
correctness+savings result on V2-Lite.

## Update: FULL KERNEL SIDE VALIDATED (cos 1.0)
- Fused MLA-decode w/ in-kernel 4-bit dequant (kvarn_mla_attn_proto.py): cos 1.0.
- Paged version, shuffled block_table (kvarn_mla_paged_proto.py): cos 1.0.
  Packed record = [256 latent | 2 scale | 2 zp | 128 rope] = 388 B/token = 2.97x.
- v1: per-token RTN, NO Hadamard (accuracy refinement to add later).

## Remaining: BACKEND PLUMBING (the integration)
Crux: decouple cache-record-size (388 B) from compute-dims (kv_lora_rank=512 +
rope=64). vLLM MLA assumes spec head_size == compute head_size. Plan:
1. torch_utils: "kvarn_mla_k4_g128" -> torch.uint8 ; cache.py CacheDType.
2. MLA layer get_kv_cache_spec (mla_attention.py:973): for kvarn_mla_ ->
   MLAAttentionSpec(head_size=388, dtype=uint8) so page_size = block*388
   (2.97x more blocks). Keep self.kv_lora_rank/qk_rope_head_dim for compute.
3. KVarNMLABackend(TritonMLABackend): get_kv_cache_shape -> (nb, block, 388) uint8;
   get_impl_cls -> KVarNMLAImpl; supported_kv_cache_dtypes += kvarn_mla.
4. KVarNMLAImpl(TritonMLAImpl): override do_kv_cache_update (per-token RTN ->
   scatter packed 388-byte records at slot_mapping) + forward_mqa (call the
   validated paged kernel kvarn_mla_paged_proto). Prefill: use current fp16
   kv_c (no cache dequant) for single-chunk prompts.
5. Selection: route kvarn_mla_ -> KVarNMLABackend (cuda.py use_mla list + registry).
6. Validate V2-Lite end-to-end vs fp16; measure num_gpu_blocks (savings) + tok/s.
   Then V4-Flash on 2 GPUs (memory-bound -> where savings->speed should show).
Status: kernels DONE; plumbing = multi-hour iterative integration (head_size
decouple + store + forward_mqa + spec + selection + V2-Lite debug cycles).

## Update 2: PACKED CACHE WIRED IN vLLM (savings infra confirmed)
V2-Lite with kv_cache_dtype=kvarn_mla_k4_g128 now: passes dtype validation,
MoE(triton), rope, the head_dim/concurrency guards (excluded kvarn_mla from all
4 standard-KVarN startswith paths: cuda.py x2, gpu_worker pool, attention.py
spec, kvarn_attn.supports), and ALLOCATES the packed paged cache at the
compressed 388 B/token footprint (spec head_size=388, uint8). So the savings
machinery works end-to-end up to the store.

NEXT (the last 2 overrides — validated kernels are ready to plug in):
1. STORE: base AttentionImpl.do_kv_cache_update (vllm/v1/attention/backend.py
   ~L910/990) calls concat_and_cache_mla (C++) which writes dense [..,576] and
   FAILS on the packed uint8 [..,388] cache. Override for kvarn_mla: per-token
   RTN (mla_quant.pack_tokens) -> scatter 388-byte records at slot_mapping.
2. DECODE: TritonMLAImpl.forward_mqa -> call the validated paged kernel
   (kvarn_mla_paged_proto) instead of decode_attention_fwd. Prefill: current
   fp16 kv_c (single-chunk) -> no cache dequant.
Then: correctness vs fp16 on V2-Lite, burst (savings+speed), V4-Flash on 2 GPU.

EFFORT: store+decode override + numerical-correctness debug + burst + V4 is a
multi-hour iterative stretch. Everything HARD (3 kernels cos 1.0, savings 2.97x,
packed cache wired) is DONE; remaining is bounded plumbing on 2 methods.

## Update 3: END-TO-END WORKS + V2-Lite burst (the key result)
KVarN-MLA runs end-to-end on V2-Lite (fused decode kernel, correct output).
V2-Lite decode-burst (eager, in=16/out=4096, 32 seqs):
  FP16:       1756 tok/s, cap 1.71M
  KVarN-MLA:   691 tok/s, cap 5.08M  => SAVINGS 2.97x (matches theory!), SPEED 0.39x
Speed 0.39x = (a) v1 kernel unoptimized (per-(batch,head) serial loop, NO
KV-splits/BLOCK_N) + (b) V2-Lite not KV-bound.

## IMPORTANT CONCLUSION (value assessment)
MLA's entire purpose is a TINY KV cache (one ~512-dim latent/token, shared
across heads). So MLA models are essentially NEVER KV-capacity-bound — KV is
already minuscule vs weights/compute. Therefore KVarN-MLA's ~3x latent savings
has little to convert into throughput: there is no memory-bound regime to
exploit, while the in-kernel dequant ADDS per-token cost. Net: on MLA, KVarN
tends to be slower with capacity that isn't the bottleneck. (Contrast standard
attention, where KV is large and KVarN's capacity->throughput win is real.)
This is a feasibility SUCCESS but a VALUE-NEGATIVE result for MLA: the method
works + quantizes the latent losslessly-ish + 2.97x savings, but MLA already
solved the KV-size problem, so it rarely pays off. V4-Flash (149GB MoE) would
be compute/MoE-bound, not KV-bound -> same conclusion expected.
Caveat: an optimized kernel (KV-splits) would close the 0.39x toward ~1x, but
not produce a >1x WIN absent a KV-bound regime, which MLA structurally avoids.

## Update 4: kernel v2 (BLOCK_N tiling) + prefill gather + V4-Flash recon
- Decode kernel rewritten with BLOCK_N=32 token-tiling (was per-token serial).
  Standalone cos=1.0 vs fp16 ref. Record layout changed to 16-byte-aligned
  fields (kvarn_mla_layout(): NB, scale@au(NB), zp@+16, rope@+16, REC=au(...))
  so the gathered fp16 loads vectorize without misaligned-address faults.
  For V2-Lite (R=512,RP=64): REC=416 (was 388) -> 2.77x vs fp16's 1152.
- Added _kvarn_mla_gather_dequant_kernel + branch in _compute_prefill_context:
  unpacks the packed cache into the prefill workspace for chunked prefill
  (context_lens>0), replacing the C++ gather_and_maybe_dequant_cache that
  assumed fp8. Validates on long-prompt V2-Lite (chunked) [pending run].
- V4-Flash recon: it is a SPARSE MLA model (config has index_topk/index_n_heads
  =64/index_head_dim=128 -> NSA-style sparse attention), with PRE-QUANTIZED
  weights (quantization_config) and its OWN compression (hc_sinkhorn_iters,
  compress_ratios). head_dim=512, qk_rope_head_dim=64, num_kv_heads=1, 43 layers,
  256 experts. Sparse MLA uses SparseMLAAttentionImpl (decode-only forward_mqa),
  a DIFFERENT impl path than TritonMLAImpl where kvarn_mla is integrated -> V4
  needs separate sparse-impl integration. And sparse attention reads even fewer
  KV entries, so the "MLA KV not the bottleneck" conclusion is STRONGER for V4.

## Update 5: BLOCK_N=64 + V4-Flash is a separate sparse path (infeasible target)
- BLOCK_N sweep at burst shape (B=32,S=4096,H=16): bn16=1832us, bn32=1142,
  bn64=1048 (best), bn128=2163, bn256=7011. Set decode BLOCK_N=64.
- Chunked-prefill gather VALIDATED: V2-Lite 1300-tok prompt (3x512 chunks) ->
  kvarn_mla output IDENTICAL to FP16. Prefill path complete.
- V4-Flash is NOT a viable kvarn_mla target without major dedicated work:
  * arch DeepseekV4ForCausalLM uses DeepseekV4SparseMLAAttentionImpl (sparse
    indexer SparseAttnIndexer + top-k + compressor state cache + flashmla_sparse
    backend) -- a SEPARATE impl from TritonMLAImpl where kvarn_mla lives.
  * KV cache is HARDCODED fp8 sparse; config forces kv dtype "deepseek_v4_fp8".
  * flashmla_sparse backend: sm_120 support doubtful.
  * Sparse top-k attention reads even FEWER KV entries -> capacity savings matter
    even less. Structurally reinforces the value-negative conclusion for MLA.
  => V4-Flash would need a full sparse-MLA + indexer + compressor kvarn rewrite,
     out of scope. Documented as future work; not attempted further.

## FINAL V2-Lite numbers (best config: BLOCK_N=32)
| V2-Lite decode-burst, eager, 32 seqs | tok/s | vs FP16 | KV cap | vs FP16 |
| FP16                                  | 1718  | 1.00x   | 1.71M  | 1.00x   |
| KVarN-MLA v1 (per-token serial)       |  691  | 0.39x   | 5.08M  | 2.97x   |
| KVarN-MLA v2 (BLOCK_N=32 tiled)       | 1131  | 0.65x   | 4.74M  | 2.77x   |
Kernel optimization: 691->1131 tok/s = 1.64x kernel speedup (0.39x->0.65x FP16).
Capacity dropped 2.97x->2.77x from 16-byte field alignment (REC 388->416); the
alignment is what lets the v2 gathered fp16 loads vectorize. Net win: +64% speed
for -7% capacity. Still <1x speed: V2-Lite is not KV-bound (MLA KV already tiny)
+ FP16 path uses the highly-tuned split decode kernel. KV-splits would help
single-stream latency but not this high-occupancy burst (B*H=512 programs).

## Update 6: V4-Flash is hardware-blocked on this sm_120 box (3 independent reasons)
Empirical probe results (TP=2, 149GB fp8 weights DID load + sparse backend init'd):
  1. kv_cache_dtype=auto -> AssertionError "DeepseekV4 only supports fp8 kv-cache
     format" (V4 hardcodes fp8 KV; kvarn_mla cannot apply without sparse rewrite).
  2. kv_cache_dtype=fp8 -> ValueError "Mxfp4 MoE backend 'TRITON' does not support
     the deployment configuration since kernel does not support current device
     cuda". V4's MoE is MXFP4-quantized; NO MoE backend supports MXFP4 on sm_120.
  => V4-Flash cannot even load NATIVELY on this Blackwell-consumer box, regardless
     of KVarN. A V4 kvarn burst here is infeasible (hardware + separate sparse path
     + hardcoded fp8 KV). Needs a data-center GPU (H100/B200, sm_90/sm_100) AND a
     sparse-MLA kvarn integration. Documented as future work.

## Update 7: realistic prefill-inclusive burst (in=512 x32, out=2048) -- WORKS
| V2-Lite real burst | decode tok/s | total tok/s | KV cap | wall |
| FP16               | 1549         | 1937        | 1.72M  | 42s  |
| KVarN-MLA k4       | 1045         | 1306        | 4.75M  | 63s  |
=> 0.67x speed, 2.77x capacity. End-to-end with 512-token prompts (prefill path
incl. chunked gather) ran with NO errors -- the full pipeline (prefill + decode)
is correct and stable for realistic prompts. Consistent with the decode-only
burst (0.65x). Confirms: kernel-optimized KVarN-MLA is correct + stable + ~0.67x
FP16 speed at ~2.8x capacity on V2-Lite (not KV-bound -> speed<1x as expected).

## Update 8: batch=1 latency -> KV-splits NOT worth it (kernel work done)
batch=1, ctx=1024, 256 decode steps:
  FP16:       17.32 ms/tok (58 tok/s)
  KVarN-MLA:  20.23 ms/tok (49 tok/s) = 0.86x FP16 (only 17% slower)
At batch=1 the decode-attention kernel is a SMALL fraction of per-token latency
(MoE + dense layers dominate, identical for both paths), so KVarN is already
0.86x FP16. KV-splits would optimize only that small attention slice -> negligible
gain, not worth the stage1/stage2 complexity. At high batch (burst) occupancy is
already saturated (B*H programs). => BLOCK_N tiling was the right + sufficient
kernel optimization. KV-splits abandoned (empirically low-value). KERNEL DONE.

## Update 9: CORRECTION to Update 6 -- V4-Flash IS runnable here (user was right)
Update 6's "hardware-blocked / cannot load natively" was WRONG. It came from two
mistakes in my probe, not real limitations:
  (a) I force-set kernel_config={'moe_backend':'triton'} (carried over from the
      V2-Lite sm_120 MoE workaround). For V4's FP4 experts that routes through the
      triton mxfp4 path, which genuinely doesn't support sm_120. UNFORCED, the
      mxfp4 oracle auto-picks MARLIN, which works on Blackwell. (FLASHINFER_TRTLLM
      / MARLIN / DEEPGEMM are the valid mxfp4 backends; triton is not one.)
  (b) nvcc 11.5 was on PATH; TileLang compiled V4's "hc" (Hadamard-compression)
      sparse kernel for sm_120a and 11.5 rejected it ("Value 'sm_120a' is not
      defined"). Putting /usr/local/cuda-12.9 on PATH (CUDA_HOME too) fixes it
      (12.9 supports sm_120a; verified by a standalone -arch=sm_120a compile).
With (a)+(b), V4-Flash on 2x RTX PRO 6000 Blackwell (sm_120, 96GB each) loads
FULLY: 149GB fp8 weights, MARLIN FP4 MoE, FP8 Lightning Indexer, DEEPSEEK_SPARSE_
SWA KV cache, TileLang hc kernel all initialize. It then fails ONLY in the profile
forward pass at an FP8 block-scaled DENSE linear (fp8.py -> BlockScaledMMLinear ->
cutlass_scaled_mm): "Not yet supported ScalarType 44" -- the ue8m0 scale dtype
can't cross this prebuilt vLLM's torch-stable-ABI cutlass op. flashinfer block-
scale fp8 gemm is unavailable here (has_flashinfer_fp8_blockscale_gemm()=False on
flashinfer 0.6.11), so it falls back to the broken cutlass path. Canonical fix =
DeepGEMM (env.txt forced exactly this via VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0).
Building DeepGEMM with cuda-12.9 now; open risk = DeepGEMM is Hopper-oriented, sm_120
support unverified. Launch recipe so far:
  PATH=/usr/local/cuda-12.9/bin:$PATH CUDA_HOME=/usr/local/cuda-12.9
  kv_cache_dtype=fp8, TP=2, trust_remote_code, NO forced moe_backend.

## Update 10: V4-Flash FINAL -- true blocker = DeepGEMM has no sm_120 kernels
Walked the V4-Flash failure all the way down. It is NOT impossible/MoE-blocked
(Update 6 was wrong). On 2x RTX PRO 6000 Blackwell (sm_120) it loads FULLY:
MARLIN FP4 MoE, FP8 Lightning Indexer, DEEPSEEK_SPARSE_SWA, TileLang hc kernel
(needs cuda-12.9 on PATH). The ONLY remaining blocker is V4's FP8 block-scaled
GEMM + HyperConnection ("hc") compression, which must go through DeepGEMM:
  - cutlass fallback: torch-stable-ABI op can't accept the ue8m0 scale dtype
    (ScalarType 44) -> "Not yet supported ScalarType 44".
  - DeepGEMM 2.5.0 (built fine for sm_120a, installed): explicitly asserts
    "Unsupported architecture" on sm_120 in TWO C++ kernels --
    csrc/apis/layout.hpp:59 (SF transform) and csrc/apis/hyperconnection.hpp:56.
    Its kernels are written for sm_90 (Hopper) / sm_100 (B200) only.
  - flashinfer 0.6.11: has_flashinfer_fp8_blockscale_gemm()=False (no path).
=> V4-Flash needs sm_90/sm_100 (your H200 run) OR a DeepGEMM port to sm_120 OR a
   vLLM cutlass build that handles ue8m0. NOT runnable on this sm_120 box without
   DeepGEMM kernel work. This is a SOFTWARE coverage gap, not "MXFP4 impossible".

WORKING LAUNCH RECIPE (gets to the DeepGEMM wall; everything else runs on sm_120):
  PATH=/usr/local/cuda-12.9/bin:$PATH  CUDA_HOME=/usr/local/cuda-12.9
  LLM(model=V4-Flash, trust_remote_code=True, tensor_parallel_size=2,
      kv_cache_dtype="fp8", enforce_eager=True)   # do NOT force moe_backend
  (MARLIN FP4 MoE auto-selected.) To even attempt DeepGEMM on sm_120 you must also
  patch support_deep_gemm() (cuda.py) to include family(120) -- but it then faults
  in DeepGEMM's C++ as above. deep_gemm 2.5.0 installed in .venv via
  tools/install_deepgemm.sh (ref 891d57b) built with cuda-12.9.

## Update 11: GSM8K accuracy -- V2-Lite FP16 vs KVarN-MLA 4-bit
8-shot CoT, greedy, full test set (1319), identical harness (0 extraction-None;
sample 26/80 matches full):
  FP16:            30.78% (406/1319)
  KVarN-MLA 4-bit: 27.82% (367/1319)
=> 4-bit costs ~3.0pp (9.6% rel), ~2.3 SE -> real but modest. (FP16 ~31% vs the
paper's ~41% = base model + 400-tok cap truncating some CoT; comparison is fair
since both share the harness.) CAVEAT: MLA path is per-token RTN ONLY -- no
Hadamard, no Sinkhorn (eager-only MLA work; Sinkhorn breaks CUDA-graph). So 3pp
is the RTN-only FLOOR; full KVarN (had+sinkhorn) would likely recover most of it.
GSM8K chosen over AIME (AIME floors a 2.4B-active model). Harness: /tmp/v2lite_gsm8k.py.

## Update 12: FULL KVarN (Had+Sinkhorn+RTN) accuracy + CUDA-graph reconciliation
GSM8K V2-Lite, full-KVarN via mla_probe (roundtrip applied at top of forward()
so the FP16 cache stores lossy values -> all attention sees KVarN values):
  FP16 (run A / run B):  30.78% / 25.55%   <- ~5pp RUN-TO-RUN NOISE (V2-Lite MoE
                                              is not bitwise-deterministic across
                                              processes in vLLM: expert routing +
                                              bf16 atomic reductions + batching).
  full-KVarN 4-bit:      30.25%  -> inside the FP16 noise band => NEAR-LOSSLESS.
  full-KVarN 2-bit:      17.51%  -> ~10pp below FP16, FAR beyond noise => real
                                    large drop (latent too compressed for 2-bit).
Caveat: the ~5pp MoE noise floor swamps few-pp effects; the earlier RTN-only
"-3pp" was within noise (not reliable). 2-bit's collapse is the one above-noise
signal. Roundtrip cosine (deterministic, noise-free): full-KVarN 4-bit=0.9997,
2-bit=0.9926.

CUDA-GRAPH reconciliation (corrects my "Sinkhorn breaks graphs / eager-only"):
Sinkhorn runs at QUANT/FLUSH time, not the per-token decode hot path. Verified:
the existing RTN KVarN-MLA backend CAPTURES CUDA GRAPHS cleanly (enforce_eager=
False works, decode kernel + per-token store are graph-safe). So a fast KVarN-MLA
= graph-captured decode (in-kernel dequant + Hadamard-rotated query, both static)
+ eager periodic Sinkhorn chunk-flush + small FP16 residual for unflushed tokens.
Deciding next via real-latent fidelity whether Sinkhorn is needed at 4-bit or
Hadamard+RTN (fully graph-safe, no flush machinery) already suffices.

## Update 13: DECISIVE -- Hadamard/Sinkhorn are ~inert on the MLA latent
Real V2-Lite prefill latent (773 tok, per-ch std 0.010..1.035), deterministic
round-trip cosine (full variant uses the REAL kvarn variance_normalize):
  4-bit:  RTN-only 0.99667 | Hadamard+RTN 0.99681 | full KVarN 0.99681
  2-bit:  RTN-only 0.92802 | Hadamard+RTN 0.92926 | full KVarN 0.92926
=> Sinkhorn adds 0.00000 over Hadamard+RTN; Hadamard adds +0.00014 over RTN.
Both far below the ~5pp GSM8K noise floor. REASON: the MLA latent is kv_c_NORMED
(RMSNorm-whitened) -> already balanced, no heavy-tailed outlier channels, so the
real variance_normalize early-stops to ~identity. KVarN's machinery targets raw
K/V outliers, which MLA's normed latent doesn't have.
CONSEQUENCE (resolves "want Had+Sinkhorn AND cuda graphs"): no conflict on MLA --
Sinkhorn is inert, so the GRAPH-SAFE RTN backend (already captures CUDA graphs,
near-lossless 4-bit) is full-KVarN-EQUIVALENT. Building the eager-flush Sinkhorn
machinery would add large complexity + break clean hot-path capture for ~0 gain.
RECOMMENDATION: ship the CUDA-graph RTN backend; optionally add Hadamard (graph-
safe static matmul, +0.0001, gives the "KVarN rotation"); skip Sinkhorn on MLA.

## Update 14: COMMIT to full method. Phase 1 (tile math) VALIDATED.
User decision: implement the real KVarN method (Hadamard+Sinkhorn+RTN), not the
RTN shortcut, with CUDA graphs. Plan = port the proven kvarn_attn.py backend
(chunked Sinkhorn flush + fp16 sink/tail + capture-correct metadata + graph-safe
decode) to MLA. Latent = K-path (per-channel quant). Sinkhorn flush is EAGER ->
use shape-agnostic variance_normalize([512,128]) (the Triton sinkhorn hard-codes
128x128; not needed for the eager flush). Graph-safe decode = rotated absorbed-
query . dequant'd rotated int4 tiles + fp16 tail.
Algebra guarantee: <q, unrotate(d)> == <q@H, d> (orthonormal H) -> store rotated
quantized tile + rotate query; no un-rotate in the hot kernel.
Phase 1 validated on REAL V2-Lite latent (kvarn_mla_tile_validate.py):
  4-bit: score_cos 0.99744, attn_out_cos 0.99984 (near-lossless)
  2-bit: score_cos 0.93135, attn_out_cos 0.99153
Tasks #47 (done) / #48 (tile cache+flush store) / #49 (graph-safe decode+graphs).
Phases 2-3 = large multi-component port of the kvarn_attn machinery to MLA.

## Update 15: Phase 2 tile pack/unpack VALIDATED (foundation de-risked)
Per-block tile record (kvarn_mla_tilepack.py), GROUP=128, R=512, ROPE=64, 4-bit:
  packed 32768 + scale[R]1024 + zp[R]1024 + s_row[GROUP]256 + rope 16384 = 51456 B
  = 402 B/tok vs 1152 fp16 -> 2.87x. score_cos=0.99802, rope exact.
Sinkhorn axes (variance_normalize on [R,GROUP]): s_row[R,1]=per-channel (absorbed
into scale/zp), s_col[1,GROUP]=per-token (stored as the row scale). Dequant:
deq_rot[t,c]=(q[t,c]*scale_abs[c]+zp_abs[c])*s_col[t] -> rotated frame; attention
dots H@q with it (no un-rotate). Both math foundations (Phase 1 attn-equivalence,
Phase 2 tile pack) validated on the real latent.
REMAINING (the heavy backend plumbing, mirrors kvarn_attn.py): do_kv_cache_update
fp16 buffering + 128-tok flush via pack_tile; fp16 sink/tail; tile-aware decode
kernel (int4 tiles + fp16 tail, online-softmax); capture-correct metadata for
CUDA graphs. Large multi-step port; foundations now de-risked.

## Update 16: User DECISION = build full method. Dequant Triton kernel validated.
Decision: build the full tile-based Hadamard+Sinkhorn+CUDA-graph KVarN-MLA backend
(heard the caveat that it's ~inert on MLA; proceeding anyway). Architecture per
the user's FlashAttention guidance: persistent kvarn TILE cache -> per-step Triton
DEQUANT kernel -> fp16 scratch (rotated latent + rope) -> STOCK TritonMLA decode
(rotated query, H^T folded into W_UV). No fused dequant+attn kernel; reuses the
tuned decode (also fixes the slow 0.65x).
Validated stateless units (all on real latent): Phase1 method (attn cos 0.99984),
Phase2 tile pack (2.87x, score 0.99802), Phase3-A decode-ref (cos 0.99971),
Phase3 dequant Triton kernel vs unpack_tile (cos 1.0, lat 2.2e-4, rope exact).
REMAINING = stateful backend plumbing: tile-store buffering + 128-tok flush
(do_kv_cache_update), fp16 sink/tail + pool, paged fp16 scratch + dequant wiring,
capture-correct metadata. Port of kvarn_attn machinery to MLA; multi-iteration.
