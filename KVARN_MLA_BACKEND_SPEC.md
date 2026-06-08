# KVarN-MLA full-method backend — implementation spec

All stateless units validated (see KVARN_MLA_STATUS.md Updates 14–16). This spec
defines the remaining stateful backend so it can be built cleanly. Architecture:
**persistent kvarn TILE cache → per-step dequant kernel → fp16 scratch → stock
TritonMLA decode** (no fused kernel; rotated query; Hᵀ folded into W_UV).

## Constants (V2-Lite: kv_lora_rank R=512, qk_rope_head_dim ROPE=64, GROUP=128, 4-bit)
Per-block tile record (uint8), validated in kvarn_mla_tilepack.py:
- packed   [0 : 32768)     token-major, R/2 bytes/token
- scale[R] [32768 : 33792) fp16, per-channel (s_row sinkhorn absorbed)
- zp[R]    [33792 : 34816) fp16, per-channel
- s_row[G] [34816 : 35072) fp16, per-token (s_col sinkhorn)
- rope     [35072 : 51456) fp16, GROUP*ROPE
REC = 51456 → spec head_size = REC/GROUP = 402 (uint8), block_size = GROUP = 128.

## (1) Spec — mla_attention.py get_kv_cache_spec
kvarn_mla branch: head_size = kvarn_mla_tile_layout(R, ROPE, 4).rec // GROUP = 402.
block_size forced to GROUP (128).

## (2) Store — do_kv_cache_update (impl-stateful)
Stage incoming fp16 (kv_c_normed, k_pe) per block_id in self._stage[block_id] =
{lat: [≤GROUP, R] fp16, rope: [≤GROUP, ROPE] fp16, n: int}. slot_mapping gives
block_id + offset per token. On block fill (n == GROUP): flush → pack_tile(lat@H
done inside) → write 51456-B record to cache[block_id]; del staging entry.
Sink (block 0 of each seq) + in-progress tail kept fp16 (in staging / pool).
GRAPH-SAFE version: move flush to the metadata builder between replays (mirror
kvarn_attn _flush_watermark_by_sink); do_kv_cache_update only writes fp16 tail.

## (3) Decode — forward_mqa (dequant → stock attention)
- Rotate absorbed query: q_lat_rot = q_lat @ H  (batched matmul, graph-safe).
- Allocate fp16 scratch MLA cache for the active blocks (paged, parallel block
  table). Run _dequant_tile kernel on flushed blocks → scratch (rotated latent +
  rope). Copy staged fp16 tail blocks → scratch (rotate latent by H first).
- Call super().forward_mqa() (stock TritonMLA split decode) on the scratch fp16
  cache with [q_lat_rot | q_rope]. Returns o_rot.
- Un-rotate: fold Hᵀ into W_UV once at load (W_UV' = Hᵀ @ W_UV) so o_rot up-
  projects correctly; OR o = o_rot @ Hᵀ before up-proj.

## (4) Prefill — forward_mha / _compute_prefill_context
Single-chunk prefill: attention over in-memory kv_c (rotate q+k by H, or just use
fp16 path on the un-quantized current chunk — prefill KV not yet quantized).
Context gather: dequant flushed tiles via _dequant_tile into the workspace
(replaces the per-token _kvarn_mla_gather_dequant_kernel).

## (5) CUDA graphs
Decode (dequant kernel + stock split decode + query rotate) is all static →
capturable. Flush (Sinkhorn) runs eager in the builder between replays. Mirror
kvarn_attn: build_for_cudagraph_capture, _is_sink_t, capture-correct metadata,
pool slot reclamation. AttentionCGSupport.UNIFORM_BATCH.

## Validation gates (each before next)
a. Store: write N blocks, read back via unpack_tile, cos vs fp16 ≥ 0.999.   [pack done]
b. Eager decode end-to-end on V2-Lite: coherent output, matches fp16 prompt.
c. GSM8K full method (eager) vs fp16: within noise (expect ~near-lossless 4-bit).
d. CUDA graphs on: capture OK, output matches eager.
e. Graphs-on burst: speed vs fp16 (stock decode → expect ≫ the 0.65% fused).
