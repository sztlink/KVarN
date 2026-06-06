"""Stage 1 unit: graph-safe tensorized scatter-store for KVarN-MLA.

Replaces the eager Python per-token loop + dict staging in
TritonMLAImpl.do_kv_cache_update. Each decode/prefill step scatters incoming
fp16 (latent, rope) rows into a SPARSE fp16 tail pool at
  slot = block_to_slot[ slot_mapping[i] // GROUP ],
  pos  = slot_mapping[i] % GROUP
with NO Python loop and NO .item() host-sync -> safe to run inside a captured
CUDA graph. The flush (Hadamard+Sinkhorn+RTN pack of a filled 128-token tile)
is done separately in the metadata builder (eager, between replays).

Pool (per attention layer): lat [POOL, GROUP, R] fp16, rope [POOL, GROUP, ROPE]
fp16. POOL is small (~2*max_num_seqs) — only in-progress (unflushed) blocks live
here; flushed history lives int4 in the cache. (A dense 1:1 pool would be ~44GB
on GLM-4.7-Flash, defeating the int4 win.)

This file validates the kernel standalone (scatter -> read back == reference)
before it is wired into triton_mla.py.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _kvarn_mla_scatter_store_kernel(
    Lat_in_ptr,          # [T, R]      fp16 incoming latent (kv_c_normed)
    Rope_in_ptr,         # [T, ROPE]   fp16 incoming rope (k_pe)
    Slot_mapping_ptr,    # [T]         int64  (slot < 0 => pad/skip)
    Block_to_slot_ptr,   # [num_blocks_lookup] int32 (-1 = no pool slot)
    Pool_lat_ptr,        # [POOL, GROUP, R]     fp16
    Pool_rope_ptr,       # [POOL, GROUP, ROPE]  fp16
    stride_lat_t, stride_rope_t,
    stride_pl_b, stride_pl_t, stride_pr_b, stride_pr_t,
    R: tl.constexpr, ROPE: tl.constexpr, GROUP: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """Scatter one token row -> pool[slot, pos]. Grid: (T,)."""
    i = tl.program_id(0)
    sm = tl.load(Slot_mapping_ptr + i)
    if sm < 0:
        return
    block_id = sm // GROUP
    pos = (sm % GROUP).to(tl.int64)
    in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
    if not in_range:
        return
    slot = tl.load(Block_to_slot_ptr + block_id)
    if slot < 0:
        return
    offs_r = tl.arange(0, R)
    lat = tl.load(Lat_in_ptr + i * stride_lat_t + offs_r)
    tl.store(Pool_lat_ptr + slot.to(tl.int64) * stride_pl_b + pos * stride_pl_t + offs_r, lat)
    offs_rope = tl.arange(0, ROPE)
    rope = tl.load(Rope_in_ptr + i * stride_rope_t + offs_rope)
    tl.store(Pool_rope_ptr + slot.to(tl.int64) * stride_pr_b + pos * stride_pr_t + offs_rope, rope)


def scatter_store(lat, rope, slot_mapping, block_to_slot, pool_lat, pool_rope, group):
    T, R = lat.shape
    ROPE = rope.shape[-1]
    _kvarn_mla_scatter_store_kernel[(T,)](
        lat, rope, slot_mapping, block_to_slot, pool_lat, pool_rope,
        lat.stride(0), rope.stride(0),
        pool_lat.stride(0), pool_lat.stride(1),
        pool_rope.stride(0), pool_rope.stride(1),
        R=R, ROPE=ROPE, GROUP=group,
        NUM_BLOCKS_LOOKUP=block_to_slot.shape[0],
    )


def _test():
    torch.manual_seed(0)
    dev = "cuda"
    R, ROPE, GROUP = 512, 64, 128
    NUM_BLOCKS, POOL = 64, 8
    # Map a few blocks -> pool slots; others -1 (flushed / no slot).
    block_to_slot = torch.full((NUM_BLOCKS,), -1, dtype=torch.int32, device=dev)
    mapped = {5: 0, 6: 1, 9: 2}          # block_id -> slot
    for b, s in mapped.items():
        block_to_slot[b] = s
    pool_lat = torch.zeros(POOL, GROUP, R, dtype=torch.float16, device=dev)
    pool_rope = torch.zeros(POOL, GROUP, ROPE, dtype=torch.float16, device=dev)

    # Build T tokens: some into mapped blocks at various positions, some pad,
    # some into unmapped blocks (must be skipped).
    slots = [5 * GROUP + 0, 5 * GROUP + 1, 6 * GROUP + 10, 9 * GROUP + 127,
             -1, 7 * GROUP + 3]           # 7 is unmapped -> skip; -1 -> pad
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=dev)
    T = len(slots)
    lat = torch.randn(T, R, dtype=torch.float16, device=dev)
    rope = torch.randn(T, ROPE, dtype=torch.float16, device=dev)

    scatter_store(lat, rope, slot_mapping, block_to_slot, pool_lat, pool_rope, GROUP)

    ok = True
    for i, sm in enumerate(slots):
        if sm < 0:
            continue
        b, pos = sm // GROUP, sm % GROUP
        if b not in mapped:
            continue
        s = mapped[b]
        if not torch.equal(pool_lat[s, pos], lat[i]):
            print(f"FAIL lat token {i} block {b} pos {pos}"); ok = False
        if not torch.equal(pool_rope[s, pos], rope[i]):
            print(f"FAIL rope token {i} block {b} pos {pos}"); ok = False
    # Unmapped/pad must leave pool zero where nothing was written.
    if not torch.equal(pool_lat[3], torch.zeros_like(pool_lat[3])):  # slot 3 unused
        print("FAIL: unused slot got written"); ok = False
    print("SCATTER STORE UNIT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    _test()
