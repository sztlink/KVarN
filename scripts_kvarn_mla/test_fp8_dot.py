"""#1 prototype: fp8 tensor-core QK dot in the grouped decode kernel.
A/B the fp16 baseline vs an fp8-operand QK-scores variant: measure (a) output
cosine vs fp16 (accuracy) and (b) graphed us/call (speed). PV accumulate stays
fp16 (accuracy-sensitive). Decides whether #1 (Triton int4/fp8 dot) is worth it.
No CUDA — pure Triton. Run on one GPU (standalone, no model load)."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/mnt/nvme1/KVarN/scripts_kvarn_mla")
from kvarn_mla_grouped_decode import _grouped_stage1, _grouped_stage2
from kvarn_mla_tile_decode_kernel import tile_layout
from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize

dev = "cuda"
R, ROPE, G, bits = 512, 64, 128, 4
NB, SC, ZP, SR, RP, REC = tile_layout(R, ROPE, G, bits)
B, NH = 16, 20
HGROUP = 16; n_hg = (NH + HGROUP - 1) // HGROUP; NSPL = 16
seq_len = 1024; n_full = seq_len // G
num_blocks = B * (n_full + 1); qmax = (1 << bits) - 1


@triton.jit
def _grouped_stage1_fp8(
    Q_ptr, Cache_ptr, PoolLat_ptr, PoolRope_ptr, BlockTable_ptr, Seqlens_ptr,
    Block2Slot_ptr, PartO_ptr, PartLse_ptr, sm_scale,
    stride_qb, stride_qh, stride_btb,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_plb, stride_plt, stride_prb, stride_prt,
    H: tl.constexpr, HGROUP: tl.constexpr, R: tl.constexpr, ROPE: tl.constexpr,
    G: tl.constexpr, NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr,
    SR: tl.constexpr, RP: tl.constexpr, REC: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    b = tl.program_id(0); hg = tl.program_id(1); s = tl.program_id(2)
    seq_len = tl.load(Seqlens_ptr + b)
    n_blocks = (seq_len + G - 1) // G
    bps = (n_blocks + NUM_SPLITS - 1) // NUM_SPLITS
    blk0 = s * bps; blk1 = tl.minimum(blk0 + bps, n_blocks)
    offs_h = hg * HGROUP + tl.arange(0, HGROUP); hmask = offs_h < H
    offs_r = tl.arange(0, R); offs_rope = tl.arange(0, ROPE); half = tl.arange(0, R // 2)
    offs_n = tl.arange(0, BLOCK_N)
    qrow = Q_ptr + b * stride_qb + offs_h[:, None] * stride_qh
    q_lat = tl.load(qrow + offs_r[None, :], mask=hmask[:, None], other=0.0).to(tl.float16)
    q_rope = tl.load(qrow + R + offs_rope[None, :], mask=hmask[:, None], other=0.0).to(tl.float16)
    q_lat_f8 = q_lat.to(tl.float8e4nv)                                   # fp8 QK operand
    m_i = tl.full([HGROUP], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([HGROUP], dtype=tl.float32)
    acc = tl.zeros([HGROUP, R], dtype=tl.float32)
    for j in range(blk0, blk1):
        block_id = tl.load(BlockTable_ptr + b * stride_btb + j)
        slot = tl.load(Block2Slot_ptr + block_id,
                       mask=(block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP), other=-1)
        base = block_id.to(tl.int64) * REC
        sc = tl.load((Cache_ptr + base + SC).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        zp = tl.load((Cache_ptr + base + ZP).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        for c0 in range(0, G, BLOCK_N):
            nn = c0 + offs_n
            tok_mask = nn < (seq_len - j * G)
            if slot >= 0:
                pl = PoolLat_ptr + slot.to(tl.int64) * stride_plb + nn[:, None] * stride_plt + offs_r[None, :]
                deq = tl.load(pl, mask=tok_mask[:, None], other=0.0).to(tl.float32)
                pr = PoolRope_ptr + slot.to(tl.int64) * stride_prb + nn[:, None] * stride_prt + offs_rope[None, :]
                rope = tl.load(pr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            else:
                pk = tl.load(Cache_ptr + base + nn[:, None] * (R // 2) + half[None, :],
                             mask=tok_mask[:, None], other=0).to(tl.uint32)
                q4 = tl.interleave((pk & 0xF).to(tl.float32), ((pk >> 4) & 0xF).to(tl.float32))
                srow = tl.load((Cache_ptr + base + SR).to(tl.pointer_type(tl.float16)) + nn,
                               mask=tok_mask, other=0.0).to(tl.float32)
                deq = (q4 * sc[None, :] + zp[None, :]) * srow[:, None]
                rp_ptr = (Cache_ptr + base + RP).to(tl.pointer_type(tl.float16)) + nn[:, None] * ROPE + offs_rope[None, :]
                rope = tl.load(rp_ptr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            deqh = deq.to(tl.float16)                       # fp16 kept for PV accumulate
            deq_f8 = deq.to(tl.float8e4nv)                  # fp8 QK operand
            scores = (tl.dot(q_lat_f8, tl.trans(deq_f8))
                      + tl.dot(q_rope, tl.trans(rope.to(tl.float16)))) * sm_scale
            scores = tl.where(tok_mask[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), deqh)
            m_i = m_new
    empty = blk0 >= n_blocks
    o = tl.where(empty | (l_i <= 0)[:, None], 0.0, acc / tl.where(l_i > 0, l_i, 1.0)[:, None])
    lse = tl.where(empty | (l_i <= 0), -float("inf"), m_i + tl.log(l_i))
    po = PartO_ptr + b * stride_pob + offs_h[:, None] * stride_poh + s * stride_pos + offs_r[None, :]
    tl.store(po, o, mask=hmask[:, None])
    tl.store(PartLse_ptr + b * stride_plseb + offs_h * stride_plseh + s, lse, mask=hmask)


# ---- build cache (real KVarN int4 tiles) ----
Hm = torch.ones(1, 1)
while Hm.shape[0] < R: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
Hd = (Hm / R ** 0.5).to(dev).float()
cache = torch.zeros(num_blocks * REC, dtype=torch.uint8, device=dev)
pool_lat = torch.zeros(B + 4, G, R, dtype=torch.float16, device=dev)
pool_rope = torch.zeros(B + 4, G, ROPE, dtype=torch.float16, device=dev)
block2slot = torch.full((num_blocks,), -1, dtype=torch.int32, device=dev)
block_table = torch.full((B, n_full + 2), -1, dtype=torch.int32, device=dev)
seqlens = []; bid = 0
for b in range(B):
    seqlens.append(seq_len)
    for j in range(n_full):
        block_table[b, j] = bid
        crot = torch.randn(G, R, device=dev) @ Hd
        rot = crot.t().contiguous()
        bal, s_col, s_row = variance_normalize(rot)
        lo = bal.amin(1, keepdim=True); hi = bal.amax(1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round((bal - lo) / scale), 0, qmax).to(torch.uint8)
        qT = q.t().contiguous()
        rec = cache.view(-1, REC)[bid]
        rec[:NB] = (qT[:, 0::2] | (qT[:, 1::2] << 4)).reshape(-1)
        rec[SC:SC+R*2] = (scale*s_row).squeeze(1).to(torch.float16).view(torch.uint8)
        rec[ZP:ZP+R*2] = (lo*s_row).squeeze(1).to(torch.float16).view(torch.uint8)
        rec[SR:SR+G*2] = s_col.squeeze(0).to(torch.float16).view(torch.uint8)
        # rope
        rope_t = torch.randn(G, ROPE, device=dev).to(torch.float16)
        rec[RP:RP+G*ROPE*2] = rope_t.reshape(-1).view(torch.uint8)
        bid += 1
seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
sm = 1.0 / (R + ROPE) ** 0.5
Q = torch.randn(B, NH, R + ROPE, device=dev)
partO = torch.zeros(B, NH, NSPL, R, dtype=torch.float32, device=dev)
partLse = torch.full((B, NH, NSPL), -float("inf"), dtype=torch.float32, device=dev)
O = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
Lse = torch.zeros(B, NH, dtype=torch.float32, device=dev)


def run(stage1):
    partO.zero_(); partLse.fill_(-float("inf")); O.zero_(); Lse.zero_()
    stage1[(B, n_hg, NSPL)](
        Q, cache, pool_lat, pool_rope, block_table, seqlens_t, block2slot, partO, partLse, sm,
        Q.stride(0), Q.stride(1), block_table.stride(0),
        partO.stride(0), partO.stride(1), partO.stride(2), partLse.stride(0), partLse.stride(1),
        pool_lat.stride(0), pool_lat.stride(1), pool_rope.stride(0), pool_rope.stride(1),
        H=NH, HGROUP=HGROUP, R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC,
        BLOCK_N=32, NUM_SPLITS=NSPL, NUM_BLOCKS_LOOKUP=num_blocks, num_warps=8, num_stages=2)
    _grouped_stage2[(B, NH)](
        partO, partLse, O, Lse, partO.stride(0), partO.stride(1), partO.stride(2),
        partLse.stride(0), partLse.stride(1), O.stride(0), O.stride(1), Lse.stride(0),
        R=R, NUM_SPLITS=NSPL)
    return O.clone()


def timeit(stage1, iters=200):
    for _ in range(10): run(stage1)
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(iters): run(stage1)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000

o16 = run(_grouped_stage1)
o8 = run(_grouped_stage1_fp8)
cos = torch.nn.functional.cosine_similarity(o16.flatten().float(), o8.flatten().float(), 0).item()
relerr = ((o8 - o16).norm() / o16.norm()).item()
t16 = timeit(_grouped_stage1); t8 = timeit(_grouped_stage1_fp8)
print(f"[fp8 QK] seq={seq_len} NSPL={NSPL}: cos(O_fp8,O_fp16)={cos:.6f} rel_err={relerr:.4f}")
print(f"[fp8 QK] stage1+2 fp16={t16:.1f}us  fp8={t8:.1f}us  speedup={t16/t8:.2f}x")
print("FP8_DONE")
