"""Stage 2 opt v2: GROUPED-HEAD split-K KVarN-MLA decode (the fast kernel).

Key insight: in MLA the latent KV is SHARED across all query heads. The grid
(B, num_heads) kernel re-dequantizes the same int4 tile once PER HEAD (20x
redundant on GLM) and uses tl.sum (no tensor cores). This kernel groups ALL H
query heads in one program: dequant each block ONCE, then tl.dot([H,R],[R,BN])
for scores and tl.dot([H,BN],[BN,R]) for the V-accumulate -> tensor cores +
H-fold less dequant. Split-K (grid (B, NSPL)) for occupancy. H padded to pow2.
Validated cos vs the single-program kernel.
"""
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/mnt/nvme1/KVarN/scripts_kvarn_mla")
from kvarn_mla_tile_decode_kernel import _kvarn_mla_tile_decode_kernel, tile_layout


@triton.jit
def _grouped_stage1(
    Q_ptr, Cache_ptr, PoolLat_ptr, PoolRope_ptr, BlockTable_ptr, Seqlens_ptr,
    Block2Slot_ptr, PartO_ptr, PartLse_ptr, sm_scale,
    stride_qb, stride_qh, stride_btb,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_plb, stride_plt, stride_prb, stride_prt,
    H: tl.constexpr, HGROUP: tl.constexpr,
    R: tl.constexpr, ROPE: tl.constexpr, G: tl.constexpr,
    NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr, SR: tl.constexpr,
    RP: tl.constexpr, REC: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    b = tl.program_id(0); hg = tl.program_id(1); s = tl.program_id(2)
    seq_len = tl.load(Seqlens_ptr + b)
    n_blocks = (seq_len + G - 1) // G
    bps = (n_blocks + NUM_SPLITS - 1) // NUM_SPLITS
    blk0 = s * bps
    blk1 = tl.minimum(blk0 + bps, n_blocks)
    offs_h = hg * HGROUP + tl.arange(0, HGROUP)
    hmask = offs_h < H
    offs_r = tl.arange(0, R); offs_rope = tl.arange(0, ROPE); half = tl.arange(0, R // 2)
    offs_n = tl.arange(0, BLOCK_N)
    # q for all heads: [HPAD, R] and [HPAD, ROPE]
    qrow = Q_ptr + b * stride_qb + offs_h[:, None] * stride_qh
    # load query as fp16 (used only in the fp16 tensor-core dots) to save shared mem
    q_lat = tl.load(qrow + offs_r[None, :], mask=hmask[:, None], other=0.0).to(tl.float16)   # [HPAD,R]
    q_rope = tl.load(qrow + R + offs_rope[None, :], mask=hmask[:, None], other=0.0).to(tl.float16)
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
                deq = tl.load(pl, mask=tok_mask[:, None], other=0.0).to(tl.float32)            # [BN,R]
                pr = PoolRope_ptr + slot.to(tl.int64) * stride_prb + nn[:, None] * stride_prt + offs_rope[None, :]
                rope = tl.load(pr, mask=tok_mask[:, None], other=0.0).to(tl.float32)           # [BN,ROPE]
            else:
                pk = tl.load(Cache_ptr + base + nn[:, None] * (R // 2) + half[None, :],
                             mask=tok_mask[:, None], other=0).to(tl.uint32)
                q4 = tl.interleave((pk & 0xF).to(tl.float32), ((pk >> 4) & 0xF).to(tl.float32))
                srow = tl.load((Cache_ptr + base + SR).to(tl.pointer_type(tl.float16)) + nn,
                               mask=tok_mask, other=0.0).to(tl.float32)
                deq = (q4 * sc[None, :] + zp[None, :]) * srow[:, None]                         # [BN,R]
                rp_ptr = (Cache_ptr + base + RP).to(tl.pointer_type(tl.float16)) + nn[:, None] * ROPE + offs_rope[None, :]
                rope = tl.load(rp_ptr, mask=tok_mask[:, None], other=0.0).to(tl.float32)        # [BN,ROPE]
            # tensor-core scores (fp16 operands, fp32 accumulate -> fits shared mem):
            deqh = deq.to(tl.float16)
            scores = (tl.dot(q_lat, tl.trans(deqh))
                      + tl.dot(q_rope, tl.trans(rope.to(tl.float16))))          # [HPAD,BN]
            scores = scores * sm_scale
            scores = tl.where(tok_mask[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])                                                # [HPAD,BN]
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), deqh)                        # [HPAD,R]
            m_i = m_new
    empty = blk0 >= n_blocks
    o = tl.where(empty | (l_i <= 0)[:, None], 0.0, acc / tl.where(l_i > 0, l_i, 1.0)[:, None])
    lse = tl.where(empty | (l_i <= 0), -float("inf"), m_i + tl.log(l_i))
    po = PartO_ptr + b * stride_pob + offs_h[:, None] * stride_poh + s * stride_pos + offs_r[None, :]
    tl.store(po, o, mask=hmask[:, None])
    tl.store(PartLse_ptr + b * stride_plseb + offs_h * stride_plseh + s, lse, mask=hmask)


@triton.jit
def _grouped_stage2(
    PartO_ptr, PartLse_ptr, O_ptr, Lse_ptr,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_ob, stride_oh, stride_lb,
    R: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0); h = tl.program_id(1)
    offs_r = tl.arange(0, R); s_off = tl.arange(0, NUM_SPLITS)
    lse = tl.load(PartLse_ptr + b * stride_plseb + h * stride_plseh + s_off)
    gm = tl.max(lse, axis=0)
    w = tl.exp(lse - gm); wsum = tl.sum(w, axis=0); w = w / wsum
    acc = tl.zeros([R], dtype=tl.float32)
    for s in range(0, NUM_SPLITS):
        o_s = tl.load(PartO_ptr + b * stride_pob + h * stride_poh + s * stride_pos + offs_r)
        acc += o_s * tl.sum(tl.where(s_off == s, w, 0.0), axis=0)
    tl.store(O_ptr + b * stride_ob + h * stride_oh + offs_r, acc)
    tl.store(Lse_ptr + b * stride_lb + h, gm + tl.log(wsum))


def _test():
    torch.manual_seed(2)
    dev = "cuda"
    R, ROPE, G, bits = 512, 64, 128, 4
    NB, SC, ZP, SR, RP, REC = tile_layout(R, ROPE, G, bits)
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < R:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    Hd = (Hm / R ** 0.5).to(dev).float()
    B, NH, num_blocks = 2, 20, 24       # NH=20 like GLM
    HGROUP = 16
    n_hg = (NH + HGROUP - 1) // HGROUP

    from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize
    qmax = (1 << bits) - 1
    cache = torch.zeros(num_blocks * REC, dtype=torch.uint8, device=dev)
    pool_lat = torch.zeros(8, G, R, dtype=torch.float16, device=dev)
    pool_rope = torch.zeros(8, G, ROPE, dtype=torch.float16, device=dev)
    block2slot = torch.full((num_blocks,), -1, dtype=torch.int32, device=dev)
    block_table = torch.full((B, 12), -1, dtype=torch.int32, device=dev)
    seqlens = []; slot_ctr = 0; bid = 0
    for b in range(B):
        n_full, partial = 3, 70
        slen = n_full * G + partial; seqlens.append(slen)
        for j in range(n_full + 1):
            block_table[b, j] = bid
            n = G if j < n_full else partial
            crot = torch.randn(n, R, device=dev) @ Hd
            rope = torch.randn(n, ROPE, device=dev)
            if j < n_full:
                rot = crot.t().contiguous()
                bal, s_col, s_row = variance_normalize(rot)
                lo = bal.amin(1, keepdim=True); hi = bal.amax(1, keepdim=True)
                scale = ((hi - lo) / qmax).clamp_min(1e-8)
                q = torch.clamp(torch.round((bal - lo) / scale), 0, qmax).to(torch.uint8)
                qT = q.t().contiguous()
                packed = (qT[:, 0::2] | (qT[:, 1::2] << 4)).contiguous()
                rec = cache.view(-1, REC)[bid]
                rec[:NB] = packed.reshape(-1)
                rec[SC:SC + R * 2] = (scale * s_row).squeeze(1).to(torch.float16).view(torch.uint8)
                rec[ZP:ZP + R * 2] = (lo * s_row).squeeze(1).to(torch.float16).view(torch.uint8)
                rec[SR:SR + G * 2] = s_col.squeeze(0).to(torch.float16).view(torch.uint8)
                rec[RP:RP + G * ROPE * 2] = rope.to(torch.float16).reshape(-1).view(torch.uint8)
            else:
                block2slot[bid] = slot_ctr
                pool_lat[slot_ctr, :n] = crot.to(torch.float16)
                pool_rope[slot_ctr, :n] = rope.to(torch.float16)
                slot_ctr += 1
            bid += 1
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    sm = 1.0 / (R + ROPE) ** 0.5
    q_lat_rot = torch.randn(B, NH, R, device=dev) @ Hd
    q_rope = torch.randn(B, NH, ROPE, device=dev)
    Q = torch.cat([q_lat_rot, q_rope], -1).contiguous()
    # ref single-program
    Oref = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
    Lref = torch.zeros(B, NH, dtype=torch.float32, device=dev)
    _kvarn_mla_tile_decode_kernel[(B, NH)](
        Q, cache, pool_lat, pool_rope, block_table, seqlens_t, block2slot, Oref, Lref, sm,
        Q.stride(0), Q.stride(1), block_table.stride(0), Oref.stride(0), Oref.stride(1), Lref.stride(0),
        pool_lat.stride(0), pool_lat.stride(1), pool_rope.stride(0), pool_rope.stride(1),
        R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC, BLOCK_N=64,
        NUM_BLOCKS_LOOKUP=num_blocks)
    for NSPL in (1, 4):
        partO = torch.zeros(B, NH, NSPL, R, dtype=torch.float32, device=dev)
        partLse = torch.full((B, NH, NSPL), -float("inf"), dtype=torch.float32, device=dev)
        O = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
        Lse = torch.zeros(B, NH, dtype=torch.float32, device=dev)
        _grouped_stage1[(B, n_hg, NSPL)](
            Q, cache, pool_lat, pool_rope, block_table, seqlens_t, block2slot, partO, partLse, sm,
            Q.stride(0), Q.stride(1), block_table.stride(0),
            partO.stride(0), partO.stride(1), partO.stride(2), partLse.stride(0), partLse.stride(1),
            pool_lat.stride(0), pool_lat.stride(1), pool_rope.stride(0), pool_rope.stride(1),
            H=NH, HGROUP=HGROUP, R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC,
            BLOCK_N=32, NUM_SPLITS=NSPL, NUM_BLOCKS_LOOKUP=num_blocks, num_stages=1)
        _grouped_stage2[(B, NH)](
            partO, partLse, O, Lse,
            partO.stride(0), partO.stride(1), partO.stride(2), partLse.stride(0), partLse.stride(1),
            O.stride(0), O.stride(1), Lse.stride(0), R=R, NUM_SPLITS=NSPL)
        cos = torch.nn.functional.cosine_similarity(O.flatten(), Oref.flatten(), dim=0).item()
        print(f"GROUPED NSPL={NSPL}: o_cos={cos:.6f}", "PASS" if cos > 0.999 else "FAIL")


if __name__ == "__main__":
    _test()
