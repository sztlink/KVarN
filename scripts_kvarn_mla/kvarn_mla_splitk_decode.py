"""Stage 2 opt: split-K (flash-decoding) KVarN-MLA tile decode.

The single-program kernel runs one (seq, head) serial over the whole KV; at
burst that's only B*H programs and a long serial block loop. Split-K parallelizes
the KV dimension: stage1 grid (B, H, NUM_SPLITS) each does a contiguous slice of
the sequence's blocks -> normalized partial o_s [R] + lse_s; stage2 grid (B, H)
LSE-combines the splits. Dual-source (int4 tiles + rotated fp16 pool), rotated
frame (query pre-rotated, output un-rotated by caller). Validated vs the
single-program kernel (cos ~1.0).
"""
import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, "/mnt/nvme1/KVarN/scripts_kvarn_mla")
from kvarn_mla_tile_decode_kernel import (_kvarn_mla_tile_decode_kernel,
                                          tile_layout)


@triton.jit
def _splitk_stage1(
    Q_ptr, Cache_ptr, PoolLat_ptr, PoolRope_ptr, BlockTable_ptr, Seqlens_ptr,
    Block2Slot_ptr, PartO_ptr, PartLse_ptr, sm_scale,
    stride_qb, stride_qh, stride_btb,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_plb, stride_plt, stride_prb, stride_prt,
    R: tl.constexpr, ROPE: tl.constexpr, G: tl.constexpr,
    NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr, SR: tl.constexpr,
    RP: tl.constexpr, REC: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    seq_len = tl.load(Seqlens_ptr + b)
    n_blocks = (seq_len + G - 1) // G
    bps = (n_blocks + NUM_SPLITS - 1) // NUM_SPLITS       # blocks per split
    blk0 = s * bps
    blk1 = tl.minimum(blk0 + bps, n_blocks)
    offs_r = tl.arange(0, R)
    offs_rope = tl.arange(0, ROPE)
    half = tl.arange(0, R // 2)
    qbase = Q_ptr + b * stride_qb + h * stride_qh
    q_lat = tl.load(qbase + offs_r).to(tl.float32)
    q_rope = tl.load(qbase + R + offs_rope).to(tl.float32)
    e_max = -float("inf"); e_sum = 0.0
    acc = tl.zeros([R], dtype=tl.float32)
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
            offs_n = c0 + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < (seq_len - j * G)
            if slot >= 0:
                pl = PoolLat_ptr + slot.to(tl.int64) * stride_plb + offs_n[:, None] * stride_plt + offs_r[None, :]
                deq = tl.load(pl, mask=tok_mask[:, None], other=0.0).to(tl.float32)
                pr = PoolRope_ptr + slot.to(tl.int64) * stride_prb + offs_n[:, None] * stride_prt + offs_rope[None, :]
                rope = tl.load(pr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            else:
                pk = tl.load(Cache_ptr + base + offs_n[:, None] * (R // 2) + half[None, :],
                             mask=tok_mask[:, None], other=0).to(tl.uint32)
                q4 = tl.interleave((pk & 0xF).to(tl.float32), ((pk >> 4) & 0xF).to(tl.float32))
                srow = tl.load((Cache_ptr + base + SR).to(tl.pointer_type(tl.float16)) + offs_n,
                               mask=tok_mask, other=0.0).to(tl.float32)
                deq = (q4 * sc[None, :] + zp[None, :]) * srow[:, None]
                rp_ptr = (Cache_ptr + base + RP).to(tl.pointer_type(tl.float16)) + offs_n[:, None] * ROPE + offs_rope[None, :]
                rope = tl.load(rp_ptr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            score = (tl.sum(deq * q_lat[None, :], axis=1) + tl.sum(rope * q_rope[None, :], axis=1)) * sm_scale
            score = tl.where(tok_mask, score, -float("inf"))
            new_max = tl.maximum(e_max, tl.max(score, axis=0))
            p = tl.exp(score - new_max)
            alpha = tl.exp(e_max - new_max)
            e_sum = e_sum * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * deq, axis=0)
            e_max = new_max
    # store NORMALIZED partial o_s and lse_s (= m + log l). Empty split -> -inf.
    empty = blk0 >= n_blocks
    o_s = tl.where(empty, 0.0, acc / tl.where(e_sum > 0, e_sum, 1.0))
    lse_s = tl.where(empty | (e_sum <= 0), -float("inf"), e_max + tl.log(e_sum))
    tl.store(PartO_ptr + b * stride_pob + h * stride_poh + s * stride_pos + offs_r, o_s)
    tl.store(PartLse_ptr + b * stride_plseb + h * stride_plseh + s, lse_s)


@triton.jit
def _splitk_stage2(
    PartO_ptr, PartLse_ptr, O_ptr, Lse_ptr,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_ob, stride_oh, stride_lb,
    R: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_r = tl.arange(0, R)
    s_off = tl.arange(0, NUM_SPLITS)
    lse = tl.load(PartLse_ptr + b * stride_plseb + h * stride_plseh + s_off)   # [S]
    gm = tl.max(lse, axis=0)
    w = tl.exp(lse - gm)                                                       # [S]
    wsum = tl.sum(w, axis=0)
    w = w / wsum
    acc = tl.zeros([R], dtype=tl.float32)
    for s in range(0, NUM_SPLITS):
        o_s = tl.load(PartO_ptr + b * stride_pob + h * stride_poh + s * stride_pos + offs_r)
        acc += o_s * tl.sum(tl.where(s_off == s, w, 0.0), axis=0)
    tl.store(O_ptr + b * stride_ob + h * stride_oh + offs_r, acc)
    tl.store(Lse_ptr + b * stride_lb + h, gm + tl.log(wsum))


def _test():
    torch.manual_seed(1)
    dev = "cuda"
    R, ROPE, G, bits = 512, 64, 128, 4
    NB, SC, ZP, SR, RP, REC = tile_layout(R, ROPE, G, bits)
    H = torch.ones(1, 1)
    while H.shape[0] < R:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    Hd = (H / R ** 0.5).to(dev).float()

    B, NH, num_blocks = 2, 4, 24
    cache = torch.zeros(num_blocks * REC, dtype=torch.uint8, device=dev)
    pool_lat = torch.zeros(8, G, R, dtype=torch.float16, device=dev)
    pool_rope = torch.zeros(8, G, ROPE, dtype=torch.float16, device=dev)
    block2slot = torch.full((num_blocks,), -1, dtype=torch.int32, device=dev)
    from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize
    qmax = (1 << bits) - 1
    block_table = torch.full((B, 12), -1, dtype=torch.int32, device=dev)
    seqlens = []; slot_ctr = 0; bid_ctr = 0
    for b in range(B):
        n_full, partial = 4, 50
        slen = n_full * G + partial; seqlens.append(slen)
        for j in range(n_full + 1):
            bid = bid_ctr; bid_ctr += 1; block_table[b, j] = bid
            n = G if j < n_full else partial
            crot = (torch.randn(n, R, device=dev) @ Hd)
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
                slot = slot_ctr; slot_ctr += 1; block2slot[bid] = slot
                pool_lat[slot, :n] = crot.to(torch.float16)
                pool_rope[slot, :n] = rope.to(torch.float16)
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    sm = 1.0 / (R + ROPE) ** 0.5
    q_lat_rot = torch.randn(B, NH, R, device=dev) @ Hd
    q_rope = torch.randn(B, NH, ROPE, device=dev)
    Q = torch.cat([q_lat_rot, q_rope], -1).contiguous()

    # reference: single-program validated kernel
    Oref = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
    Lref = torch.zeros(B, NH, dtype=torch.float32, device=dev)
    _kvarn_mla_tile_decode_kernel[(B, NH)](
        Q, cache, pool_lat, pool_rope, block_table, seqlens_t, block2slot, Oref, Lref, sm,
        Q.stride(0), Q.stride(1), block_table.stride(0), Oref.stride(0), Oref.stride(1), Lref.stride(0),
        pool_lat.stride(0), pool_lat.stride(1), pool_rope.stride(0), pool_rope.stride(1),
        R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC, BLOCK_N=64,
        NUM_BLOCKS_LOOKUP=num_blocks)

    for NSPL in (2, 4):
        partO = torch.zeros(B, NH, NSPL, R, dtype=torch.float32, device=dev)
        partLse = torch.full((B, NH, NSPL), -float("inf"), dtype=torch.float32, device=dev)
        O = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
        Lse = torch.zeros(B, NH, dtype=torch.float32, device=dev)
        _splitk_stage1[(B, NH, NSPL)](
            Q, cache, pool_lat, pool_rope, block_table, seqlens_t, block2slot, partO, partLse, sm,
            Q.stride(0), Q.stride(1), block_table.stride(0),
            partO.stride(0), partO.stride(1), partO.stride(2), partLse.stride(0), partLse.stride(1),
            pool_lat.stride(0), pool_lat.stride(1), pool_rope.stride(0), pool_rope.stride(1),
            R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC, BLOCK_N=64,
            NUM_SPLITS=NSPL, NUM_BLOCKS_LOOKUP=num_blocks)
        _splitk_stage2[(B, NH)](
            partO, partLse, O, Lse,
            partO.stride(0), partO.stride(1), partO.stride(2), partLse.stride(0), partLse.stride(1),
            O.stride(0), O.stride(1), Lse.stride(0), R=R, NUM_SPLITS=NSPL)
        cos = torch.nn.functional.cosine_similarity(O.flatten(), Oref.flatten(), dim=0).item()
        lse_err = (Lse - Lref).abs().max().item()
        print(f"NUM_SPLITS={NSPL}: o_cos={cos:.6f} lse_max_err={lse_err:.5f}",
              "PASS" if cos > 0.9999 and lse_err < 0.01 else "FAIL")


if __name__ == "__main__":
    _test()
