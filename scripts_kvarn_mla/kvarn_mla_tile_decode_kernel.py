"""Stage 2: fused dual-source KVarN-MLA tile-decode kernel (the fast path).

Per (sequence, query-head) program, online-softmax over the whole context,
reading each block from ONE of two sources in the ROTATED frame:
  - flushed blocks  -> int4 tile cache, dequantized in-register
                       deq_rot[t,c] = (nibble[t,c]*scale[c] + zp[c]) * s_row[t]
  - in-progress block -> sparse fp16 pool[slot] (already c@H rotated)
Query is pre-rotated (q_lat@H); output stays rotated (caller un-rotates by Hᵀ,
or folds Hᵀ into W_UV). No fp16 materialization of the KV — int4 read straight
into registers. Tile layout = kvarn_mla_tile_layout (per-channel scale/zp +
per-token s_row, token-major pack, fp16 rope).

Validated standalone here (cos vs a torch reference over a mixed
pooled/flushed multi-block batch) before wiring into TritonMLAImpl.forward_mqa.
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/mnt/nvme1/KVarN/vllm/v1/attention/backends/mla")


def tile_layout(R, ROPE, G, bits):
    nb = G * R * bits // 8
    sc = nb
    zp = sc + R * 2
    sr = zp + R * 2
    rp = sr + G * 2
    rec = rp + G * ROPE * 2
    return nb, sc, zp, sr, rp, rec


@triton.jit
def _kvarn_mla_tile_decode_kernel(
    Q_ptr,             # [B, H, R+ROPE] fp32/fp16 (q_lat ALREADY rotated | q_rope)
    Cache_ptr,         # [num_blocks * REC] uint8  (int4 tile records)
    PoolLat_ptr,       # [POOL, G, R] fp16   (rotated)
    PoolRope_ptr,      # [POOL, G, ROPE] fp16
    BlockTable_ptr,    # [B, max_blocks] int32
    Seqlens_ptr,       # [B] int32
    Block2Slot_ptr,    # [num_blocks] int32  (-1 = flushed/int4)
    O_ptr,             # [B, H, R] fp32  (rotated output)
    Lse_ptr,           # [B, H] fp32
    sm_scale,
    stride_qb, stride_qh, stride_btb, stride_ob, stride_oh, stride_lb,
    stride_plb, stride_plt, stride_prb, stride_prt,
    R: tl.constexpr, ROPE: tl.constexpr, G: tl.constexpr,
    NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr, SR: tl.constexpr,
    RP: tl.constexpr, REC: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(Seqlens_ptr + b)
    offs_r = tl.arange(0, R)
    offs_rope = tl.arange(0, ROPE)
    half = tl.arange(0, R // 2)
    qbase = Q_ptr + b * stride_qb + h * stride_qh
    q_lat = tl.load(qbase + offs_r).to(tl.float32)          # rotated query latent
    q_rope = tl.load(qbase + R + offs_rope).to(tl.float32)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([R], dtype=tl.float32)

    n_blocks = (seq_len + G - 1) // G
    for j in range(0, n_blocks):
        block_id = tl.load(BlockTable_ptr + b * stride_btb + j)
        slot = tl.load(Block2Slot_ptr + block_id,
                       mask=(block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP),
                       other=-1)
        base = block_id.to(tl.int64) * REC
        # per-channel scale/zp (int4 path) — loaded once per block, unused if pooled
        sc = tl.load((Cache_ptr + base + SC).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)          # [R]
        zp = tl.load((Cache_ptr + base + ZP).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)          # [R]
        for c0 in range(0, G, BLOCK_N):
            offs_n = c0 + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < (seq_len - j * G)
            if slot >= 0:
                # pooled rotated fp16
                pl = PoolLat_ptr + slot.to(tl.int64) * stride_plb + offs_n[:, None] * stride_plt + offs_r[None, :]
                deq = tl.load(pl, mask=tok_mask[:, None], other=0.0).to(tl.float32)   # [BN,R]
                pr = PoolRope_ptr + slot.to(tl.int64) * stride_prb + offs_n[:, None] * stride_prt + offs_rope[None, :]
                rope = tl.load(pr, mask=tok_mask[:, None], other=0.0).to(tl.float32)  # [BN,ROPE]
            else:
                # int4 tile dequant (token-major: token t has R/2 packed bytes)
                pk_ptr = Cache_ptr + base + offs_n[:, None] * (R // 2) + half[None, :]
                pk = tl.load(pk_ptr, mask=tok_mask[:, None], other=0).to(tl.uint32)   # [BN,R/2]
                lo = (pk & 0xF).to(tl.float32)
                hi = ((pk >> 4) & 0xF).to(tl.float32)
                q4 = tl.interleave(lo, hi)                                            # [BN,R]
                srow = tl.load((Cache_ptr + base + SR).to(tl.pointer_type(tl.float16)) + offs_n,
                               mask=tok_mask, other=0.0).to(tl.float32)               # [BN]
                deq = (q4 * sc[None, :] + zp[None, :]) * srow[:, None]                # [BN,R]
                rp_ptr = (Cache_ptr + base + RP).to(tl.pointer_type(tl.float16)) + offs_n[:, None] * ROPE + offs_rope[None, :]
                rope = tl.load(rp_ptr, mask=tok_mask[:, None], other=0.0).to(tl.float32)  # [BN,ROPE]
            score = (tl.sum(deq * q_lat[None, :], axis=1)
                     + tl.sum(rope * q_rope[None, :], axis=1)) * sm_scale            # [BN]
            score = tl.where(tok_mask, score, -float("inf"))
            chunk_max = tl.max(score, axis=0)
            new_max = tl.maximum(e_max, chunk_max)
            p = tl.exp(score - new_max)                                              # [BN]
            alpha = tl.exp(e_max - new_max)
            e_sum = e_sum * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * deq, axis=0)                     # [R]
            e_max = new_max
    tl.store(O_ptr + b * stride_ob + h * stride_oh + offs_r, acc / e_sum)
    tl.store(Lse_ptr + b * stride_lb + h, e_max + tl.log(e_sum))


def _ref_and_test():
    torch.manual_seed(0)
    dev = "cuda"
    R, ROPE, G, bits = 512, 64, 128, 4
    NB, SC, ZP, SR, RP, REC = tile_layout(R, ROPE, G, bits)
    qmax = (1 << bits) - 1
    H = torch.ones(1, 1)
    while H.shape[0] < R:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    Hd = (H / R ** 0.5).to(dev).float()                                  # [R,R]

    B, NH = 2, 4
    num_blocks = 16
    cache = torch.zeros(num_blocks * REC, dtype=torch.uint8, device=dev)
    pool_slots = 4
    pool_lat = torch.zeros(pool_slots, G, R, dtype=torch.float16, device=dev)
    pool_rope = torch.zeros(pool_slots, G, ROPE, dtype=torch.float16, device=dev)
    block2slot = torch.full((num_blocks,), -1, dtype=torch.int32, device=dev)

    # Per sequence: 2 full flushed blocks + 1 partial pooled block (40 tok).
    seqlens = []
    block_table = torch.full((B, 8), -1, dtype=torch.int32, device=dev)
    ref_lat_rot = {}   # (b,j) -> [n,R] rotated latent
    ref_rope = {}
    slot_ctr = 0
    bid_ctr = 0
    for b in range(B):
        n_full, partial = 2, 40
        slen = n_full * G + partial
        seqlens.append(slen)
        for j in range(n_full + 1):
            bid = bid_ctr; bid_ctr += 1
            block_table[b, j] = bid
            n = G if j < n_full else partial
            c = torch.randn(n, R, device=dev)            # raw latent
            crot = (c @ Hd)                              # rotated [n,R]
            rope = torch.randn(n, ROPE, device=dev)
            ref_lat_rot[(b, j)] = crot
            ref_rope[(b, j)] = rope
            if j < n_full:
                # flush to int4 tile (mirror _kvarn_flush_tile, already_rotated)
                from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize
                rot = crot.t().contiguous()              # [R,G]
                bal, s_col, s_row = variance_normalize(rot)
                lo = bal.amin(1, keepdim=True); hi = bal.amax(1, keepdim=True)
                scale = ((hi - lo) / qmax).clamp_min(1e-8)
                q = torch.clamp(torch.round((bal - lo) / scale), 0, qmax).to(torch.uint8)
                scale_abs = (scale * s_row).squeeze(1)
                zp_abs = (lo * s_row).squeeze(1)
                per_tok = s_col.squeeze(0)
                qT = q.t().contiguous()
                packed = (qT[:, 0::2] | (qT[:, 1::2] << 4)).contiguous()
                rec = cache.view(-1, REC)[bid]
                rec[:NB] = packed.reshape(-1)
                rec[SC:SC + R * 2] = scale_abs.to(torch.float16).view(torch.uint8)
                rec[ZP:ZP + R * 2] = zp_abs.to(torch.float16).view(torch.uint8)
                rec[SR:SR + G * 2] = per_tok.to(torch.float16).view(torch.uint8)
                rec[RP:RP + G * ROPE * 2] = rope.reshape(-1).to(torch.float16).view(torch.uint8)
                # Reference uses the SAME dequantized tile (isolate kernel logic
                # from 4-bit quant error): deq[t,c]=(q[t,c]*scale_abs[c]+zp_abs[c])*per_tok[t]
                qd = q.t().float()                                  # [G,R]
                deq = (qd * scale_abs.float()[None, :] + zp_abs.float()[None, :]) * per_tok.float()[:, None]
                ref_lat_rot[(b, j)] = deq
                ref_rope[(b, j)] = rope.half().float()
            else:
                slot = slot_ctr; slot_ctr += 1
                block2slot[bid] = slot
                pool_lat[slot, :n] = crot.to(torch.float16)
                pool_rope[slot, :n] = rope.to(torch.float16)
                ref_lat_rot[(b, j)] = crot.half().float()           # pool is fp16
                ref_rope[(b, j)] = rope.half().float()

    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    sm_scale = 1.0 / (R + ROPE) ** 0.5
    # query (rotated latent | rope)
    q_lat_raw = torch.randn(B, NH, R, device=dev)
    q_rope = torch.randn(B, NH, ROPE, device=dev)
    q_lat_rot = q_lat_raw @ Hd
    Q = torch.cat([q_lat_rot, q_rope], dim=-1).contiguous()

    O = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
    Lse = torch.zeros(B, NH, dtype=torch.float32, device=dev)
    _kvarn_mla_tile_decode_kernel[(B, NH)](
        Q, cache, pool_lat, pool_rope, block_table, seqlens_t, block2slot, O, Lse,
        sm_scale,
        Q.stride(0), Q.stride(1), block_table.stride(0), O.stride(0), O.stride(1), Lse.stride(0),
        pool_lat.stride(0), pool_lat.stride(1), pool_rope.stride(0), pool_rope.stride(1),
        R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC,
        BLOCK_N=32, NUM_BLOCKS_LOOKUP=num_blocks,
    )

    # Reference: gather rotated latents + rope per seq, attention in rotated frame.
    cos_all = []
    for b in range(B):
        nblk = (seqlens[b] + G - 1) // G
        Krot = torch.cat([ref_lat_rot[(b, j)] for j in range(nblk)], 0)   # [slen,R]
        Krope = torch.cat([ref_rope[(b, j)] for j in range(nblk)], 0)     # [slen,ROPE]
        for h in range(NH):
            ql = q_lat_rot[b, h]; qr = q_rope[b, h]
            sc = (ql @ Krot.t() + qr @ Krope.t()) * sm_scale
            p = torch.softmax(sc, -1)
            o_ref = p @ Krot                                              # [R] rotated
            o_ker = O[b, h]
            cos = torch.nn.functional.cosine_similarity(o_ref, o_ker, dim=0).item()
            cos_all.append(cos)
    mn = min(cos_all)
    print(f"TILE DECODE KERNEL cos min={mn:.5f} mean={sum(cos_all)/len(cos_all):.5f}",
          "PASS" if mn > 0.999 else "FAIL")


if __name__ == "__main__":
    _ref_and_test()
