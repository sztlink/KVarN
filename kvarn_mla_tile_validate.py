"""Phase 1: validate the KVarN-MLA *tile* method (the real method, K-path on the
latent): Hadamard rotate -> Sinkhorn(512x128) -> per-channel asymmetric RTN ->
store ROTATED quantized tile + scale[R]+zp[R]+s_row[group]. Decode dots the
Hadamard-rotated query with the dequant'd rotated tile (no un-rotate in kernel).

Validates attention-score equivalence vs FP16 and vs the un-rotate probe, on a
REAL V2-Lite latent tile."""
import torch, torch.nn.functional as F, sys
sys.path.insert(0, "/mnt/nvme1/KVarN")
from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize

lat = torch.load("/tmp/v2lite_latent.pt").float().cuda()      # [T, R]
T, R = lat.shape
GROUP = 128
dev = lat.device


def hadamard(n, dev):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / n ** 0.5).to(dev)


H = hadamard(R, dev)


def flush_tile(tile, bits):
    """tile: [group, R] fp32 latent. Returns dequant'd ROTATED tile [group, R]."""
    g = tile.shape[0]
    rot = tile @ H                              # [g, R] rotated
    frame = rot.t().contiguous()                # [R, g] channels x tokens
    bal, s_col, s_row = variance_normalize(frame)   # bal=frame/s_col/s_row
    qmax = (1 << bits) - 1
    lo = bal.amin(1, keepdim=True)              # per-channel (R)
    hi = bal.amax(1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    q = torch.clamp(torch.round((bal - lo) / scale), 0, qmax)
    deq = (q * scale + lo) * s_col * s_row      # invert sinkhorn -> rotated frame [R,g]
    return deq.t()                              # [g, R] rotated


for bits in (4, 2):
    q = torch.randn(8, R, device=dev)           # 8 query heads (absorbed q_lat)
    qH = q @ H                                   # rotated query
    score_ref = q @ lat.t()                      # [8, T] FP16 scores
    # build dequant'd rotated tiles for all tokens
    deq_rot = torch.empty_like(lat)
    for s in range(0, T, GROUP):
        deq_rot[s:s + GROUP] = flush_tile(lat[s:s + GROUP], bits)
    score_kvarn = qH @ deq_rot.t()               # [8, T] rotated-query . rotated-tile
    sc_cos = F.cosine_similarity(score_ref.flatten(), score_kvarn.flatten(), 0).item()
    # full softmax attention output equivalence (latent as V too)
    o_ref = F.softmax(score_ref / R ** 0.5, -1) @ lat
    o_kv = F.softmax(score_kvarn / R ** 0.5, -1) @ (deq_rot @ H.t())  # V = unrotate(deq)
    o_cos = F.cosine_similarity(o_ref.flatten(), o_kv.flatten(), 0).item()
    print(f"{bits}-bit: score_cos={sc_cos:.5f}  attn_out_cos={o_cos:.5f}")
print("TILE_VALIDATE_DONE")
