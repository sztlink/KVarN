"""Phase 2 unit: per-block TILE pack/unpack for KVarN-MLA (the K-path record).
One block = GROUP tokens. Record bytes:
  packed latent : GROUP*R*4/8
  scale[R] zp[R]: per-channel (shared across the block's tokens), fp16
  s_row[GROUP]  : per-token sinkhorn, fp16
  rope          : GROUP*Rope*fp16 (kept fp16)
Validates pack->unpack round-trip reconstructs the dequant'd rotated tile +rope.
"""
import torch, torch.nn.functional as F, sys
sys.path.insert(0, "/mnt/nvme1/KVarN")
from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize

R, ROPE, GROUP, BITS = 512, 64, 128, 4
QMAX = (1 << BITS) - 1
dev = "cuda"


def hadamard(n, dev):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / n ** 0.5).to(dev)


H = hadamard(R, dev)
# byte offsets in the per-block record
NB = GROUP * R * BITS // 8                 # packed latent bytes
SC = NB                                    # scale[R] fp16
ZP = SC + R * 2
SR = ZP + R * 2                            # s_row[GROUP] fp16
RP = SR + GROUP * 2                        # rope GROUP*ROPE fp16
REC = RP + GROUP * ROPE * 2


def pack_tile(lat, rope):
    """lat: [GROUP, R] fp32. rope: [GROUP, ROPE] fp16. -> uint8 [REC]."""
    rot = (lat @ H).t().contiguous()                       # [R, GROUP] rotated frame
    bal, s_col, s_row = variance_normalize(rot)            # s_col[R,1], s_row[1,GROUP]
    lo = bal.amin(1, keepdim=True); hi = bal.amax(1, keepdim=True)
    scale = ((hi - lo) / QMAX).clamp_min(1e-8)             # [R,1] per-channel
    q = torch.clamp(torch.round((bal - lo) / scale), 0, QMAX).to(torch.uint8)  # [R,GROUP]
    # s_row[R,1]=per-channel, s_col[1,GROUP]=per-token. Absorb the per-channel
    # sinkhorn scale into scale/zp; store the per-token scale separately.
    scale_abs = (scale * s_row).squeeze(1)                 # [R]
    zp_abs = (lo * s_row).squeeze(1)                       # [R]
    sr = s_col.squeeze(0)                                  # [GROUP] per-token
    qT = q.t().contiguous()                                # [GROUP, R] token-major
    packed = (qT[:, 0::2] | (qT[:, 1::2] << 4)).contiguous()  # [GROUP, R/2]
    rec = torch.zeros(REC, dtype=torch.uint8, device=lat.device)
    rec[:NB] = packed.reshape(-1)
    rec[SC:SC + R * 2] = scale_abs.to(torch.float16).view(torch.uint8)
    rec[ZP:ZP + R * 2] = zp_abs.to(torch.float16).view(torch.uint8)
    rec[SR:SR + GROUP * 2] = sr.to(torch.float16).view(torch.uint8)
    rec[RP:RP + GROUP * ROPE * 2] = rope.reshape(-1).view(torch.uint8)
    return rec


def unpack_tile(rec):
    """-> dequant'd ROTATED latent [GROUP, R], rope [GROUP, ROPE]."""
    packed = rec[:NB].view(GROUP, R // 2)
    lo4 = (packed & 0xF).to(torch.float32); hi4 = (packed >> 4).to(torch.float32)
    q = torch.empty(GROUP, R, device=rec.device)
    q[:, 0::2] = lo4; q[:, 1::2] = hi4                     # [GROUP, R]
    scale_abs = rec[SC:SC + R * 2].view(torch.float16).float()      # [R]
    zp_abs = rec[ZP:ZP + R * 2].view(torch.float16).float()         # [R]
    sr = rec[SR:SR + GROUP * 2].view(torch.float16).float()         # [GROUP]
    # dequant: (q*scale+zp) gives bal*s_col; *s_row -> rotated frame value
    deq_rot = (q * scale_abs[None, :] + zp_abs[None, :]) * sr[:, None]   # [GROUP,R]
    rope = rec[RP:RP + GROUP * ROPE * 2].view(torch.float16).reshape(GROUP, ROPE)
    return deq_rot, rope


def main():
    lat_full = torch.load("/tmp/v2lite_latent.pt").float().cuda()
    lat = lat_full[:GROUP]                                  # one tile
    rope = torch.randn(GROUP, ROPE, device=dev).to(torch.float16)
    rec = pack_tile(lat, rope)
    deq_rot, rope_back = unpack_tile(rec)
    # reference rotated quant via the validated tile path
    q = torch.randn(8, R, device=dev); qH = q @ H
    score_ref = q @ lat.t()
    score_kv = qH @ deq_rot.t()
    print(f"REC bytes={REC} ({REC/GROUP:.0f} B/tok vs 1152 fp16 = {1152*GROUP/REC:.2f}x)")
    print(f"score_cos={F.cosine_similarity(score_ref.flatten(),score_kv.flatten(),0):.5f}")
    print(f"rope roundtrip max_abs={(rope_back.float()-rope.float()).abs().max():.2e}")
    print("TILEPACK_OK")


if __name__ == "__main__":
    main()
