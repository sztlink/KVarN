"""Phase 3: Triton dequant kernel for the KVarN-MLA tile cache.
Per block: unpack int4 (token-major) + per-channel scale/zp + per-token s_row ->
fp16 ROTATED latent [GROUP,R] and copy fp16 rope [GROUP,ROPE] into a scratch
fp16 buffer. This is the 'dequant' half of dequant->stock-attention. Validated
vs the PyTorch unpack_tile."""
import torch, triton, triton.language as tl, sys
sys.path.insert(0, "/mnt/nvme1/KVarN")
sys.path.insert(0, "/tmp")
from kvarn_mla_tilepack import (pack_tile, unpack_tile, R, ROPE, GROUP,
                                NB, SC, ZP, SR, RP, REC)
dev = "cuda"


@triton.jit
def _dequant_tile(Cache, OutLat, OutRope,
                  stride_cb, stride_olb, stride_olt, stride_orb, stride_ort,
                  R: tl.constexpr, ROPE: tl.constexpr, GROUP: tl.constexpr,
                  NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr,
                  SR: tl.constexpr, RP: tl.constexpr):
    blk = tl.program_id(0)
    t = tl.program_id(1)                       # token within block
    base = Cache + blk * stride_cb
    offs_c = tl.arange(0, R)                    # channels
    half = tl.arange(0, R // 2)
    # packed token row: NB bytes laid out token-major, R/2 bytes per token
    pk = tl.load(base + t * (R // 2) + half).to(tl.uint32)   # [R/2]
    lo4 = (pk & 0xF).to(tl.float32)
    hi4 = ((pk >> 4) & 0xF).to(tl.float32)
    q = tl.interleave(lo4, hi4)                 # [R] token-major channel order
    scale = tl.load((base + SC).to(tl.pointer_type(tl.float16)) + offs_c).to(tl.float32)
    zp = tl.load((base + ZP).to(tl.pointer_type(tl.float16)) + offs_c).to(tl.float32)
    srow = tl.load((base + SR).to(tl.pointer_type(tl.float16)) + t).to(tl.float32)
    deq = (q * scale + zp) * srow               # rotated latent [R]
    tl.store(OutLat + blk * stride_olb + t * stride_olt + offs_c, deq.to(OutLat.dtype.element_ty))
    offs_r = tl.arange(0, ROPE)
    rope = tl.load((base + RP).to(tl.pointer_type(tl.float16)) + t * ROPE + offs_r)
    tl.store(OutRope + blk * stride_orb + t * stride_ort + offs_r, rope)


def main():
    lat_full = torch.load("/tmp/v2lite_latent.pt").float().cuda()
    nblk = 3
    cache = torch.zeros(nblk, REC, dtype=torch.uint8, device=dev)
    rope_in = torch.randn(nblk, GROUP, ROPE, device=dev).to(torch.float16)
    refs = []
    for b in range(nblk):
        rec = pack_tile(lat_full[b*GROUP:(b+1)*GROUP], rope_in[b])
        cache[b] = rec
        refs.append(unpack_tile(rec))
    out_lat = torch.zeros(nblk, GROUP, R, dtype=torch.float16, device=dev)
    out_rope = torch.zeros(nblk, GROUP, ROPE, dtype=torch.float16, device=dev)
    _dequant_tile[(nblk, GROUP)](
        cache, out_lat, out_rope,
        cache.stride(0), out_lat.stride(0), out_lat.stride(1),
        out_rope.stride(0), out_rope.stride(1),
        R=R, ROPE=ROPE, GROUP=GROUP, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP)
    # compare to PyTorch unpack
    maxlat = max((out_lat[b].float() - refs[b][0]).abs().max().item() for b in range(nblk))
    maxrope = max((out_rope[b].float() - refs[b][1].float()).abs().max().item() for b in range(nblk))
    cos = torch.nn.functional.cosine_similarity(
        out_lat.float().flatten(), torch.cat([r[0] for r in refs]).flatten(), 0).item()
    print(f"dequant kernel vs unpack_tile: lat_max_abs={maxlat:.2e} rope_max_abs={maxrope:.2e} cos={cos:.6f}")
    print("DEQUANT_KERNEL_OK" if maxlat < 1e-2 and maxrope < 1e-3 else "DEQUANT_KERNEL_MISMATCH")


if __name__ == "__main__":
    main()
