"""Phase 3 Step A: PyTorch reference for tile-aware KVarN-MLA decode.
Sequence = N full int4 tiles (pack_tile) + an fp16 tail (<GROUP tokens). One
decode query attends over all of it. Validates the full decode math incl. the
V-side un-rotate, vs FP16 full attention. (Triton port follows once math is OK.)
"""
import torch, torch.nn.functional as F, sys
sys.path.insert(0, "/mnt/nvme1/KVarN")
sys.path.insert(0, "/tmp")
from kvarn_mla_tilepack import (pack_tile, unpack_tile, hadamard, R, ROPE, GROUP, H)

dev = "cuda"
SM = 1.0 / ((R + ROPE) ** 0.5)


def decode_ref(q_lat, q_rope, full_lat, full_rope, tail_lat, tail_rope):
    """q_*: [Hh, .]. full_*: list of dequant'd ROTATED tiles + rope. tail_*: fp16
    rotated latent + rope for the unflushed tail. Returns o [Hh, R] (un-rotated)."""
    qH = q_lat @ H                                   # rotated query [Hh, R]
    # concatenate all rotated keys + rope
    K_rot = torch.cat(full_lat + [tail_lat], 0)      # [T, R] rotated
    K_rope = torch.cat(full_rope + [tail_rope], 0)   # [T, ROPE]
    scores = (qH @ K_rot.t() + q_rope @ K_rope.t()) * SM   # [Hh, T]
    p = F.softmax(scores, -1)
    acc_rot = p @ K_rot                              # [Hh, R] = o @ H (rotated)
    return acc_rot @ H.t()                           # un-rotate -> o [Hh, R]


def main():
    lat_full = torch.load("/tmp/v2lite_latent.pt").float().cuda()
    T = 400                                          # 3 full tiles + 16 tail
    lat = lat_full[:T]
    rope = torch.randn(T, ROPE, device=dev).to(torch.float16)
    Hh = 8
    q_lat = torch.randn(Hh, R, device=dev)
    q_rope = torch.randn(Hh, ROPE, device=dev)

    # FP16 reference (latent as K and V, rope appended to K only)
    sc = (q_lat @ lat.t() + q_rope @ rope.float().t()) * SM
    o_ref = F.softmax(sc, -1) @ lat

    # KVarN: pack full tiles, keep tail fp16
    n_full = T // GROUP
    full_lat, full_rope = [], []
    for i in range(n_full):
        rec = pack_tile(lat[i*GROUP:(i+1)*GROUP], rope[i*GROUP:(i+1)*GROUP])
        d, rp = unpack_tile(rec)
        full_lat.append(d); full_rope.append(rp.float())
    tail = lat[n_full*GROUP:]
    tail_lat = tail @ H                              # tail stored fp16 ROTATED
    tail_rope = rope[n_full*GROUP:].float()
    o_kv = decode_ref(q_lat, q_rope, full_lat, full_rope, tail_lat, tail_rope)

    cos = F.cosine_similarity(o_ref.flatten(), o_kv.flatten(), 0).item()
    rel = ((o_kv - o_ref).norm() / o_ref.norm()).item()
    print(f"T={T} n_full_tiles={n_full} tail={T-n_full*GROUP}")
    print(f"decode attn-output cos={cos:.5f} rel_err={rel:.4f}")
    print("DECODE_REF_OK" if cos > 0.999 else "DECODE_REF_MISMATCH")


if __name__ == "__main__":
    main()
