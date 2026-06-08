"""Micro-benchmark the KVarN-MLA decode components at GLM-4.7-Flash shape to find
the real bottleneck. MoE/dense are identical FP16-vs-KVarN, so the ~22ms/step gap
is the attention path: (a) grouped decode kernel, (b) per-layer rotation matmuls
(q_lat@H + O@Hᵀ). Times each per-call -> x47 layers -> ms/step, vs the 22ms gap.
"""
import sys, torch, triton
sys.path.insert(0, "/mnt/nvme1/KVarN/scripts_kvarn_mla")
from kvarn_mla_grouped_decode import _grouped_stage1, _grouped_stage2
from kvarn_mla_tile_decode_kernel import tile_layout
from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize

dev = "cuda"
R, ROPE, G, bits = 512, 64, 128, 4
NB, SC, ZP, SR, RP, REC = tile_layout(R, ROPE, G, bits)
B, NH, LAYERS = 16, 20, 47
HGROUP = 16; n_hg = (NH + HGROUP - 1) // HGROUP; NSPL = 4
seq_len = 512; n_full = seq_len // G   # 4 full blocks (approx steady state)
num_blocks = B * (n_full + 1)
qmax = (1 << bits) - 1

Hm = torch.ones(1, 1)
while Hm.shape[0] < R:
    Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
Hd = (Hm / R ** 0.5).to(dev).float()

cache = torch.zeros(num_blocks * REC, dtype=torch.uint8, device=dev)
pool_lat = torch.zeros(B + 4, G, R, dtype=torch.float16, device=dev)
pool_rope = torch.zeros(B + 4, G, ROPE, dtype=torch.float16, device=dev)
block2slot = torch.full((num_blocks,), -1, dtype=torch.int32, device=dev)
block_table = torch.full((B, n_full + 2), -1, dtype=torch.int32, device=dev)
seqlens = []; bid = 0; slot = 0
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
        rec[SC:SC + R*2] = (scale*s_row).squeeze(1).to(torch.float16).view(torch.uint8)
        rec[ZP:ZP + R*2] = (lo*s_row).squeeze(1).to(torch.float16).view(torch.uint8)
        rec[SR:SR + G*2] = s_col.squeeze(0).to(torch.float16).view(torch.uint8)
        bid += 1
seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
sm = 1.0 / (R + ROPE) ** 0.5
Q = torch.randn(B, NH, R + ROPE, device=dev)
partO = torch.zeros(B, NH, NSPL, R, dtype=torch.float32, device=dev)
partLse = torch.full((B, NH, NSPL), -float("inf"), dtype=torch.float32, device=dev)
O = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
Lse = torch.zeros(B, NH, dtype=torch.float32, device=dev)


def time_us(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def decode():
    _grouped_stage1[(B, n_hg, NSPL)](
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


qf = Q[:, :, :R].float().reshape(B * NH, R)
Of = O.reshape(B * NH, R)
def rotate():
    _ = qf @ Hd          # q_lat @ H
    _ = Of @ Hd.t()      # O @ Hᵀ


d_us = time_us(decode)
r_us = time_us(rotate)
print(f"grouped decode kernel: {d_us:8.1f} us/call -> {d_us*LAYERS/1000:6.2f} ms/step (x{LAYERS})")
print(f"rotation matmuls     : {r_us:8.1f} us/call -> {r_us*LAYERS/1000:6.2f} ms/step (x{LAYERS})")
print(f"(measured KVarN-vs-FP16 gap was ~22 ms/step; FP16 step ~9.6ms, KVarN ~32ms)")
