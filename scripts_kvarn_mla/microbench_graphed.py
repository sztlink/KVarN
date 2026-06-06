"""Graphed micro-bench: measure decode kernel vs rotation cost UNDER CUDA graphs
(the production burst path), since eager timing inflates rotation by launch
overhead that graphs remove. Captures (a) decode-only and (b) rotation-only in
CUDA graphs at GLM shape, reports graphed us/call -> ms/step (x47)."""
import sys, torch
sys.path.insert(0, "/mnt/nvme1/KVarN/scripts_kvarn_mla")
from kvarn_mla_grouped_decode import _grouped_stage1, _grouped_stage2
from kvarn_mla_tile_decode_kernel import tile_layout
from vllm.model_executor.layers.quantization.kvarn.sinkhorn import variance_normalize

dev = "cuda"
R, ROPE, G, bits = 512, 64, 128, 4
NB, SC, ZP, SR, RP, REC = tile_layout(R, ROPE, G, bits)
B, NH, LAYERS = 16, 20, 47
HGROUP = 16; n_hg = (NH + HGROUP - 1) // HGROUP; NSPL = 16
seq_len = 512; n_full = seq_len // G
num_blocks = B * (n_full + 1); qmax = (1 << bits) - 1
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
        block_table[b, j] = bid; bid += 1
seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
sm = 1.0 / (R + ROPE) ** 0.5
Q = torch.randn(B, NH, R + ROPE, device=dev)
partO = torch.zeros(B, NH, NSPL, R, dtype=torch.float32, device=dev)
partLse = torch.full((B, NH, NSPL), -float("inf"), dtype=torch.float32, device=dev)
O = torch.zeros(B, NH, R, dtype=torch.float32, device=dev)
Lse = torch.zeros(B, NH, dtype=torch.float32, device=dev)
# Match the LIVE kernel: rotation is bf16 tensor-core (q @ Hb, Hb bf16), not fp32.
Hb = Hd.to(torch.bfloat16)
qf = Q[:, :, :R].to(torch.bfloat16).reshape(B * NH, R)
Of = O.to(torch.bfloat16).reshape(B * NH, R)

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

def rotate():
    torch.mm(qf, Hb); torch.mm(Of, Hb.t())

def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g

def time_graph(g, iters=200):
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000

gd = capture(decode); gr = capture(rotate)
d = time_graph(gd); r = time_graph(gr)
print(f"[graphed NSPL={NSPL}] decode kernel: {d:6.1f} us/call -> {d*LAYERS/1000:5.2f} ms/step")
print(f"[graphed NSPL={NSPL}] rotation     : {r:6.1f} us/call -> {r*LAYERS/1000:5.2f} ms/step")
print(f"[graphed] attn-path total: {(d+r)*LAYERS/1000:.2f} ms/step  (rotation share {r/(d+r)*100:.0f}%)")
print("GRAPHED_DONE")
