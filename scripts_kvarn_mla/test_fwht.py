"""FWHT for the KVarN-MLA rotation: replace dense [M,512]@[512,512] Hadamard with
the Kronecker factorization H512 = H16 (x) H32 (Sylvester order) -> two small
batched matmuls. Verify EXACTNESS vs dense H and compare graphed speed.
Run standalone on one GPU."""
import torch
dev = "cuda"; R = 512; M = 16 * 20  # B*NH at GLM shape
torch.manual_seed(0)

def sylvester(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H

H512 = (sylvester(512) / 512 ** 0.5).to(dev).bfloat16()           # normalized, as live kernel
H16 = sylvester(16).to(dev).bfloat16()
H32 = sylvester(32).to(dev).bfloat16()
norm = 1.0 / 512 ** 0.5
x = torch.randn(M, R, device=dev, dtype=torch.bfloat16)

def dense(x):
    return x @ H512

def kron(x):
    # x@(H16 (x) H32): reshape [M,16,32], contract H32 on last, H16 on middle
    t = x.view(M, 16, 32)
    t = t @ H32                          # [M,16,32]
    t = torch.einsum('mij,ik->mkj', t, H16)
    return (t.reshape(M, R) * norm)

yd = dense(x).float(); yk = kron(x).float()
cos = torch.nn.functional.cosine_similarity(yd.flatten(), yk.flatten(), 0).item()
rel = ((yk - yd).norm() / yd.norm()).item()
print(f"[fwht] exactness vs dense H: cos={cos:.6f} rel_err={rel:.5f}")

def cap(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn(x)
    return g

def t_us(g, it=500):
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1000

gd = cap(dense); gk = cap(kron)
td = t_us(gd); tk = t_us(gk)
print(f"[fwht] dense matmul: {td:.2f} us   kron(16x32): {tk:.2f} us   speedup={td/tk:.2f}x")
print(f"[fwht] (rotation is done TWICE/layer q@H + O@Ht; x47 layers)")
print("FWHT_DONE")
