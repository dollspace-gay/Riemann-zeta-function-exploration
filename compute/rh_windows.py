"""
Sliding-window interlacing: is lambda_min a LOCAL quantity?
============================================================
By Cauchy interlacing, lambda_min(G) <= min over all k x k principal
submatrices of lambda_min. For contiguous windows of k zeros, define
    W_k = min_i lambda_min( G[i:i+k, i:i+k] ).
If W_k -> lambda_min(G) rapidly in k, the null direction is supported on a
small cluster of consecutive zeros, and the kappa ~ N^0.6 scaling law reduces
to the extreme-value statistics of a local cluster functional.

Also reports where the lambda_min eigenvector localizes (top sites and
participation ratio) and where the best window sits.
"""

import torch
import numpy as np
import os, time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
DATA_DIR = os.path.expanduser("~/rh_data")
OUTPUT_DIR = os.path.expanduser("~/rh_output")

zeros_list = []
with open(os.path.join(DATA_DIR, "zeros6")) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                v = float(line.split()[-1])
                if v > 0:
                    zeros_list.append(v)
            except ValueError:
                continue
zeros_np = np.array(zeros_list, dtype=np.float64)

def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

all_primes = sieve_primes(10_000_000).astype(np.float64)
log_primes_np = np.log(all_primes)

def build_gram(gammas, n_primes, batch_size=2000):
    N = len(gammas)
    P = min(n_primes, len(all_primes))
    gamma = torch.tensor(gammas, dtype=DTYPE, device=DEVICE)
    G = torch.zeros((N, N), dtype=DTYPE, device=DEVICE)
    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        bp = torch.tensor(all_primes[start:end], dtype=DTYPE, device=DEVICE)
        blp = torch.tensor(log_primes_np[start:end], dtype=DTYPE, device=DEVICE)
        w = blp / torch.sqrt(bp)
        pa = -gamma.unsqueeze(1) * blp.unsqueeze(0)
        sw = torch.sqrt(w).unsqueeze(0)
        wc = torch.cos(pa) * sw
        ws = torch.sin(pa) * sw
        G += wc @ wc.T + ws @ ws.T
    G *= N / torch.trace(G)
    return G

def window_min(G, k):
    """min over i of lambda_min(G[i:i+k, i:i+k]), via batched eigvalsh."""
    N = G.shape[0]
    idx = torch.arange(N - k + 1, device=DEVICE).unsqueeze(1)
    off = torch.arange(k, device=DEVICE).unsqueeze(0)
    rows = (idx + off)  # [N-k+1, k]
    blocks = G[rows.unsqueeze(2), rows.unsqueeze(1)]  # [N-k+1, k, k]
    ev = torch.linalg.eigvalsh(blocks)[:, 0]
    j = int(torch.argmin(ev).item())
    return float(ev[j].item()), j

out = open(os.path.join(OUTPUT_DIR, "windows_results.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

KS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64]
for N in [1000, 2000, 4000, 8000]:
    P = min(20 * N, len(all_primes))
    g = zeros_np[:N]
    t0 = time.time()
    G = build_gram(g, P)
    evals, evecs = torch.linalg.eigh(G)
    lmin = float(evals[0].item())
    v = evecs[:, 0].cpu().numpy()
    pr = 1.0 / np.sum(v**4)
    top = np.argsort(v**2)[::-1][:6]
    gaps = np.diff(g)
    report(f"\nN={N}  P={P}  lambda_min={lmin:.6f}  PR={pr:.1f}")
    report(f"  eigvec top sites: {list(top)}  weights {np.round(v[top]**2, 3)}")
    report(f"  gaps near top site {top[0]}: "
           f"{np.round(gaps[max(0,top[0]-3):top[0]+3], 3)}")
    report(f"  global min gap: {gaps.min():.4f} at {gaps.argmin()}")
    report(f"  {'k':>4} {'W_k':>10} {'W_k/lmin':>9} {'argmin window':>14}")
    for k in KS:
        Wk, j = window_min(G, k)
        report(f"  {k:>4} {Wk:>10.6f} {Wk/lmin:>9.3f} {f'[{j},{j+k})':>14}")
    report(f"  [{time.time()-t0:.0f}s]")

out.close()
