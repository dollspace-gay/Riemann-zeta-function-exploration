"""
Fixed prime-count sweep (v4 open problem 3)
============================================
In rh_windows.py the prime count grew with N (P = 20N), so the slow growth
of the window plateau W_k / lambda_min could be either (a) a real delocalized
component of the null direction, or (b) an artifact of the growing prime set.

Here P is FIXED at 160,000 primes while N sweeps 1000..8000. If the plateau
ratio stops growing with N at fixed P, the delocalized component was a
prime-count artifact; if it still grows, it is real.
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
    N = G.shape[0]
    idx = torch.arange(N - k + 1, device=DEVICE).unsqueeze(1)
    off = torch.arange(k, device=DEVICE).unsqueeze(0)
    rows = (idx + off)
    blocks = G[rows.unsqueeze(2), rows.unsqueeze(1)]
    ev = torch.linalg.eigvalsh(blocks)[:, 0]
    j = int(torch.argmin(ev).item())
    return float(ev[j].item()), j

P_FIXED = 160_000
out = open(os.path.join(OUTPUT_DIR, "fixedP_results.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

report(f"Fixed P = {P_FIXED} primes (max prime {all_primes[P_FIXED-1]:.0f})")
report(f"{'N':>6} {'lmin':>10} {'kappa':>9} {'W3':>10} {'W3/lmin':>8} "
       f"{'W32':>10} {'W32/lmin':>9} {'argmin w3':>12}")

Ns, lmins, kappas = [], [], []
for N in [1000, 2000, 4000, 8000]:
    g = zeros_np[:N]
    t0 = time.time()
    G = build_gram(g, P_FIXED)
    ev = torch.linalg.eigvalsh(G)
    lmin, lmax = float(ev[0].item()), float(ev[-1].item())
    W3, j3 = window_min(G, 3)
    W32, _ = window_min(G, 32)
    Ns.append(N); lmins.append(lmin); kappas.append(lmax/lmin)
    report(f"{N:>6} {lmin:>10.6f} {lmax/lmin:>9.2f} {W3:>10.6f} {W3/lmin:>8.3f} "
           f"{W32:>10.6f} {W32/lmin:>9.3f} {f'[{j3},{j3+3})':>12}  [{time.time()-t0:.0f}s]")

fit_l = np.polyfit(np.log(Ns), np.log(lmins), 1)[0]
fit_k = np.polyfit(np.log(Ns), np.log(kappas), 1)[0]
report("")
report(f"Fixed-P exponents: lambda_min ~ N^{fit_l:+.3f}, kappa ~ N^{fit_k:+.3f}")
report("(Compare growing-P run: lambda_min ~ N^-0.53, kappa ~ N^+0.64,")
report(" plateau W32/lmin grew 1.21 -> 1.64 over N=1000..8000.)")
out.close()
