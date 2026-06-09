"""
Theory tests: divided-difference stencils and the scaling exponent
===================================================================
THEORY. Expanding c(delta) = sum_p w_hat(p) cos(delta log p)
              ~ 1 - (M2/2) d^2 + (M4/24) d^4 - ...
with M_{2k} = sum_p w_hat(p) (log p)^{2k}, pure algebra gives, for a test
vector x over zeros at positions t with sum x t^m = 0 for m < k
(the k-th divided-difference stencil):

    x^T G x  ~  (M_{2k} / (2k)!) * binom(2k,k) * (sum_i x_i t_i^k)^2.

Parameter-free predictions for planted clusters of span ~ delta:
    pair   (1,-1)/sqrt2     : lambda ~ (M2/2)  * delta^2
    triple (1,-2,1)/sqrt6   : lambda ~ (M4/6)  * delta^4
    quad   (1,-3,3,-1)/sqrt20: lambda ~ (M6/20) * delta^6

GUE extreme-value step: P(k+1 consecutive zeros within span s) ~ s^{k^2+2k}
per site, so channel k over N zeros gives lambda_min ~ N^{-2/(k+2)}:
    pairs -> N^{-2/3},  triples -> N^{-1/2},  quads -> N^{-2/5}.

TEST A: planted clusters in a rigid background; measure lambda_min(delta)
        and compare against the zero-parameter predictions above.
TEST B: stencil structure of the real lambda_min eigenvector at N=8000
        (signs + shape vs divided-difference stencils).
TEST C: GUE ensemble across N with multiple seeds: ensemble lambda_min
        exponent (predicted -1/2 in the triple regime, drifting to -2/3),
        PR of the localization, and the tail CDF of window eigenvalues
        (predicted P(W3 < t) ~ t^2 in the triple channel, t^{3/2} pair).
"""

import torch
import numpy as np
import os, time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
DATA_DIR = os.path.expanduser("~/rh_data")
OUTPUT_DIR = os.path.expanduser("~/rh_output")

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

out = open(os.path.join(OUTPUT_DIR, "theory_results.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

# =====================================================================
# TEST A: planted clusters, parameter-free coefficient check
# =====================================================================
report("=" * 78)
report("TEST A: planted clusters vs divided-difference predictions")
report("=" * 78)

N_BG = 500
P_A = 10_000
# weight moments over the SAME primes (normalized weights)
w_np = log_primes_np[:P_A] / np.sqrt(all_primes[:P_A])
w_hat = w_np / w_np.sum()
M2 = float(np.sum(w_hat * log_primes_np[:P_A]**2))
M4 = float(np.sum(w_hat * log_primes_np[:P_A]**4))
M6 = float(np.sum(w_hat * log_primes_np[:P_A]**6))
report(f"Moments (P={P_A}): M2={M2:.2f}  M4={M4:.1f}  M6={M6:.0f}")
report(f"Quadratic regime requires delta << 1/log(p_max) = {1/log_primes_np[P_A-1]:.4f}")

def planted(kind, delta):
    """Rigid unit-spaced background of N_BG zeros with a cluster planted
    at the center. Cluster zeros replace background zeros."""
    g = np.arange(N_BG, dtype=np.float64) * 1.0 + 100.0
    c = N_BG // 2
    if kind == "pair":
        g[c+1] = g[c] + delta
    elif kind == "triple":
        g[c+1] = g[c] + delta
        g[c+2] = g[c] + 2*delta
    elif kind == "quad":
        g[c+1] = g[c] + delta
        g[c+2] = g[c] + 2*delta
        g[c+3] = g[c] + 3*delta
    return g

preds = {
    "pair":   lambda d: (M2/2) * d**2,
    "triple": lambda d: (M4/6) * d**4,
    "quad":   lambda d: (M6/20) * d**6,
}

deltas = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
for kind in ["pair", "triple", "quad"]:
    report(f"\n  {kind}:  (prediction: {['','(M2/2) d^2','(M4/6) d^4','(M6/20) d^6'][['','pair','triple','quad'].index(kind)]})")
    report(f"  {'delta':>7} {'lmin':>12} {'predicted':>12} {'ratio':>7}")
    lams = []
    for d in deltas:
        G = build_gram(planted(kind, d), P_A)
        lmin = float(torch.linalg.eigvalsh(G)[0].item())
        pred = preds[kind](d)
        lams.append(lmin)
        report(f"  {d:>7.3f} {lmin:>12.3e} {pred:>12.3e} {lmin/pred:>7.3f}")
    # local slope in the quadratic regime (first three deltas)
    sl = np.polyfit(np.log(deltas[:3]), np.log(lams[:3]), 1)[0]
    report(f"  fitted slope (small delta): {sl:.2f}   "
           f"(theory: {dict(pair=2, triple=4, quad=6)[kind]})")

# =====================================================================
# TEST B: stencil structure of the real lambda_min eigenvector (N=8000)
# =====================================================================
report("")
report("=" * 78)
report("TEST B: eigenvector stencil structure, real zeros, N=8000")
report("=" * 78)

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

N = 8000
G = build_gram(zeros_np[:N], 160_000)
evals, evecs = torch.linalg.eigh(G)
v = evecs[:, 0].cpu().numpy()
site = int(np.argmax(v**2))
lo, hi = site - 4, site + 5
report(f"lambda_min = {float(evals[0]):.6f}, localization site {site}")
report(f"{'idx':>6} {'gamma':>12} {'v_i':>9} {'v_i^2':>7}")
for ix in range(lo, hi):
    report(f"{ix:>6} {zeros_np[ix]:>12.4f} {v[ix]:>9.4f} {v[ix]**2:>7.3f}")
s = np.sign(v[np.abs(v) > 0.05 * np.abs(v).max()])
report(f"sign pattern of significant entries: {s.astype(int)}")
report(f"sum of v over window [{lo},{hi}): {v[lo:hi].sum():+.4f} "
       f"(divided-difference stencils have sum ~ 0)")
tw = zeros_np[lo:hi] - zeros_np[site]
report(f"sum of v*t over window: {(v[lo:hi]*tw).sum():+.4f} "
       f"(~0 would indicate k>=2 stencil)")

# =====================================================================
# TEST C: GUE ensemble — exponents, PR, and W3 tail
# =====================================================================
report("")
report("=" * 78)
report("TEST C: GUE ensemble (8 seeds): lambda_min exponent, PR, W3 tail")
report("=" * 78)

def smooth_count(T):
    return T / (2*np.pi) * np.log(T / (2*np.pi*np.e)) + 7.0/8.0

def inv_smooth_count(u, T_init):
    T = np.full_like(u, T_init, dtype=np.float64)
    for _ in range(60):
        f = smooth_count(T) - u
        df = np.log(T / (2*np.pi)) / (2*np.pi)
        T = np.clip(T - f / df, 10.0, None)
    return T

def gue_zeros(Nz, rng):
    M = 2 * Nz
    A = torch.tensor(rng.standard_normal((M, M)), dtype=DTYPE, device=DEVICE) \
        + 1j * torch.tensor(rng.standard_normal((M, M)), dtype=DTYPE, device=DEVICE)
    H = (A + A.conj().T) / 2.0
    ev = torch.linalg.eigvalsh(H).cpu().numpy() / np.sqrt(M)
    x = np.clip(ev, -2.0, 2.0)
    F = 0.5 + (x * np.sqrt(4.0 - x**2) + 4.0 * np.arcsin(x / 2.0)) / (4.0 * np.pi)
    u = M * F
    lo = (M - Nz) // 2
    u = u[lo:lo+Nz]
    u = (u - u[0]) / np.mean(np.diff(u))
    u = u + smooth_count(14.13)
    return inv_smooth_count(u, 14.13)

def window_eigs(G, k):
    Nw = G.shape[0]
    idx = torch.arange(Nw - k + 1, device=DEVICE).unsqueeze(1)
    off = torch.arange(k, device=DEVICE).unsqueeze(0)
    rows = (idx + off)
    blocks = G[rows.unsqueeze(2), rows.unsqueeze(1)]
    return torch.linalg.eigvalsh(blocks)[:, 0].cpu().numpy()

SEEDS = 8
NS = [1000, 2000, 4000, 8000]
all_W3 = []
report(f"{'N':>6} {'<lmin>':>10} {'med lmin':>10} {'<kappa>':>9} {'<PR>':>6}")
ens = {Nz: [] for Nz in NS}
for Nz in NS:
    lmins, kappas, prs = [], [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        g = gue_zeros(Nz, rng)
        G = build_gram(g, 20 * Nz)
        evals, evecs = torch.linalg.eigh(G)
        lmin, lmax = float(evals[0]), float(evals[-1])
        vv = evecs[:, 0].cpu().numpy()
        lmins.append(lmin); kappas.append(lmax/lmin)
        prs.append(1.0/np.sum(vv**4))
        if Nz == 4000:
            all_W3.append(window_eigs(G, 3))
    ens[Nz] = lmins
    report(f"{Nz:>6} {np.mean(lmins):>10.6f} {np.median(lmins):>10.6f} "
           f"{np.mean(kappas):>9.2f} {np.mean(prs):>6.2f}")

means = [np.mean(ens[Nz]) for Nz in NS]
meds = [np.median(ens[Nz]) for Nz in NS]
report("")
report(f"Ensemble exponents: mean lambda_min ~ N^{np.polyfit(np.log(NS), np.log(means), 1)[0]:+.3f}, "
       f"median ~ N^{np.polyfit(np.log(NS), np.log(meds), 1)[0]:+.3f}")
report("Theory: triple channel N^-0.50, pair channel N^-0.667 (asymptotic)")

# W3 tail CDF at N=4000 pooled over seeds
W3 = np.sort(np.concatenate(all_W3))
report("")
report("W3 tail (pooled, N=4000 x 8 seeds): P(W3 < t) ~ t^alpha")
qs = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
ts = [float(np.quantile(W3, q)) for q in qs]
report(f"  quantiles {qs} -> t = {[f'{t:.4f}' for t in ts]}")
alpha = np.polyfit(np.log(ts), np.log(qs), 1)[0]
report(f"  fitted tail exponent alpha = {alpha:.2f} "
       f"(theory: 1.5 = pair channel, 2.0 = triple channel)")

report("")
report("Done.")
out.close()
