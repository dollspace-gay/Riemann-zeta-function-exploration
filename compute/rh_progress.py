"""
RH Crystal: Progress Experiments
=================================
New experiments targeting the open problems in papers/rh_crystal_v3_final.md.

Test A (Open Problems 2+3): The interlacing pair bound.
    By Cauchy interlacing, lambda_min(G) <= smallest eigenvalue of any 2x2
    principal submatrix. For an adjacent pair with gap delta, that eigenvalue
    is exactly   B(delta) = sum_p w_hat(p) * (1 - cos(delta * log p))
    (w_hat = w / S0 after trace normalization), and B(delta) ~ (S2/2S0) delta^2
    for small delta. If the bound is TIGHT, then lambda_min is governed by the
    minimal zero gap, and kappa ~ N^0.61 reduces to extreme-gap statistics
    (Ben Arous-Bourgade: smallest unfolded GUE gap among N ~ N^{-1/3},
    predicting lambda_min ~ N^{-2/3}).

Test B (control ensembles): Same Gram construction on synthetic spectra with
    the SAME smooth density as the zeta zeros (mapped through the inverse of
    the smooth counting function N(T)) but different local statistics:
      - GUE   (unfolded eigenvalues of a random Hermitian matrix)
      - Poisson (i.i.d. exponential gaps)
      - Picket fence (perfectly rigid lattice)
    Distinguishes zeta-specific structure from generic point-process behavior.

Test C (Open Problem 1): The curvature operator
    D_{ij} = (1/2) sum_p w(p,1/2) (log p)^2 cos((gamma_i-gamma_j) log p).
    D is PSD by the same Gram factorization as G (weights w*(log p)^2/2 >= 0).
    The second-order condition for sigma=1/2 to be a local min of kappa is
    r(N) = <vmax, D vmax>/lmax - <vmin, D vmin>/lmin > 0.
    The paper verified this only at N=30. Track r(N) up to N=2000 and
    cross-check the curvature prediction against a direct sigma sweep.

Test D (Open Problem 4): Height-adaptive prime cutoff. Fixed blocks of 1000
    zeros at increasing heights, with prime cutoffs 1e5 / 1e6 / 1e7. Does the
    kappa-vs-height curve flatten as the cutoff grows?

Requires ~/rh_data/zeros6 (run compute/download_zeros.sh first).
"""

import torch
import numpy as np
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
DATA_DIR = os.path.expanduser("~/rh_data")
OUTPUT_DIR = os.path.expanduser("~/rh_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =====================================================================
# DATA
# =====================================================================
print("\nLoading zeros...")
zeros_list = []
with open(os.path.join(DATA_DIR, "zeros6"), "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            val = float(line.split()[-1])
            if val > 0:
                zeros_list.append(val)
        except ValueError:
            continue
zeros_np = np.array(zeros_list, dtype=np.float64)
print(f"Loaded {len(zeros_np)} zeros, range [{zeros_np[0]:.3f}, {zeros_np[-1]:.1f}]")

print("Sieving primes...")
def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

MAX_PRIME = 10_000_000
all_primes = sieve_primes(MAX_PRIME).astype(np.float64)
log_primes_np = np.log(all_primes)
print(f"{len(all_primes)} primes up to {MAX_PRIME}")

# =====================================================================
# CORE CONSTRUCTION (matches rh_gpu.py exactly; optional extra_logp_sq
# turns G into the curvature operator D)
# =====================================================================
def build_gram(gammas, n_primes, sigma=0.5, extra_logp_sq=False,
               normalize=True, batch_size=2000):
    N = len(gammas)
    P = min(n_primes, len(all_primes))
    gamma = torch.tensor(gammas, dtype=DTYPE, device=DEVICE)
    G = torch.zeros((N, N), dtype=DTYPE, device=DEVICE)
    S0 = 0.0
    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch_p = torch.tensor(all_primes[start:end], dtype=DTYPE, device=DEVICE)
        batch_logp = torch.tensor(log_primes_np[start:end], dtype=DTYPE, device=DEVICE)
        w = batch_logp * torch.cosh((sigma - 0.5) * batch_logp) / torch.sqrt(batch_p)
        if extra_logp_sq:
            w = w * batch_logp**2 / 2.0
        S0 += w.sum().item()
        phase_args = -gamma.unsqueeze(1) * batch_logp.unsqueeze(0)
        sqrt_w = torch.sqrt(w).unsqueeze(0)
        wcos = torch.cos(phase_args) * sqrt_w
        wsin = torch.sin(phase_args) * sqrt_w
        G += wcos @ wcos.T + wsin @ wsin.T
    if normalize:
        tr = torch.trace(G)
        if tr > 1e-15:
            G *= N / tr
    return G, S0

def pair_bound(gammas, n_primes, batch_size=20000):
    """Exact 2x2 interlacing bound for every adjacent pair:
       B_i = sum_p w_hat(p) * (1 - cos((g_{i+1}-g_i) log p)),  w_hat = w/S0.
       Returns the array of bounds (in trace-normalized units)."""
    P = min(n_primes, len(all_primes))
    gaps = torch.tensor(np.diff(gammas), dtype=DTYPE, device=DEVICE)  # [N-1]
    acc = torch.zeros_like(gaps)
    S0 = 0.0
    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch_p = torch.tensor(all_primes[start:end], dtype=DTYPE, device=DEVICE)
        batch_logp = torch.tensor(log_primes_np[start:end], dtype=DTYPE, device=DEVICE)
        w = batch_logp / torch.sqrt(batch_p)  # sigma = 1/2
        S0 += w.sum().item()
        acc += ((1.0 - torch.cos(gaps.unsqueeze(1) * batch_logp.unsqueeze(0))) * w).sum(dim=1)
    return (acc / S0).cpu().numpy()

def spectrum(G, want_vecs=False):
    if want_vecs:
        evals, evecs = torch.linalg.eigh(G)
        return evals.cpu().numpy(), evecs
    return torch.linalg.eigvalsh(G).cpu().numpy()

def fit_exponent(xs, ys):
    return np.polyfit(np.log(xs), np.log(ys), 1)[0]

results = open(os.path.join(OUTPUT_DIR, "progress_results.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    results.write(msg + "\n")
    results.flush()

# =====================================================================
# TEST A: interlacing pair bound vs actual lambda_min
# =====================================================================
report("=" * 78)
report("TEST A: Interlacing pair bound  lambda_min <= B(delta_min)")
report("=" * 78)
report(f"{'N':>6} {'P':>7} {'lmin':>12} {'minB':>12} {'lmin/minB':>10} "
       f"{'d_min':>8} {'argmin pair':>12} {'vec wt':>8} {'same?':>6}")

A_N, A_lmin, A_minB, A_kappa, A_dmin = [], [], [], [], []
for N in [250, 500, 1000, 2000, 4000, 8000]:
    P = min(20 * N, len(all_primes))
    g = zeros_np[:N]
    t0 = time.time()
    G, _ = build_gram(g, P)
    evals, evecs = spectrum(G, want_vecs=True)
    lmin, lmax = evals[0], evals[-1]
    B = pair_bound(g, P)
    iB = int(np.argmin(B))
    minB = float(B[iB])
    # weight of the lambda_min eigenvector on the argmin pair
    v = evecs[:, 0].cpu().numpy()
    wt = v[iB]**2 + v[iB+1]**2
    # where does the eigenvector actually localize?
    iv = int(np.argmax(v**2))
    same = abs(iv - iB) <= 1
    dmin = g[iB+1] - g[iB]
    A_N.append(N); A_lmin.append(lmin); A_minB.append(minB)
    A_kappa.append(lmax/lmin); A_dmin.append(dmin)
    report(f"{N:>6} {P:>7} {lmin:>12.6f} {minB:>12.6f} {lmin/minB:>10.3f} "
           f"{dmin:>8.4f} {f'({iB},{iB+1})':>12} {wt:>8.3f} {str(same):>6}"
           f"   [{time.time()-t0:.0f}s]")

report("")
report(f"Exponent fits:  lambda_min ~ N^{fit_exponent(A_N, A_lmin):+.3f}   "
       f"minB ~ N^{fit_exponent(A_N, A_minB):+.3f}   "
       f"kappa ~ N^{fit_exponent(A_N, A_kappa):+.3f}   "
       f"delta_min ~ N^{fit_exponent(A_N, A_dmin):+.3f}")
report("GUE extreme-gap prediction: unfolded delta_min ~ N^(-1/3) "
       "=> B ~ delta^2 => lambda_min ~ N^(-2/3)")

# =====================================================================
# TEST B: control ensembles with matched smooth density
# =====================================================================
report("")
report("=" * 78)
report("TEST B: zeta vs GUE vs Poisson vs picket fence (same smooth density)")
report("=" * 78)

def smooth_count(T):
    return T / (2*np.pi) * np.log(T / (2*np.pi*np.e)) + 7.0/8.0

def inv_smooth_count(u, T_init):
    """Newton inversion of the smooth counting function (vectorized)."""
    T = np.full_like(u, T_init, dtype=np.float64)
    for _ in range(60):
        f = smooth_count(T) - u
        df = np.log(T / (2*np.pi)) / (2*np.pi)
        T = np.clip(T - f / df, 10.0, None)
    return T

def gue_unfolded(N, rng):
    """Central N unfolded eigenvalues of an M x M GUE matrix, unit mean gap."""
    M = 2 * N
    A = torch.tensor(rng.standard_normal((M, M)), dtype=DTYPE, device=DEVICE) \
        + 1j * torch.tensor(rng.standard_normal((M, M)), dtype=DTYPE, device=DEVICE)
    H = (A + A.conj().T) / 2.0
    ev = torch.linalg.eigvalsh(H).cpu().numpy() / np.sqrt(M)
    x = np.clip(ev, -2.0, 2.0)
    F = 0.5 + (x * np.sqrt(4.0 - x**2) + 4.0 * np.arcsin(x / 2.0)) / (4.0 * np.pi)
    u = M * F
    lo = (M - N) // 2
    u = u[lo:lo+N]
    return (u - u[0]) / np.mean(np.diff(u))  # unit mean spacing, start at 0

def synth_zeros(kind, N, rng):
    if kind == "gue":
        u = gue_unfolded(N, rng)
    elif kind == "poisson":
        u = np.concatenate([[0.0], np.cumsum(rng.exponential(1.0, N - 1))])
    elif kind == "picket":
        u = np.arange(N, dtype=np.float64)
    u = u + smooth_count(zeros_np[0])
    return inv_smooth_count(u, zeros_np[0])

rng = np.random.default_rng(42)
report(f"{'N':>6} | {'zeta k':>10} {'gue k':>10} {'poisson k':>12} {'picket k':>10} "
       f"| {'zeta lmin':>10} {'gue lmin':>10} {'poiss lmin':>10} {'pick lmin':>10}")

B_data = {k: {"N": [], "k": [], "l": []} for k in ["zeta", "gue", "poisson", "picket"]}
for N in [500, 1000, 2000, 4000]:
    P = min(20 * N, len(all_primes))
    row_k, row_l = {}, {}
    for kind in ["zeta", "gue", "poisson", "picket"]:
        g = zeros_np[:N] if kind == "zeta" else synth_zeros(kind, N, rng)
        G, _ = build_gram(g, P)
        evals = spectrum(G)
        lmin, lmax = evals[0], evals[-1]
        row_k[kind], row_l[kind] = lmax / lmin, lmin
        B_data[kind]["N"].append(N)
        B_data[kind]["k"].append(lmax / lmin)
        B_data[kind]["l"].append(lmin)
    report(f"{N:>6} | {row_k['zeta']:>10.2f} {row_k['gue']:>10.2f} "
           f"{row_k['poisson']:>12.2f} {row_k['picket']:>10.2f} "
           f"| {row_l['zeta']:>10.6f} {row_l['gue']:>10.6f} "
           f"{row_l['poisson']:>10.6f} {row_l['picket']:>10.6f}")

report("")
for kind in ["zeta", "gue", "poisson", "picket"]:
    d = B_data[kind]
    report(f"  {kind:>8}: kappa ~ N^{fit_exponent(d['N'], d['k']):+.3f}, "
           f"lambda_min ~ N^{fit_exponent(d['N'], d['l']):+.3f}")

# =====================================================================
# TEST C: curvature operator D, second-order minimum across N
# =====================================================================
report("")
report("=" * 78)
report("TEST C: curvature operator D (PSD by construction); r(N) > 0 <=> local min")
report("=" * 78)
report(f"{'N':>6} {'Rmax':>10} {'Rmin':>10} {'r=Rmax-Rmin':>12} {'pred k ratio':>12} {'actual':>10}")

EPS = 0.02
for N in [100, 500, 1000, 2000]:
    P = min(20 * N, len(all_primes))
    g = zeros_np[:N]
    G, S0 = build_gram(g, P, normalize=False)
    D, _ = build_gram(g, P, extra_logp_sq=True, normalize=False)
    evals, evecs = spectrum(G, want_vecs=True)
    vmin, vmax = evecs[:, 0], evecs[:, -1]
    lmin, lmax = evals[0], evals[-1]
    Rmax = (vmax @ (D @ vmax)).item() / lmax
    Rmin = (vmin @ (D @ vmin)).item() / lmin
    r = Rmax - Rmin
    pred = 1.0 + EPS**2 * r          # kappa(1/2+eps)/kappa(1/2) to 2nd order
    Ge, _ = build_gram(g, P, sigma=0.5 + EPS, normalize=False)
    ee = spectrum(Ge)
    actual = (ee[-1] / ee[0]) / (lmax / lmin)
    report(f"{N:>6} {Rmax:>10.2f} {Rmin:>10.2f} {r:>12.2f} {pred:>12.5f} {actual:>10.5f}")

# =====================================================================
# TEST D: height-adaptive prime cutoff
# =====================================================================
report("")
report("=" * 78)
report("TEST D: blocks of 1000 zeros at height H, prime cutoff scan")
report("=" * 78)
cutoffs = [9592, 78498, 664579]          # pi(1e5), pi(1e6), pi(1e7)
report(f"{'block start':>12} {'height':>10} | {'k (P<=1e5)':>12} {'k (P<=1e6)':>12} {'k (P<=1e7)':>12}")
for start in [0, 100_000, 500_000, 1_000_000]:
    g = zeros_np[start:start+1000]
    ks = []
    for P in cutoffs:
        G, _ = build_gram(g, P)
        ev = spectrum(G)
        ks.append(ev[-1] / ev[0])
    report(f"{start:>12} {g[0]:>10.0f} | {ks[0]:>12.1f} {ks[1]:>12.1f} {ks[2]:>12.1f}")

report("")
report("Done.")
results.close()
print(f"\nResults written to {os.path.join(OUTPUT_DIR, 'progress_results.txt')}")
