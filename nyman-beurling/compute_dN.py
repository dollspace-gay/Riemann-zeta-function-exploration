"""
Nyman-Beurling / Baez-Duarte distance d_N and Gram forensics.
==============================================================
See README.md in this folder for the criterion, derivations, and references.

Computes A_{mn} = <e_m, e_n>_{L2(0,inf)}, e_k(t) = rho(1/(kt)), via:
  A_{mn} = (1/g) * Acop(m/g, n/g),  g = gcd
  Acop(m,n) = 1/(mn) + HEAD + TAIL   (m,n coprime, L = mn)
    HEAD = exact piecewise integral of rho(u/m)rho(u/n)/u^2 over [1, 1+L]
    TAIL = int_0^L rho((1+w)/m)rho((1+w)/n) * psi1((1+L+w)/L)/L^2 dw,
           Gauss-Legendre per breakpoint segment (smooth integrand).
Cross terms: b_k = (log k + 1 - gamma)/k  (exact).
Nyman normalization: G01 = A - 1/(mn) elementwise (exact rank structure).

Self-tests: A(1,1) = log(2pi) - gamma; diagonal reduction; brute-force
midpoint integration for (1,2), (2,3), (2,4); d_1 anchor.
"""

import torch
import numpy as np
import os, time, math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
OUTPUT_DIR = os.path.expanduser("~/rh_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EULER_GAMMA = 0.5772156649015328606
LOG_2PI = math.log(2.0 * math.pi)
BURNOL_C = 2.0 + EULER_GAMMA - math.log(4.0 * math.pi)  # 0.0461914...

out = open(os.path.join(OUTPUT_DIR, "nb_results.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

# ---------------------------------------------------------------------
# trigamma via Euler-Maclaurin (args >= 1 here); validated against series
# ---------------------------------------------------------------------
def psi1(x):
    """Trigamma for x >= 0.5, vectorized torch float64."""
    J = 16
    s = torch.zeros_like(x)
    for j in range(J):
        s = s + (x + j) ** -2
    y = x + J
    s = s + 1.0/y + 0.5/y**2 + 1.0/(6.0*y**3) - 1.0/(30.0*y**5)
    return s

# ---------------------------------------------------------------------
# Gauss-Legendre 8-point on [-1, 1]
# ---------------------------------------------------------------------
GL_X = torch.tensor([-0.9602898564975363, -0.7966664774136267,
                     -0.5255324099163290, -0.1834346424956498,
                      0.1834346424956498,  0.5255324099163290,
                      0.7966664774136267,  0.9602898564975363],
                    dtype=DTYPE, device=DEVICE)
GL_W = torch.tensor([0.1012285362903763, 0.2223810344533745,
                     0.3137066458778873, 0.3626837833783620,
                     0.3626837833783620, 0.3137066458778873,
                     0.2223810344533745, 0.1012285362903763],
                    dtype=DTYPE, device=DEVICE)

def acop_batch(ms, ns):
    """Acop(m,n) for arrays of coprime pairs (m <= n). Returns torch tensor."""
    P = len(ms)
    m = torch.tensor(ms, dtype=torch.int64, device=DEVICE)
    n = torch.tensor(ns, dtype=torch.int64, device=DEVICE)
    L = m * n
    # ragged breakpoints: multiples of m (j=1..n) and of n (j=1..m), plus 1, 1+L
    cnt = (n + m + 2)
    tot = int(cnt.sum().item())
    pair_of = torch.repeat_interleave(torch.arange(P, device=DEVICE), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(tot, device=DEVICE) - start[pair_of]
    mm, nn, LL = m[pair_of], n[pair_of], L[pair_of]
    # local 0 -> endpoint 1; local in [1, n] -> local*m; local in [n+1, n+m] -> (local-n)*n
    # local == n+m+1 -> endpoint 1+L
    v = torch.where(local == 0, torch.ones_like(LL),
        torch.where(local <= nn, local * mm,
        torch.where(local <= nn + mm, (local - nn) * nn, LL + 1)))
    # clamp stray multiples equal to 1 (m=1, j=1) -- they just duplicate endpoint
    # sort within pairs
    key = pair_of * (int(L.max().item()) + 3) + v
    order = torch.argsort(key)
    v_sorted = v[order].to(DTYPE)
    p_sorted = pair_of[order]
    # segments: consecutive entries within same pair
    same = p_sorted[1:] == p_sorted[:-1]
    v0 = v_sorted[:-1][same]
    v1 = v_sorted[1:][same]
    pidx = p_sorted[:-1][same]
    mm = m[pidx].to(DTYPE); nn = n[pidx].to(DTYPE); LL = L[pidx].to(DTYPE)
    mid = 0.5 * (v0 + v1)
    a = torch.floor(mid / mm)
    bb = torch.floor(mid / nn)
    nonzero = (v1 > v0)
    # HEAD: antiderivative u/(mn) - (b/m + a/n) log u - ab/u
    def H(u):
        return u / (mm * nn) - (bb / mm + a / nn) * torch.log(u) - a * bb / u
    head = torch.where(nonzero, H(v1) - H(v0), torch.zeros_like(v0))
    # TAIL: Gauss on w in [v0-1, v1-1] of f(1+w) psi1((1+L+w)/L)/L^2
    w0 = v0 - 1.0; w1 = v1 - 1.0
    half = 0.5 * (w1 - w0); cen = 0.5 * (w0 + w1)
    tail = torch.zeros_like(v0)
    for q in range(8):
        wq = cen + half * GL_X[q]
        u = 1.0 + wq
        f = (u / mm - a) * (u / nn - bb)
        tail = tail + GL_W[q] * half * f * psi1((1.0 + LL + wq) / LL) / LL**2
    tail = torch.where(nonzero, tail, torch.zeros_like(tail))
    contrib = head + tail
    A = torch.zeros(P, dtype=DTYPE, device=DEVICE)
    A.index_add_(0, pidx, contrib)
    A = A + 1.0 / (m.to(DTYPE) * n.to(DTYPE))   # the [0,1) piece
    return A

def build_gram(N, chunk_cost=4_000_000):
    """Full A matrix (N x N) in L2(0,inf), plus the coprime table."""
    mg, ng = np.meshgrid(np.arange(1, N+1), np.arange(1, N+1), indexing="ij")
    gcds = np.gcd(mg, ng)
    # coprime pairs m <= n
    cm, cn = [], []
    for mv in range(1, N+1):
        ns = np.arange(mv, N+1)
        mask = np.gcd(mv, ns) == 1
        for nv in ns[mask]:
            cm.append(mv); cn.append(int(nv))
    cm = np.array(cm); cn = np.array(cn)
    cost = cm + cn
    orderc = np.argsort(cost)
    cm, cn, cost = cm[orderc], cn[orderc], cost[orderc]
    Acop_tab = np.zeros((N+1, N+1))
    i = 0
    t0 = time.time()
    while i < len(cm):
        j = i
        c = 0
        while j < len(cm) and c < chunk_cost:
            c += cost[j]; j += 1
        vals = acop_batch(cm[i:j], cn[i:j]).cpu().numpy()
        Acop_tab[cm[i:j], cn[i:j]] = vals
        i = j
    report(f"  [coprime Gram: {len(cm)} pairs in {time.time()-t0:.1f}s]")
    # assemble full A via A_{mn} = Acop(m/g, n/g)/g
    mp = (mg // gcds); np_ = (ng // gcds)
    lo = np.minimum(mp, np_); hi = np.maximum(mp, np_)
    A = Acop_tab[lo, hi] / gcds
    return A

# ---------------------------------------------------------------------
# SELF-TESTS
# ---------------------------------------------------------------------
report("=" * 74)
report("SELF-TESTS")
report("=" * 74)

# closed forms
A11_exact = LOG_2PI - EULER_GAMMA
A11_num = float(acop_batch(np.array([1]), np.array([1]))[0].item())
report(f"A(1,1): scheme {A11_num:.12f}  exact {A11_exact:.12f}  "
       f"diff {abs(A11_num - A11_exact):.2e}")

def brute_A(mv, nv, U=20000.0, du=1e-4):
    """Brute midpoint integration of int_0^inf rho(u/m)rho(u/n)/u^2 du."""
    total = 1.0 / (mv * nv)  # [0,1) exact
    grid = torch.arange(1.0 + du/2, U, du, dtype=DTYPE, device=DEVICE)
    f = (torch.frac(grid / mv)) * (torch.frac(grid / nv)) / grid**2
    total += float((f.sum() * du).item())
    # periodic mean tail: Fbar/U
    Lv = mv * nv // math.gcd(mv, nv)
    wgrid = torch.arange(du/2, Lv, du, dtype=DTYPE, device=DEVICE)
    Fbar = float((torch.frac(wgrid / mv) * torch.frac(wgrid / nv)).mean().item())
    total += Fbar / U
    return total

for (mv, nv) in [(1, 2), (2, 3), (2, 4)]:
    g = math.gcd(mv, nv)
    sch = float(acop_batch(np.array([mv//g]), np.array([nv//g]))[0].item()) / g
    bru = brute_A(mv, nv)
    report(f"A({mv},{nv}): scheme {sch:.9f}  brute {bru:.9f}  "
           f"diff {abs(sch-bru):.2e}")

b1 = 1.0 - EULER_GAMMA
d1sq = 1.0 - b1**2 / A11_exact
report(f"d_1^2 anchor: 1 - (1-g)^2/(log2pi-g) = {d1sq:.9f}")

# trigamma vs reference at a few points
ref = {1.0: 1.6449340668482264, 1.5: 0.9348022005446793, 2.0: 0.6449340668482264}
for xv, rv in ref.items():
    pv = float(psi1(torch.tensor([xv], dtype=DTYPE, device=DEVICE))[0].item())
    report(f"psi1({xv}) = {pv:.12f}  ref {rv:.12f}  diff {abs(pv-rv):.2e}")

# ---------------------------------------------------------------------
# MAIN: d_N curves and forensics
# ---------------------------------------------------------------------
NMAX = 500
report("")
report("=" * 74)
report(f"GRAM MATRIX to N = {NMAX}")
report("=" * 74)
A = build_gram(NMAX)
ks = np.arange(1, NMAX + 1)
b = (np.log(ks) + 1.0 - EULER_GAMMA) / ks
G01 = A - 1.0 / np.outer(ks, ks)

np.savez(os.path.join(OUTPUT_DIR, "nb_gram.npz"), A=A, b=b)

def dist_curve(G, b, Ns, label):
    report("")
    report(f"--- {label} ---")
    report(f"{'N':>5} {'d_N^2':>12} {'d^2 logN':>10} {'kappa(G)':>10} "
           f"{'lmin':>10} {'cut sens':>9}")
    rows = []
    for Nv in Ns:
        Gs = G[:Nv, :Nv]; bs = b[:Nv]
        evals, evecs = np.linalg.eigh(Gs)
        proj = evecs.T @ bs
        def dsq_cut(rel):
            keep = evals > rel * evals[-1]
            return 1.0 - float(np.sum(proj[keep]**2 / evals[keep]))
        d2 = dsq_cut(1e-14)
        sens = abs(dsq_cut(1e-12) - d2)
        kap = float(evals[-1] / max(evals[0], 1e-300))
        rows.append((Nv, d2))
        report(f"{Nv:>5} {d2:>12.8f} {d2*math.log(Nv):>10.6f} "
               f"{kap:>10.2e} {evals[0]:>10.2e} {sens:>9.1e}")
    return rows

Ns = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
rows_inf = dist_curve(A, b, Ns, "L2(0,inf)  [Baez-Duarte setting]")
rows_01 = dist_curve(G01, b, Ns, "L2(0,1)    [Nyman setting]")
report("")
report(f"Burnol/BDBLS constant C = 2 + gamma - log(4pi) = {BURNOL_C:.7f}")

# ---------------------------------------------------------------------
# FORENSICS at N = NMAX (L2(0,inf) setting)
# ---------------------------------------------------------------------
report("")
report("=" * 74)
report(f"FORENSICS at N = {NMAX}, L2(0,inf)")
report("=" * 74)

# Mobius function
mu = np.ones(NMAX + 1, dtype=np.int64)
primes_small = [p for p in range(2, NMAX + 1)
                if all(p % q for q in range(2, int(p**0.5) + 1))]
for p in primes_small:
    mu[p::p] *= -1
    mu[p*p::p*p] = 0
mu = mu[1:]

evals, evecs = np.linalg.eigh(A)
proj = evecs.T @ b
keep = evals > 1e-14 * evals[-1]
c = evecs[:, keep] @ (proj[keep] / evals[keep])

# From sum_{k<=x} mu(k) floor(x/k) = 1: chi(t) = (1/t)sum mu(k)/k - sum mu(k) e_k(t)
# for t <= 1, so the natural approximation has NEGATIVE Mobius coefficients.
pred = -mu * (1.0 - np.log(ks) / math.log(NMAX))
corr = np.corrcoef(c, pred)[0, 1]
report(f"optimal c_k vs Mobius prediction -mu(k)(1 - logk/logN): corr = {corr:.4f}")
report(f"{'k':>4} {'c_k':>10} {'mu pred':>9}")
for kk in range(12):
    report(f"{kk+1:>4} {c[kk]:>10.4f} {pred[kk]:>9.4f}")

# lambda_min eigenvector structure
vmin = evecs[:, 0]
top = np.argsort(np.abs(vmin))[::-1][:12]
report("")
report(f"lambda_min = {evals[0]:.3e}; top |components| of null direction:")
report(f"{'k':>5} {'v_k':>10} {'mu(k)':>6}")
for kk in top:
    report(f"{kk+1:>5} {vmin[kk]:>10.4f} {mu[kk]:>6}")
pr = 1.0 / np.sum(vmin**4)
report(f"participation ratio of null direction: {pr:.1f} / {NMAX}")

report("")
report("Done.")
out.close()
