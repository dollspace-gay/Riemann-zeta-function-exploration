"""
Session 3: the doubling-chain structure of the NB null space.
==============================================================
EXACT IDENTITY (elementary, proved in chains write-up):
    e_{mk}(t) = e_k(t)/m + (1/m) * (floor(1/(kt)) mod m),
so the "chain difference" f^m_k := e_{mk} - e_k/m equals the square-wave
(b mod m)/m with b = floor(1/(kt)), supported on (0, 1/k], with

    ||f^m_k||^2 = C_m / k,    C_m = (1/m^2) sum_{r=1}^{m-1} r^2 S_r(m),
    S_r(m) = sum_{j == r mod m} 1/(j(j+1)).
For m = 2: C_2 = (log 2)/4 (alternating series), giving the closed form
    A(1,2) = (3/4)(log 2pi - gamma) - (log 2)/4.

TESTS HERE:
  1. Verify the identity pointwise and the C_m norms against the Gram data.
  2. THE KEY TEST: restrict G to the chain subspace
     span{f^2_k : k in (N/2 - w, N/2]} and compare its minimal (generalized)
     Rayleigh quotient with the true lambda_min(G_N), across N and window
     width w. If they track, the N^-2 proof reduces to an explicit small
     matrix analysis.
  3. Mixed 2- and 3-chain subspace.
  4. Eigenvector dissection across N: chain ratio v_k/v_2k -> -1/2?,
     support width, PR scaling.
  5. Dyadic block-averaged zero hunt (queued from session 2).
"""

import numpy as np
import torch
import os, math
from nb_gram import EULER_GAMMA, LOG_2PI, BURNOL_C, DEVICE, DTYPE, b_vector

OUTPUT_DIR = os.path.expanduser("~/rh_output")
NMAX = 2500
A = np.load(os.path.join(OUTPUT_DIR, f"nb_gram_{NMAX}.npz"))["A"]

out = open(os.path.join(OUTPUT_DIR, "nb_chains.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

# ---------------------------------------------------------------------
# 1. Identity + norm checks
# ---------------------------------------------------------------------
report("=" * 74)
report("1. EXACT IDENTITY CHECKS")
report("=" * 74)

rng = np.random.default_rng(7)
ts = rng.uniform(1e-4, 2.0, 200000)
ok = True
for m, k in [(2, 7), (3, 11), (5, 4), (7, 30)]:
    def e(j, t): return np.modf(1.0 / (j * t))[0]
    lhs = e(m * k, ts)
    rhs = e(k, ts) / m + (np.floor(1.0 / (k * ts)) % m) / m
    err = np.max(np.abs(lhs - rhs))
    ok &= err < 1e-12
    report(f"  identity m={m}, k={k}: max err {err:.2e}")

def C_m(m, J=10**7):
    j = np.arange(1, J, dtype=np.float64)
    w = 1.0 / (j * (j + 1.0))
    tot = 0.0
    for r in range(1, m):
        tot += r * r * w[(j % m) == r].sum()
        tot += r * r * (1.0 / m) * (1.0 / J)   # tail estimate
    return tot / (m * m)

C2 = C_m(2); C3 = C_m(3)
report(f"  C_2 = {C2:.8f}   (log2)/4 = {math.log(2)/4:.8f}")
report(f"  C_3 = {C3:.8f}")
# Gram cross-checks: ||f^m_k||^2 = A[mk,mk] + A[k,k]/m^2 - 2 A[k,mk]/m
for m, k, Cm in [(2, 100, C2), (2, 1000, C2), (3, 100, C3), (3, 800, C3)]:
    i, j = k - 1, m * k - 1
    nrm = A[j, j] + A[i, i] / m**2 - 2.0 * A[i, j] / m
    report(f"  ||f^{m}_{k}||^2 * k = {k*nrm:.8f}  vs C_{m} = {Cm:.8f}  "
           f"diff {abs(k*nrm-Cm):.1e}")
A12_closed = 0.75 * (LOG_2PI - EULER_GAMMA) - math.log(2) / 4
report(f"  A(1,2) closed form {A12_closed:.9f}  vs computed {A[0,1]:.9f}")

# ---------------------------------------------------------------------
# 2. Chain-subspace restriction vs true lambda_min
# ---------------------------------------------------------------------
report("")
report("=" * 74)
report("2. CHAIN SUBSPACE vs FULL lambda_min")
report("=" * 74)

def chain_lambda(N, w, m=2):
    """min generalized Rayleigh quotient on span{f^m_k}, k in (N//m - w, N//m]."""
    kmax = N // m
    ks = np.arange(max(2, kmax - w + 1), kmax + 1)
    i = ks - 1; j = m * ks - 1
    Mjj = A[np.ix_(j, j)]
    Mji = A[np.ix_(j, i)]
    Mii = A[np.ix_(i, i)]
    M = Mjj - Mji / m - Mji.T / m + Mii / m**2
    S = 1.0 + 1.0 / m**2   # ||f-coefficient vector||^2 (disjoint supports)
    ev = np.linalg.eigvalsh(M)
    return ev[0] / S, len(ks)

report(f"{'N':>6} {'lmin(G_N)':>11} | " +
       " ".join(f"{'w='+str(w):>10}" for w in [5, 10, 20, 40, 80, 160]))
for N in [500, 1000, 2000, 2500]:
    sub = torch.tensor(A[:N, :N], dtype=DTYPE, device=DEVICE)
    lmin = float(torch.linalg.eigvalsh(sub)[0].item())
    vals = []
    for w in [5, 10, 20, 40, 80, 160]:
        lc, _ = chain_lambda(N, w)
        vals.append(lc)
    report(f"{N:>6} {lmin:>11.3e} | " +
           " ".join(f"{v:>10.3e}" for v in vals))
report("ratios (chain / full) at best w:")
for N in [500, 1000, 2000, 2500]:
    sub = torch.tensor(A[:N, :N], dtype=DTYPE, device=DEVICE)
    lmin = float(torch.linalg.eigvalsh(sub)[0].item())
    best = min(chain_lambda(N, w)[0] for w in [5, 10, 20, 40, 80, 160])
    report(f"  N={N}: best chain {best:.3e} / lmin {lmin:.3e} = {best/lmin:.2f}")

# mixed 2- and 3-chains
def mixed_lambda(N, w):
    k2 = np.arange(max(2, N//2 - w + 1), N//2 + 1)
    k3 = np.arange(max(2, N//3 - w + 1), N//3 + 1)
    cols = []
    X = np.zeros((N, len(k2) + len(k3)))
    for a, k in enumerate(k2):
        X[2*k - 1, a] = 1.0; X[k - 1, a] = -0.5
    for bidx, k in enumerate(k3):
        X[3*k - 1, len(k2) + bidx] = 1.0; X[k - 1, len(k2) + bidx] = -1.0/3.0
    M = X.T @ A[:N, :N] @ X
    S = X.T @ X
    ev = np.linalg.eigvalsh(np.linalg.solve(S, M))
    return ev[0]

report("")
report("mixed 2+3 chains (w=80):")
for N in [1000, 2000, 2500]:
    lm = mixed_lambda(N, 80)
    sub = torch.tensor(A[:N, :N], dtype=DTYPE, device=DEVICE)
    lmin = float(torch.linalg.eigvalsh(sub)[0].item())
    report(f"  N={N}: mixed chain {lm:.3e}  ratio to full {lm/lmin:.2f}")

# ---------------------------------------------------------------------
# 3. Eigenvector dissection across N
# ---------------------------------------------------------------------
report("")
report("=" * 74)
report("3. NULL EIGENVECTOR DISSECTION")
report("=" * 74)
report(f"{'N':>6} {'PR':>6} {'support width':>14} {'median v_k/v_2k':>16}")
for N in [500, 1000, 1500, 2000, 2500]:
    sub = torch.tensor(A[:N, :N], dtype=DTYPE, device=DEVICE)
    evals, evecs = torch.linalg.eigh(sub)
    v = evecs[:, 0].cpu().numpy()
    pr = 1.0 / np.sum(v**4)
    # support width: weighted std of k over the top half of mass near N
    p = v**2
    ksup = np.arange(1, N + 1)
    big = p > 0.01 * p.max()
    width = ksup[big].max() - ksup[big].min()
    # chain ratio: for k in top components with both k,2k <= N
    tops = np.argsort(p)[::-1][:20] + 1
    ratios = [v[k-1] / v[2*k-1] for k in tops if 2*k <= N and abs(v[2*k-1]) > 1e-3]
    med = np.median(ratios) if ratios else float('nan')
    report(f"{N:>6} {pr:>6.1f} {width:>14} {med:>16.3f}")

# ---------------------------------------------------------------------
# 4. Dyadic block-averaged zero hunt
# ---------------------------------------------------------------------
report("")
report("=" * 74)
report("4. BLOCK-AVERAGED ZERO HUNT")
report("=" * 74)
y = np.load(os.path.join(OUTPUT_DIR, "nb_periodogram.npz"))["y"]
# sliding geometric blocks [n, rho*n]; statistic should be
# ~ C(1/log n - 1/log(rho n)) + oscillation
rho = 1.3
ns = np.unique(np.round(np.geomspace(60, NMAX / rho, 400)).astype(int))
T, Lmid = [], []
for n0 in ns:
    n1 = int(round(rho * n0))
    blk = np.sum(y[n0:n1]**2)   # y[k-1] is index k; block (n0, n1]
    smooth = BURNOL_C * (1.0/math.log(n0) - 1.0/math.log(n1))
    T.append(blk - smooth)
    Lmid.append(0.5*(math.log(n0) + math.log(n1)))
T = np.array(T); Lmid = np.array(Lmid)
D = np.stack([np.ones_like(Lmid), 1/Lmid, 1/Lmid**2], axis=1)
beta, *_ = np.linalg.lstsq(D, T, rcond=None)
Td = T - D @ beta
report(f"block statistic rms after drift removal: {np.std(Td):.2e}")
Lt = torch.tensor(Lmid, dtype=DTYPE, device=DEVICE)
Rt = torch.tensor(Td, dtype=DTYPE, device=DEVICE)
gammas = torch.arange(2.0, 40.0, 0.01, dtype=DTYPE, device=DEVICE)
ph = gammas.unsqueeze(1) * Lt.unsqueeze(0)
cg, sg = torch.cos(ph), torch.sin(ph)
cc=(cg*cg).sum(1); ss=(sg*sg).sum(1); cs=(cg*sg).sum(1)
cr=(cg*Rt).sum(1); sr=(sg*Rt).sum(1)
det=cc*ss-cs*cs
aa=(ss*cr-cs*sr)/det; bb=(cc*sr-cs*cr)/det
amp=torch.sqrt(aa*aa+bb*bb).cpu().numpy()
g=gammas.cpu().numpy()
peaks=[]
for i in range(2, len(amp)-2):
    if amp[i]>amp[i-1] and amp[i]>amp[i+1]:
        peaks.append((amp[i], g[i]))
peaks.sort(reverse=True)
ZZ=[14.1347,21.0220,25.0109,30.4249,32.9351,37.5862]
report("top peaks:")
for av, gv in peaks[:8]:
    near = min(ZZ, key=lambda z: abs(z-gv))
    tag = f"  <-- gamma = {near}" if abs(near-gv) < 0.3 else ""
    report(f"  amp {av:.2e}  freq {gv:7.3f}{tag}")

report("")
report("Done.")
out.close()
