"""
Session 2: the full d_N curve to N = 2500, and hunting zeta zeros in it.
=========================================================================
1. One Cholesky of G_{Nmax} gives the ENTIRE curve: with G = LL^T and
   y = L^{-1} b, the leading-submatrix property gives
       d_N^2 = 1 - sum_{k<=N} y_k^2   for every N at once.
   (y_k is the component of chi along the k-th Gram-Schmidt direction —
   itself a forensic object: y_k^2 = d_{k-1}^2 - d_k^2.)
2. Residual R(N) = d_N^2 log N - C should carry oscillations from the
   zeta zeros (frequencies gamma_j in log N). Periodogram after removing
   smooth drift; do the peaks land on gamma_1 = 14.13, gamma_2 = 21.02,
   gamma_3 = 25.01...?
3. lambda_min scaling to N = 2500 and chain structure of the null space.
4. Deviation field delta_k = c_k + mu(k)(1 - log k / log N).
"""

import torch
import numpy as np
import os, math
from nb_gram import (build_gram, b_vector, mobius, BURNOL_C,
                     DEVICE, DTYPE)

OUTPUT_DIR = os.path.expanduser("~/rh_output")
NMAX = 2500

out = open(os.path.join(OUTPUT_DIR, "nb_oscillations.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

cache = os.path.join(OUTPUT_DIR, f"nb_gram_{NMAX}.npz")
if os.path.exists(cache):
    A = np.load(cache)["A"]
    report(f"[loaded cached Gram N={NMAX}]")
else:
    report(f"Building Gram to N = {NMAX}...")
    A = build_gram(NMAX)
    np.savez_compressed(cache, A=A)
b = b_vector(NMAX)

# ---------------------------------------------------------------------
# 1. Full d_N curve from one Cholesky
# ---------------------------------------------------------------------
At = torch.tensor(A, dtype=DTYPE, device=DEVICE)
bt = torch.tensor(b, dtype=DTYPE, device=DEVICE)
Lc = torch.linalg.cholesky(At)
y = torch.linalg.solve_triangular(Lc, bt.unsqueeze(1), upper=False).squeeze(1)
y_np = y.cpu().numpy()
d2 = 1.0 - np.cumsum(y_np**2)
Ns = np.arange(1, NMAX + 1)

report("")
report(f"{'N':>6} {'d_N^2':>12} {'d^2 logN':>10}")
for Nv in [50, 100, 200, 500, 1000, 1500, 2000, 2500]:
    report(f"{Nv:>6} {d2[Nv-1]:>12.8f} {d2[Nv-1]*math.log(Nv):>10.6f}")
report(f"Burnol constant C = {BURNOL_C:.7f}")

# Gram-Schmidt energy forensics: which k contribute?
mu = mobius(NMAX)
sf = mu != 0
report("")
report(f"Gram-Schmidt energies y_k^2 (k = 2..13): "
       f"{['%.2e' % v for v in (y_np[1:13]**2)]}")
report(f"mean y_k^2, squarefree k in [100,2500]:     "
       f"{np.mean(y_np[99:][sf[99:]]**2):.3e}")
report(f"mean y_k^2, non-squarefree k in [100,2500]: "
       f"{np.mean(y_np[99:][~sf[99:]]**2):.3e}")

# ---------------------------------------------------------------------
# 2. Periodogram of R(N) = d_N^2 log N - C
# ---------------------------------------------------------------------
N0 = 50
LN = np.log(Ns[N0-1:])
R = d2[N0-1:] * LN - BURNOL_C

# remove smooth drift in 1/logN powers
D = np.stack([np.ones_like(LN), 1.0/LN, 1.0/LN**2, 1.0/LN**3], axis=1)
beta, *_ = np.linalg.lstsq(D, R, rcond=None)
Rd = R - D @ beta
report("")
report(f"drift fit beta (1, 1/L, 1/L^2, 1/L^3): {np.round(beta, 5)}")
report(f"residual after drift removal: rms = {np.std(Rd):.3e}")

# vectorized LSQ periodogram over frequency grid
Lt = torch.tensor(LN, dtype=DTYPE, device=DEVICE)
Rt = torch.tensor(Rd, dtype=DTYPE, device=DEVICE)
gammas = torch.arange(2.0, 40.0, 0.01, dtype=DTYPE, device=DEVICE)
ph = gammas.unsqueeze(1) * Lt.unsqueeze(0)          # [F, M]
cg, sg = torch.cos(ph), torch.sin(ph)
# per-frequency 2x2 normal equations
cc = (cg*cg).sum(1); ss = (sg*sg).sum(1); cs = (cg*sg).sum(1)
cr = (cg*Rt).sum(1); sr = (sg*Rt).sum(1)
det = cc*ss - cs*cs
a = (ss*cr - cs*sr) / det
bb = (cc*sr - cs*cr) / det
amp = torch.sqrt(a*a + bb*bb).cpu().numpy()
gam_np = gammas.cpu().numpy()

# top peaks (local maxima)
peaks = []
for i in range(2, len(amp)-2):
    if amp[i] > amp[i-1] and amp[i] > amp[i+1]:
        peaks.append((amp[i], gam_np[i]))
peaks.sort(reverse=True)
report("")
report("Top periodogram peaks (amplitude, frequency):")
ZETA_ZEROS = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862]
for ampv, g in peaks[:10]:
    near = min(ZETA_ZEROS, key=lambda z: abs(z - g))
    tag = f"  <-- gamma_{ZETA_ZEROS.index(near)+1} = {near}" \
          if abs(near - g) < 0.3 else ""
    report(f"  amp {ampv:.3e}  freq {g:7.3f}{tag}")
report(f"First zeta zeros: {ZETA_ZEROS}")
np.savez(os.path.join(OUTPUT_DIR, "nb_periodogram.npz"),
         gam=gam_np, amp=amp, d2=d2, y=y_np)

# ---------------------------------------------------------------------
# 3. lambda_min scaling and chain structure at scale
# ---------------------------------------------------------------------
report("")
report("lambda_min scaling:")
for Nv in [500, 1000, 1500, 2000, 2500]:
    sub = torch.tensor(A[:Nv, :Nv], dtype=DTYPE, device=DEVICE)
    ev = torch.linalg.eigvalsh(sub)
    report(f"  N={Nv:>5}: lmin = {float(ev[0]):.3e}   lmin*N^2 = {float(ev[0])*Nv*Nv:.3f}"
           f"   kappa = {float(ev[-1]/ev[0]):.2e}")

sub = torch.tensor(A, dtype=DTYPE, device=DEVICE)
evals, evecs = torch.linalg.eigh(sub)
vmin = evecs[:, 0].cpu().numpy()
top = np.argsort(np.abs(vmin))[::-1][:14]
report("")
report(f"null direction at N={NMAX} (PR = {1.0/np.sum(vmin**4):.1f}):")
for kk in top:
    k1 = kk + 1
    rel = ""
    for kj in top:
        if kj + 1 == 2*k1: rel = f"  [2k = {2*k1} also in top]"
    report(f"  k={k1:>5}  v={vmin[kk]:+.4f}{rel}")

# ---------------------------------------------------------------------
# 4. deviation from the Mobius profile
# ---------------------------------------------------------------------
c = torch.cholesky_solve(bt.unsqueeze(1), Lc, upper=False).squeeze(1).cpu().numpy()
pred = -mu * (1.0 - np.log(Ns) / math.log(NMAX))
delta = c - pred
report("")
report(f"c_k vs -mu(k)(1-logk/logN): corr = {np.corrcoef(c, pred)[0,1]:.4f}")
report(f"deviation delta_k: rms {np.std(delta):.4f}; "
       f"rms squarefree {np.std(delta[sf]):.4f}, non-sf {np.std(delta[~sf]):.4f}")
report(f"largest |delta_k|: k = {np.argsort(np.abs(delta))[::-1][:10] + 1}")

report("")
report("Done.")
out.close()
