"""
Overnight job: NB Gram matrix to N = 10000, full d_N curve, zero hunt v3.
Estimated: ~2 h Gram build (cost ~ N^3) + minutes of linear algebra.
Outputs: ~/rh_output/nb_gram_10000.npz (A, ~800 MB), nb_overnight.txt.
"""

import numpy as np
import torch
import os, math, time
from nb_gram import build_gram, b_vector, BURNOL_C, DEVICE, DTYPE

OUTPUT_DIR = os.path.expanduser("~/rh_output")
NMAX = 10000

out = open(os.path.join(OUTPUT_DIR, "nb_overnight.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

t0 = time.time()
cache = os.path.join(OUTPUT_DIR, f"nb_gram_{NMAX}.npz")
if os.path.exists(cache):
    A = np.load(cache)["A"]
    report("[loaded cache]")
else:
    report(f"Building Gram to N = {NMAX} (expect ~2h)...")
    A = build_gram(NMAX, chunk_cost=8_000_000)
    np.savez_compressed(cache, A=A)
    report(f"[built + saved in {(time.time()-t0)/60:.1f} min]")
b = b_vector(NMAX)

At = torch.tensor(A, dtype=DTYPE, device=DEVICE)
bt = torch.tensor(b, dtype=DTYPE, device=DEVICE)
Lc = torch.linalg.cholesky(At)
y = torch.linalg.solve_triangular(Lc, bt.unsqueeze(1), upper=False).squeeze(1)
y_np = y.cpu().numpy()
d2 = 1.0 - np.cumsum(y_np**2)
np.savez(os.path.join(OUTPUT_DIR, "nb_curve_10000.npz"), d2=d2, y=y_np)

report("")
report(f"{'N':>6} {'d_N^2':>12} {'d^2 logN':>10}")
for Nv in [500, 1000, 2000, 4000, 6000, 8000, 10000]:
    report(f"{Nv:>6} {d2[Nv-1]:>12.8f} {d2[Nv-1]*math.log(Nv):>10.6f}")
report(f"C = {BURNOL_C:.7f}")

# lambda_min scaling + theorem check at 10k
report("")
report("lambda_min and Theorem 1 bound:")
for Nv in [2500, 5000, 10000]:
    sub = At[:Nv, :Nv]
    ev = torch.linalg.eigvalsh(sub)
    K = Nv // 2
    H = float(np.sum(1.0/np.arange(1, K)))
    bound = (H + 1.0) / (10.0 * K * (K - 1))
    report(f"  N={Nv:>6}: lmin = {float(ev[0]):.3e}  lmin*N^2 = {float(ev[0])*Nv*Nv:.3f}  "
           f"thm bound = {bound:.3e}  ratio = {bound/float(ev[0]):.2f}")

# zero hunt v3: block-averaged increments over the full range
rho = 1.3
ns = np.unique(np.round(np.geomspace(60, NMAX/rho, 800)).astype(int))
T, Lmid = [], []
for n0 in ns:
    n1 = int(round(rho * n0))
    blk = float(np.sum(y_np[n0:n1]**2))
    smooth = BURNOL_C * (1.0/math.log(n0) - 1.0/math.log(n1))
    T.append(blk - smooth)
    Lmid.append(0.5*(math.log(n0)+math.log(n1)))
T = np.array(T); Lmid = np.array(Lmid)
D = np.stack([np.ones_like(Lmid), 1/Lmid, 1/Lmid**2], axis=1)
beta, *_ = np.linalg.lstsq(D, T, rcond=None)
Td = T - D @ beta
report("")
report(f"zero hunt v3: block rms after drift removal: {np.std(Td):.2e}")
Lt = torch.tensor(Lmid, dtype=DTYPE, device=DEVICE)
Rt = torch.tensor(Td, dtype=DTYPE, device=DEVICE)
gammas = torch.arange(2.0, 40.0, 0.005, dtype=DTYPE, device=DEVICE)
ph = gammas.unsqueeze(1) * Lt.unsqueeze(0)
cg, sg = torch.cos(ph), torch.sin(ph)
cc=(cg*cg).sum(1); ss=(sg*sg).sum(1); cs=(cg*sg).sum(1)
cr=(cg*Rt).sum(1); sr=(sg*Rt).sum(1)
det=cc*ss-cs*cs
aa=(ss*cr-cs*sr)/det; bb2=(cc*sr-cs*cr)/det
amp=torch.sqrt(aa*aa+bb2*bb2).cpu().numpy()
g=gammas.cpu().numpy()
peaks=[]
for i in range(2, len(amp)-2):
    if amp[i]>amp[i-1] and amp[i]>amp[i+1]:
        peaks.append((amp[i], g[i]))
peaks.sort(reverse=True)
ZZ=[14.1347,21.0220,25.0109,30.4249,32.9351,37.5862]
report("top peaks:")
for av, gv in peaks[:10]:
    near = min(ZZ, key=lambda z: abs(z-gv))
    tag = f"  <-- gamma = {near}" if abs(near-gv) < 0.25 else ""
    report(f"  amp {av:.2e}  freq {gv:7.3f}{tag}")

report("")
report(f"Total time: {(time.time()-t0)/60:.1f} min. Done.")
out.close()
