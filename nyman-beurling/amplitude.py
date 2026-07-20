"""
Plan 2, Experiment 1: the Mobius trial-function distance and its
periodogram.
=====================================================================
The optimal d_N^2 fluctuation is dominated by arithmetic noise (the
divisor structure of each new N) -- three hunts failed on that floor.
The trial function

    c~_k(N) = -mu(k) * (1 - log k / log N),  k <= N

is SMOOTH in the coefficients: new dilations enter with zero weight at
k = N, and no optimizer chases the arithmetic of each N. Its distance

    D~_N^2 = 1 - 2 c~.b + c~.A.c~   >=  d_N^2

is an exact quadratic form in the validated Gram data. If the zeta
zeros imprint oscillations cos(gamma_j log N) on the approach to the
Burnol constant, this curve is where they should be cleanest.

Outputs: trial curve vs optimal curve, drift-removed residuals,
least-squares periodograms (uneven sampling in log N), peak tables with
zero tags, noise floors.
"""

import numpy as np
import math, os
from nb_gram_np import b_vector, mobius, BURNOL_C, EULER_GAMMA

OUT = os.path.expanduser("~/rh_output")
NMAX = 2000
A = np.load(os.path.join(OUT, "nb_gram_2000_np.npz"))["A"]

out = open(os.path.join(OUT, "amplitude_exp1.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n"); out.flush()

ZZ = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178]

b = b_vector(NMAX)
mu = mobius(NMAX)

# ---------------------------------------------------------------------
# optimal curve via single Cholesky (leading-submatrix property)
# ---------------------------------------------------------------------
from scipy.linalg import cholesky, solve_triangular
Lc = cholesky(A, lower=True)
y = solve_triangular(Lc, b, lower=True)
d2_opt = 1.0 - np.cumsum(y * y)          # d2_opt[N-1] = d_N^2

# ---------------------------------------------------------------------
# trial curve on a log-uniform N grid
# ---------------------------------------------------------------------
Ngrid = np.unique(np.round(np.geomspace(50, NMAX, 360)).astype(int))
D2 = np.empty(len(Ngrid))
ks = np.arange(1, NMAX + 1)
logk = np.log(ks)
for i, N in enumerate(Ngrid):
    c = -mu[:N] * (1.0 - logk[:N] / math.log(N))
    D2[i] = 1.0 - 2.0 * c @ b[:N] + c @ (A[:N, :N] @ c)

logN = np.log(Ngrid.astype(float))
report("=" * 74)
report("TRIAL vs OPTIMAL: d^2 * log N")
report("=" * 74)
report(f"  Burnol C = {BURNOL_C:.7f}")
report(f"{'N':>6} {'D~^2 logN (trial)':>18} {'d^2 logN (optimal)':>19} {'ratio D~/d':>11}")
for N in [50, 100, 200, 500, 1000, 1500, 2000]:
    i = int(np.argmin(np.abs(Ngrid - N)))
    report(f"{Ngrid[i]:>6} {D2[i]*logN[i]:>18.5f} {d2_opt[Ngrid[i]-1]*logN[i]:>19.5f} "
           f"{D2[i]/d2_opt[Ngrid[i]-1]:>11.4f}")

# ---------------------------------------------------------------------
# drift removal + LSQ periodogram (uneven sampling in log N)
# ---------------------------------------------------------------------
def periodogram(x_log, r, gmin=2.0, gmax=40.0, dg=0.01):
    gam = np.arange(gmin, gmax, dg)
    amp = np.empty(len(gam))
    for i, g in enumerate(gam):
        cg = np.cos(g * x_log); sg = np.sin(g * x_log)
        M = np.array([[cg @ cg, cg @ sg], [cg @ sg, sg @ sg]])
        v = np.array([cg @ r, sg @ r])
        try:
            ab = np.linalg.solve(M, v)
            amp[i] = math.hypot(*ab)
        except np.linalg.LinAlgError:
            amp[i] = 0.0
    return gam, amp

def peaks_table(gam, amp, label, topn=8):
    report(f"\n  top peaks — {label}:")
    idx = [i for i in range(2, len(amp) - 2)
           if amp[i] > amp[i-1] and amp[i] > amp[i+1]]
    idx.sort(key=lambda i: -amp[i])
    for i in idx[:topn]:
        near = min(ZZ, key=lambda z: abs(z - gam[i]))
        tag = f"  <-- gamma = {near:.3f}" if abs(near - gam[i]) < 0.3 else ""
        report(f"    amp {amp[i]:.3e}  freq {gam[i]:7.3f}{tag}")
    report(f"    median amplitude (noise floor): {np.median(amp):.3e}")
    # amplitude AT the first three zero frequencies specifically
    for z in ZZ[:3]:
        i = int(np.argmin(np.abs(gam - z)))
        report(f"    amp at gamma = {z:7.3f}: {amp[i]:.3e}")

def analyze(curve, x_log, label, basis_powers=(0, 1, 2, 3)):
    D = np.stack([x_log ** 0.0 / x_log ** p for p in basis_powers], axis=1)
    beta, *_ = np.linalg.lstsq(D, curve, rcond=None)
    resid = curve - D @ beta
    report(f"\n  {label}: rms after drift removal {np.std(resid):.3e}")
    gam, amp = periodogram(x_log, resid)
    peaks_table(gam, amp, label)
    return resid

report("")
report("=" * 74)
report("PERIODOGRAMS (drift basis: 1, 1/logN, 1/log^2 N, 1/log^3 N)")
report("=" * 74)

r_trial = analyze(D2 * logN, logN, "TRIAL curve  D~^2 logN")

# optimal-curve baseline on the same grid (June-negative reproduction)
d2g = d2_opt[Ngrid - 1]
r_opt = analyze(d2g * logN, logN, "OPTIMAL curve  d^2 logN (baseline)")

# the excess (trial - optimal): pure second-order structure
r_exc = analyze((D2 - d2g) * logN, logN, "EXCESS  (D~^2 - d^2) logN")

np.savez_compressed(os.path.join(OUT, "amplitude_exp1.npz"),
                    Ngrid=Ngrid, D2=D2, d2_opt=d2_opt,
                    r_trial=r_trial, r_opt=r_opt, r_exc=r_exc)
report("\nSaved amplitude_exp1.npz. Done.")
out.close()
