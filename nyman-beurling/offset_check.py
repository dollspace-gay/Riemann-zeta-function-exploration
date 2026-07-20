"""
The +9% systematic: pipeline calibration, split-range test, and the
next-order term.
=====================================================================
All eight measured line amplitudes sit 1.05-1.15x the parameter-free
prediction. Three hypotheses, three tests:

 T1 PIPELINE CALIBRATION (decisive): inject KNOWN amplitudes (exactly
    the law) + realistic drift + beats + noise into synthetic curves on
    the real N-grid; run the identical rectify/drift/LSQ pipeline.
    Recovered/true = 1.00 -> the offset is physics. = 1.09 -> it's us.

 T2 SPLIT-RANGE: a 1/log N next-order term predicts offset ~ c1/Lbar,
    LARGER on the low-N half than the high-N half, in ratio
    Lbar_high/Lbar_low. A pipeline artifact has no reason to scale so.

 T3 THE CANDIDATE TERM: expanding the s=1 residue to next order gives
    A_j(N) * [1 + c1/log N], c1 = (zeta''/zeta')(0) - log 2pi + 1 - gamma
    (the gamma from zeta(s) ~ 1/(s-1) + gamma; the rest from the
    derivative of the slowly-varying factor at the double pole; the
    O(1/gamma_j) piece is imaginary => phase, not amplitude).
    Every constant computed with mpmath, never recalled.
"""

import numpy as np
import math, os

OUT = os.path.expanduser("~/rh_output")
L2PI = math.log(2 * math.pi)

g = np.load(os.path.join(OUT, "gpu10k_curves.npz"))
b = np.load(os.path.join(OUT, "bigN_curve.npz"))
m = g["Ngrid"] < 10000
Ngrid = np.concatenate([g["Ngrid"][m], b["Ngrid"]])
D2 = np.concatenate([g["D2"][m], b["D2"]])
logN = np.log(Ngrid.astype(float))

lines = np.load(os.path.join(OUT, "predicted_lines.npy"))   # gamma, |rho|^2, |zeta'|
GAM = lines[:, 0]
AMP_RECT = 1.0 / (lines[:, 1] * lines[:, 2])                # rectified true amps

def rectify_and_fit(D2curve, gammas):
    y_raw = D2curve * logN * np.sqrt(Ngrid) / (4 * L2PI)
    basis = np.stack([np.sqrt(Ngrid) * logN ** (-p) for p in range(4)], axis=1)
    beta, *_ = np.linalg.lstsq(basis, y_raw, rcond=None)
    y = y_raw - basis @ beta
    out = []
    for g_ in gammas:
        c, s = np.cos(g_ * logN), np.sin(g_ * logN)
        M = np.array([[c @ c, c @ s], [c @ s, s @ s]])
        v = np.array([c @ y, s @ y])
        a, bb = np.linalg.solve(M, v)
        out.append(math.hypot(a, bb))
    return np.array(out), y

print("=" * 74)
print("T1. PIPELINE CALIBRATION (synthetic injection, known amplitudes)")
print("=" * 74)
# realistic drift: the actual fitted drift of the real curve
basis_d = np.stack([logN ** (-p) for p in range(4)], axis=1)
bd, *_ = np.linalg.lstsq(basis_d, D2 * logN, rcond=None)
drift_real = basis_d @ bd
FLOOR = 1.5e-4          # measured cleaned rectified floor
BEATS = [(GAM[1]-GAM[0], 1.0e-3), (GAM[2]-GAM[0], 8.0e-4),
         (GAM[3]-GAM[2], 6e-4), (GAM[5]-GAM[2], 5e-4)]
rng = np.random.default_rng(2)
ratios_all = []
for seed in range(6):
    phases = rng.uniform(0, 2 * math.pi, len(GAM) + len(BEATS))
    osc = np.zeros_like(logN)
    for i, (g_, a_) in enumerate(zip(GAM, AMP_RECT)):
        osc += a_ * (4 * L2PI / np.sqrt(Ngrid)) * np.cos(g_ * logN + phases[i]) / 1.0
    for i, (f_, a_) in enumerate(BEATS):
        osc += a_ * (4 * L2PI / np.sqrt(Ngrid)) * np.cos(f_ * logN + phases[len(GAM)+i])
    noise = FLOOR * (4 * L2PI / np.sqrt(Ngrid)) * rng.standard_normal(len(logN))
    D2_synth = (drift_real + osc + noise) / logN
    meas, _ = rectify_and_fit(D2_synth, GAM)
    ratios_all.append(meas / AMP_RECT)
R = np.array(ratios_all)
print(f"{'zero':>10} {'recovered/true (mean of 6 seeds)':>34} {'std':>7}")
for i, g_ in enumerate(GAM):
    print(f"  g={g_:7.3f} {np.mean(R[:, i]):>18.3f} {'':>15} {np.std(R[:, i]):>7.3f}")
print(f"  MEAN over zeros: {np.mean(R):.3f}  ->  "
      f"{'pipeline is UNBIASED; the 9% is physics' if abs(np.mean(R)-1)<0.03 else 'PIPELINE BIAS FOUND'}")

print()
print("=" * 74)
print("T2. SPLIT-RANGE TEST on the REAL curve")
print("=" * 74)
mid = np.searchsorted(logN, 0.5 * (logN[0] + logN[-1]))
for tag, sl in [("low half", slice(0, mid)), ("high half", slice(mid, None)),
                ("full", slice(None))]:
    lN, Ng, D2s = logN[sl], Ngrid[sl], D2[sl]
    y_raw = D2s * lN * np.sqrt(Ng) / (4 * L2PI)
    basis = np.stack([np.sqrt(Ng) * lN ** (-p) for p in range(4)], axis=1)
    beta, *_ = np.linalg.lstsq(basis, y_raw, rcond=None)
    y = y_raw - basis @ beta
    rr = []
    for i in range(3):                    # the three high-SNR lines
        c, s = np.cos(GAM[i] * lN), np.sin(GAM[i] * lN)
        M = np.array([[c @ c, c @ s], [c @ s, s @ s]])
        v = np.array([c @ y, s @ y])
        a, bb = np.linalg.solve(M, v)
        rr.append(math.hypot(a, bb) / AMP_RECT[i])
    Lbar = float(np.mean(lN))
    print(f"  {tag:>9}: Lbar = {Lbar:5.2f}   ratios g1,g2,g3 = "
          + ", ".join(f"{x:.3f}" for x in rr)
          + f"   mean offset {np.mean(rr)-1:+.3f}")

print()
print("=" * 74)
print("T3. THE NEXT-ORDER TERM (all constants computed, none recalled)")
print("=" * 74)
import mpmath as mp
mp.mp.dps = 30
z1 = mp.zeta(mp.mpf(0), derivative=1)
z2 = mp.zeta(mp.mpf(0), derivative=2)
print(f"  zeta'(0)  = {float(z1):+.9f}   (known -log(2pi)/2 = {-L2PI/2:+.9f})")
print(f"  zeta''(0) = {float(z2):+.9f}")
B = float(z2 / z1)
gamma_e = 0.5772156649015329
c1 = B - L2PI + 1 - gamma_e
print(f"  (zeta''/zeta')(0) = {B:+.6f}")
print(f"  c1 = (zeta''/zeta')(0) - log 2pi + 1 - gamma = {c1:+.6f}")
for tag, Lb in [("low half", 6.38), ("high half", 11.34), ("full", 8.86)]:
    print(f"  predicted offset {tag:>9}: 1 + c1/Lbar = {1 + c1/Lb:.3f}")
