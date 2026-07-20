"""
Plan 7: envelope classification (Part A) + inverse line intensities (Part B).
=============================================================================
A: synthetic control — plant one off-line pair (envelope offset delta)
   among RH-consistent lines; show the detector classifies it by
   envelope exponent, and chart the resolution limit.
B: compute |zeta'(rho_j)| directly (mpmath), screen candidate intensity
   laws against the measured A1:A2:A3, invert the best for |zeta'(rho_1)|.
"""

import numpy as np
import math, os

OUT = os.path.expanduser("~/rh_output")
out = open(os.path.join(OUT, "spectroscopy.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n"); out.flush()

dat = np.load(os.path.join(OUT, "amplitude_exp1.npz"))
Ngrid = dat["Ngrid"]
logN = np.log(Ngrid.astype(float))
rng = np.random.default_rng(42)

GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]

def amp_at(x, r, g):
    cg, sg = np.cos(g * x), np.sin(g * x)
    M = np.array([[cg @ cg, cg @ sg], [cg @ sg, sg @ sg]])
    v = np.array([cg @ r, sg @ r])
    a, b = np.linalg.solve(M, v)
    return math.hypot(a, b)

# ---------------------------------------------------------------------
# Part A: planted off-line line, envelope classification
# ---------------------------------------------------------------------
report("=" * 74)
report("PART A: ENVELOPE CLASSIFICATION (synthetic control)")
report("=" * 74)
# calibration to the real measurement: gamma_1 amplitude 3.9e-3 at
# mean-N ~ 190 falling to 1.2e-3 at mean-N ~ 1180  ->  q ~ 0.64
Q_RH = 0.64
N0 = 190.0
A0 = [3.9e-3, 1.2e-3, 0.9e-3, 0.5e-3, 0.4e-3]   # base line amplitudes
NOISE = 2.0e-4
G_PLANT, DELTA = 18.0, 0.10                      # off-line pair: beta = 0.5 + 0.10
PHI = rng.uniform(0, 2 * math.pi, 6)

def synth(delta_plant):
    r = np.zeros_like(logN)
    for j, (g, a0) in enumerate(zip(GAMMAS, A0)):
        env = a0 * (Ngrid / N0) ** (-Q_RH)
        r += env * np.cos(g * logN + PHI[j])
    env_p = A0[1] * (Ngrid / N0) ** (-Q_RH + delta_plant)
    r += env_p * np.cos(G_PLANT * logN + PHI[5])
    r += NOISE * rng.standard_normal(len(logN))
    return r

ALL_LINES = GAMMAS + [G_PLANT]

def _lsq_resid(r, expo):
    """Residual norm of the full-range linear fit with per-line
    envelope exponents expo[g]: model = sum_g (N/N0)^expo[g] *
    (a_g cos(g logN) + b_g sin(g logN))."""
    cols = []
    for g in ALL_LINES:
        env = (Ngrid / N0) ** expo[g]
        cols.append(env * np.cos(g * logN))
        cols.append(env * np.sin(g * logN))
    D = np.stack(cols, axis=1)
    beta, res, *_ = np.linalg.lstsq(D, r, rcond=None)
    return float(np.sum((r - D @ beta) ** 2))

def envelope_exponents_global(r, sweeps=3):
    """Alternating profile fit: each line's envelope exponent scanned
    on a grid holding the others fixed; full-range resolution."""
    expo = {g: -Q_RH for g in ALL_LINES}
    grid = np.arange(-1.3, 0.31, 0.02)
    for _ in range(sweeps):
        for gt in ALL_LINES:
            vals = []
            for etry in grid:
                e2 = dict(expo); e2[gt] = etry
                vals.append(_lsq_resid(r, e2))
            expo[gt] = float(grid[int(np.argmin(vals))])
    return expo

r = synth(DELTA)
report(f"planted: freq {G_PLANT}, envelope offset delta = +{DELTA}")
report(f"{'line':>10} {'true env exp':>13} {'fitted':>8}")
fitted = envelope_exponents_global(r)
for g, tag, true_q in ([(gv, f"g={gv:.2f}", -Q_RH) for gv in GAMMAS]
                       + [(G_PLANT, "PLANTED", -Q_RH + DELTA)]):
    report(f"{tag:>10} {true_q:>13.3f} {fitted[g]:>8.3f}")
real_es = [fitted[g] for g in GAMMAS]
sep = fitted[G_PLANT] - np.mean(real_es)
scatter = np.std(real_es)
report(f"\n  classification: planted line sits {sep:+.3f} above the RH cluster")
report(f"  RH-cluster scatter: {scatter:.3f}  ->  separation = "
       f"{sep/scatter:.1f} sigma  {'DETECTED' if sep > 3*scatter else 'not separable'}")

report("\nresolution limit: smallest delta separable at 3 sigma, this range/noise:")
for d in [0.10, 0.05, 0.03, 0.02, 0.01]:
    seps = []
    for trial in range(6):
        es = envelope_exponents_global(synth(d))
        cluster = [es[g] for g in GAMMAS]
        seps.append((es[G_PLANT] - np.mean(cluster)) / max(np.std(cluster), 1e-9))
    med = float(np.median(seps))
    report(f"  delta = {d:.2f}: median separation {med:5.1f} sigma "
           f"{'<- limit region' if 2 < med < 5 else ''}")

# ---------------------------------------------------------------------
# Part B: |zeta'(rho_j)| computed, intensity laws screened
# ---------------------------------------------------------------------
report("")
report("=" * 74)
report("PART B: INVERSE LINE INTENSITIES")
report("=" * 74)
import mpmath as mp
mp.mp.dps = 30
zp = []
for j in range(1, 4):
    rho = mp.zetazero(j)
    dz = mp.zeta(rho, derivative=1)
    zp.append(float(abs(dz)))
    report(f"  rho_{j}: gamma = {float(mp.im(rho)):.6f}   |zeta'(rho)| = {float(abs(dz)):.6f}")

A_meas = np.array([1.0, 0.309, 0.221])          # measured ratios (Session 8)
rho_abs = np.array([math.hypot(0.5, g) for g in GAMMAS[:3]])
zp = np.array(zp)
laws = {
    "1/|rho|^2":            1.0 / rho_abs**2,
    "1/(|rho||zeta'|)":     1.0 / (rho_abs * zp),
    "1/(|rho||zeta'|)^2":   1.0 / (rho_abs * zp)**2,
    "1/(|rho|^2 |zeta'|)":  1.0 / (rho_abs**2 * zp),
}
report(f"\n  measured ratios      1 : {A_meas[1]:.3f} : {A_meas[2]:.3f}")
best, bestname = None, None
for name, v in laws.items():
    r_ = v / v[0]
    err = float(np.sqrt(np.mean((np.log(r_[1:]) - np.log(A_meas[1:]))**2)))
    report(f"  {name:>20}  1 : {r_[1]:.3f} : {r_[2]:.3f}   log-rms {err:.3f}")
    if best is None or err < best:
        best, bestname = err, name
report(f"\n  best-screening law: {bestname} (log-rms {best:.3f})")
report("  [3 points vs 4 laws = screening only; Plan 2 Step 2 must derive it]")
# inversion demo with the best law: estimate |zeta'(rho_1)| from A2/A1
if "zeta'" in bestname:
    # solve for zp[0] given zp[1], zp[2] true and measured ratios, law form
    p = 2 if "^2" in bestname else 1
    rr = 2 if "|rho|^2" in bestname else 1
    est = zp[1] * (rho_abs[1]**rr / rho_abs[0]**rr * A_meas[1]) ** (1.0/p) \
        if p == 1 else zp[1] * math.sqrt(rho_abs[1]**2 / rho_abs[0]**2 * A_meas[1])
    report(f"  inversion demo: |zeta'(rho_1)| from measured A2/A1 under best law "
           f"= {est:.3f}   (true {zp[0]:.3f}, error {abs(est-zp[0])/zp[0]:.1%})")
report("\nfrequency side: gamma_1 measured 14.140 vs true 14.1347 (3-4 digits).")
report("Done.")
out.close()
