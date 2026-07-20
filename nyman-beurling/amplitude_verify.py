"""
Plan 2, Experiment 1 verification: is the gamma_1 = 14.13 line real?
=====================================================================
Four hard checks on the trial-curve detection before it is believed:
  V1. Drift-basis robustness: powers (0..2), (0..3), (0..4).
  V2. Range robustness: N in [50,700], [300,2000], [700,2000].
  V3. Hann taper in log N (kills rectangular-window sidelobes; the
      16.5/11.7 peaks should die if they are gamma_1 leakage).
  V4. Phase stability: fit amp*cos(gamma_1 logN + phi) on two disjoint
      halves; a real line has the same phi in both (mod pi/8-ish).
Also: amplitude at gamma_1..gamma_3 vs two candidate laws
      1/|rho|^2 and 1/|rho zeta'(rho)|^2 (ratios only, no fit).
"""

import numpy as np
import math, os

OUT = os.path.expanduser("~/rh_output")
dat = np.load(os.path.join(OUT, "amplitude_exp1.npz"))
Ngrid, D2 = dat["Ngrid"], dat["D2"]
logN = np.log(Ngrid.astype(float))
curve = D2 * logN

out = open(os.path.join(OUT, "amplitude_verify.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n"); out.flush()

G1, G2, G3 = 14.134725, 21.022040, 25.010858

def amp_at(x, r, g, wts=None):
    w = np.ones_like(x) if wts is None else wts
    cg, sg = np.cos(g * x) * w, np.sin(g * x) * w
    rw = r * w
    M = np.array([[cg @ cg, cg @ sg], [cg @ sg, sg @ sg]])
    v = np.array([cg @ rw, sg @ rw])
    a, b = np.linalg.solve(M, v)
    return math.hypot(a, b), math.atan2(-b, a)

def drift_removed(x, y, pmax):
    D = np.stack([x ** 0.0 / x ** p for p in range(pmax + 1)], axis=1)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return y - D @ beta

def floor_of(x, r, wts=None):
    amps = [amp_at(x, r, g, wts)[0] for g in np.arange(4.0, 40.0, 0.37)]
    return float(np.median(amps))

report("V1. DRIFT-BASIS ROBUSTNESS (full range)")
for pmax in [2, 3, 4]:
    r = drift_removed(logN, curve, pmax)
    a1 = amp_at(logN, r, G1)[0]
    fl = floor_of(logN, r)
    report(f"  powers 0..{pmax}:  amp(g1) = {a1:.3e}   floor = {fl:.3e}   "
           f"SNR = {a1/fl:.1f}")

report("")
report("V2. RANGE ROBUSTNESS (drift powers 0..3)")
for lo, hi in [(50, 700), (300, 2000), (700, 2000), (50, 2000)]:
    m = (Ngrid >= lo) & (Ngrid <= hi)
    r = drift_removed(logN[m], curve[m], 3)
    a1 = amp_at(logN[m], r, G1)[0]
    fl = floor_of(logN[m], r)
    report(f"  N in [{lo:>4},{hi:>4}]:  amp(g1) = {a1:.3e}   "
           f"floor = {fl:.3e}   SNR = {a1/fl:.1f}")

report("")
report("V3. HANN TAPER (sidelobe test; drift powers 0..3, full range)")
r = drift_removed(logN, curve, 3)
w = 0.5 - 0.5 * np.cos(2 * math.pi * (logN - logN[0]) / (logN[-1] - logN[0]))
for g, tag in [(G1, "g1"), (16.55, "16.55 sidelobe?"), (11.74, "11.74 sidelobe?"),
               (G2, "g2"), (G3, "g3")]:
    at, _ = amp_at(logN, r, g, w)
    an, _ = amp_at(logN, r, g)
    report(f"  freq {g:7.3f} ({tag:>15}):  untapered {an:.3e}  "
           f"tapered {at:.3e}  ratio {at/max(an,1e-300):.2f}")

report("")
report("V4. PHASE STABILITY AT gamma_1 (two disjoint halves)")
half = len(Ngrid) // 2
for (sl, tag) in [(slice(0, half), "first half"), (slice(half, None), "second half")]:
    r = drift_removed(logN[sl], curve[sl], 2)
    a, ph = amp_at(logN[sl], r, G1)
    report(f"  {tag:>12}: amp {a:.3e}  phase {ph:+.3f} rad")

report("")
report("AMPLITUDE RATIOS vs CANDIDATE LAWS (full range, powers 0..3)")
r = drift_removed(logN, curve, 3)
a1 = amp_at(logN, r, G1)[0]; a2 = amp_at(logN, r, G2)[0]; a3 = amp_at(logN, r, G3)[0]
report(f"  measured      1 : {a2/a1:.3f} : {a3/a1:.3f}")
rho2 = [0.25 + g * g for g in (G1, G2, G3)]
report(f"  1/|rho|^2     1 : {rho2[0]/rho2[1]:.3f} : {rho2[0]/rho2[2]:.3f}")
report("  (1/|rho zeta'(rho)|^2 comparison deferred: |zeta'(rho_j)| values")
report("   must be computed, not recalled — flagged for the Step-2 derivation.)")
report("Done.")
out.close()
