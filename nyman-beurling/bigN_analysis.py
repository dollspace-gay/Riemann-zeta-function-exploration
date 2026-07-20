"""
Analysis of the out-of-sample run: matched-filter spectrum to N = 10^6.
=======================================================================
Rectified statistic  y(N) = drift-removed[ D~^2 logN * sqrt(N)/(4 log 2pi) ]:
under the derived law every zero is a CONSTANT-amplitude cosine at
frequency gamma_j with amplitude exactly 1/(|rho_j|^2 |zeta'(rho_j)|).
The gamma_4/gamma_5 amplitudes were stated in advance (predicted_lines.npy,
committed before the run). This is the pass/fail readout.
"""

import numpy as np
import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/rh_output")
L2PI = math.log(2 * math.pi)

g = np.load(os.path.join(OUT, "gpu10k_curves.npz"))
b = np.load(os.path.join(OUT, "bigN_curve.npz"))
# merge: gram-based below 10^4, sieve-based above (seam offset ~5e-7, small
# vs the faintest rectified line; noted in RESULTS)
m = g["Ngrid"] < 10000
Ngrid = np.concatenate([g["Ngrid"][m], b["Ngrid"]])
D2 = np.concatenate([g["D2"][m], b["D2"]])
logN = np.log(Ngrid.astype(float))

out = open(os.path.join(OUT, "bigN_analysis.txt"), "w")
def rep(s=""):
    print(s, flush=True); out.write(s + "\n"); out.flush()

# rectified curve, drift removed in rectified space
y_raw = D2 * logN * np.sqrt(Ngrid) / (4 * L2PI)
basis = np.stack([np.sqrt(Ngrid) * logN ** (-p) for p in range(0, 4)], axis=1)
beta, *_ = np.linalg.lstsq(basis, y_raw, rcond=None)
y = y_raw - basis @ beta

def amp_phase(x, r, g_):
    c, s = np.cos(g_ * x), np.sin(g_ * x)
    M = np.array([[c @ c, c @ s], [c @ s, s @ s]])
    v = np.array([c @ r, s @ r])
    a, bb = np.linalg.solve(M, v)
    return math.hypot(a, bb)

lines = np.load(os.path.join(OUT, "predicted_lines.npy"))  # gamma, |rho|^2, |zeta'|
gam_band = np.arange(5.0, 45.0, 0.01)
spec = np.array([amp_phase(logN, y, gg) for gg in gam_band])
floor = float(np.median(spec))

rep("=" * 74)
rep("OUT-OF-SAMPLE TEST: rectified matched-filter spectrum, N = 50 ... 10^6")
rep("=" * 74)
rep(f"grid points: {len(Ngrid)} (range factor {Ngrid[-1]/Ngrid[0]:.0f}, "
    f"resolution ~{2*math.pi/(logN[-1]-logN[0]):.2f})")
rep(f"noise floor (median rectified amplitude): {floor:.3e}")
rep("")
rep(f"{'zero':>10} {'PREDICTED':>11} {'MEASURED':>11} {'meas/pred':>9} {'SNR':>6}  verdict")
for j, (gam, r2, zd) in enumerate(lines):
    pred = 1.0 / (r2 * zd)
    meas = amp_phase(logN, y, gam)
    snr = meas / floor
    if j < 3:
        verdict = "in-sample (law was fit era)" if False else "in-sample check"
    else:
        verdict = ("DETECTED, matches" if snr > 3 and 0.5 < meas / pred < 2.0 else
                   "DETECTED, off-prediction" if snr > 3 else
                   "not detected")
    rep(f"  g{j+1}={gam:8.4f} {pred:>11.3e} {meas:>11.3e} {meas/pred:>9.2f} "
        f"{snr:>6.1f}  {verdict}")

rep("")
rep("top 10 spectrum peaks:")
idx = [i for i in range(2, len(spec) - 2)
       if spec[i] > spec[i - 1] and spec[i] > spec[i + 1]]
idx.sort(key=lambda i: -spec[i])
ZZ = list(lines[:, 0])
beats = sorted({round(abs(a - c), 3) for ii, a in enumerate(ZZ)
                for c in ZZ[:ii]})
for i in idx[:10]:
    f = gam_band[i]
    nearz = min(ZZ, key=lambda z: abs(z - f))
    nearb = min(beats, key=lambda z: abs(z - f)) if beats else 99
    tag = (f" <- gamma ({nearz:.3f})" if abs(nearz - f) < 0.3 else
           f" <- beat? ({nearb:.3f})" if abs(nearb - f) < 0.3 else "")
    rep(f"  {spec[i]:.3e} at {f:7.3f}{tag}")

# money plot
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(gam_band, spec, lw=0.8, label="measured rectified spectrum")
for j, (gam, r2, zd) in enumerate(lines):
    ax.plot([gam], [1.0 / (r2 * zd)], "v", color="crimson", ms=8,
            label="predicted amplitudes (stated in advance)" if j == 0 else None)
    ax.axvline(gam, color="crimson", ls=":", lw=0.7, alpha=0.5)
ax.axhline(floor, color="gray", ls="--", lw=0.8, label="noise floor")
ax.set_xlabel("frequency in log N")
ax.set_ylabel("rectified amplitude  =  1/(|rho|^2 |zeta'(rho)|)")
ax.set_title("out-of-sample test to N = 10^6: predicted (markers) vs measured spectrum")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "out_of_sample_10e6.png"), dpi=140)
rep(f"\nfigure: {os.path.join(OUT, 'out_of_sample_10e6.png')}")
out.close()
