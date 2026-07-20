"""
Plan 3A: the residual profile r_N(t) — where the unapproximable mass lives.
==========================================================================
EXACT SCHEME (no quadrature error): in u = 1/t coordinates,

    r~(u) = chi - sum c_k e_k  becomes  1 + T(floor(u)) - u*S   on u >= 1,
    T(n) = sum_k c_k floor(n/k)  (divisor sieve),  S = sum_k c_k / k,

piecewise LINEAR in u with integer breakpoints, so every unit interval
integrates in closed form:
    int_n^{n+1} (a - S u)^2 u^-2 du = a^2(1/n - 1/(n+1))
        - 2 a S log((n+1)/n) + S^2,   a = 1 + T(n).
The t > 1 region (u < 1) contributes exactly S^2.

ENERGY CLOSURE GATE (non-negotiable): head-to-U + S^2 + tail estimate
must reproduce d_N^2 = 1 - b.c from the linear algebra. No closure -> no
analysis.

Outputs: exact per-decade mass profile, cumulative mass, median-mass
point u_1/2(N) and its N-scaling, S_N diagnostic, rescaled overlay.
"""

import numpy as np
import math, os
from scipy.linalg import cho_factor, cho_solve
from nb_gram_np import b_vector, mobius, BURNOL_C

OUT = os.path.expanduser("~/rh_output")
A = np.load(os.path.join(OUT, "nb_gram_2000_np.npz"))["A"]

out = open(os.path.join(OUT, "residual_profile.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n"); out.flush()

def residual_profile(N, U_mult=256):
    """Exact residual mass profile for optimal coefficients at N."""
    b = b_vector(N)
    cf = cho_factor(A[:N, :N], lower=True)
    c = cho_solve(cf, b)
    d2 = 1.0 - b @ c
    S = np.sum(c / np.arange(1, N + 1))
    U = U_mult * N
    # divisor sieve: inc[n] = sum_{k | n, k <= N} c_k
    inc = np.zeros(U + 1)
    for k in range(1, N + 1):
        inc[k::k] += c[k - 1]
    T = np.cumsum(inc)                      # T[n] = sum_k c_k floor(n/k)
    n = np.arange(1, U, dtype=np.float64)
    a = 1.0 + T[1:U]
    piece = (a * a * (1.0 / n - 1.0 / (n + 1.0))
             - 2.0 * a * S * np.log((n + 1.0) / n)
             + S * S * 1.0)
    head = np.sum(piece)
    # tail estimate beyond U: r~ = 1 - sum c_k {u/k}; measure last-decade
    # mass and extrapolate one decade ratio (reported, added to closure)
    dec_lo = np.searchsorted(n, U / 10.0)
    m_lastdec = np.sum(piece[dec_lo:])
    ratio = m_lastdec / max(np.sum(piece[np.searchsorted(n, U/100.0):dec_lo]), 1e-300)
    tail_est = m_lastdec * ratio / max(1.0 - ratio, 1e-9)
    total = S * S + head + tail_est
    return dict(c=c, d2=d2, S=S, U=U, n=n, piece=piece, head=head,
                tail_est=tail_est, total=total)

report("=" * 74)
report("RESIDUAL PROFILE — exact piecewise scheme, optimal coefficients")
report("=" * 74)

Ns = [100, 500, 1000, 2000]
profiles = {}
for N in Ns:
    p = residual_profile(N)
    profiles[N] = p
    closure = abs(p["total"] - p["d2"]) / p["d2"]
    report(f"\nN = {N}:")
    report(f"  d^2 (linear algebra)   = {p['d2']:.8e}")
    report(f"  S^2 + head + tail_est  = {p['total']:.8e}")
    report(f"  CLOSURE: rel diff {closure:.2e}  "
           f"(S^2 = {p['S']**2:.2e}, tail_est = {p['tail_est']:.2e})  "
           f"{'PASS' if closure < 0.01 else '*** FAIL ***'}")

report("")
report("=" * 74)
report("WHERE THE MASS LIVES (fractions of d^2, exact)")
report("=" * 74)
report(f"{'N':>6} {'t>1 (S^2)':>10} {'u in[1,4)':>10} {'[4,16)':>8} "
       f"{'[16,64)':>8} {'[64,256)':>9} {'[256,1k)':>9} {'[1k,4k)':>8} "
       f"{'beyond':>8} {'u_1/2':>7} {'u_1/2/N':>8}")
for N in Ns:
    p = profiles[N]
    n, piece, d2 = p["n"], p["piece"], p["d2"]
    fr = [p["S"]**2 / d2]
    for lo, hi in [(1, 4), (4, 16), (16, 64), (64, 256), (256, 1000),
                   (1000, 4000)]:
        i0, i1 = np.searchsorted(n, lo), np.searchsorted(n, hi)
        fr.append(np.sum(piece[i0:i1]) / d2)
    fr.append(1.0 - sum(fr))
    cum = p["S"]**2 + np.cumsum(piece)
    ih = int(np.searchsorted(cum, 0.5 * d2))
    u_half = n[min(ih, len(n) - 1)]
    report(f"{N:>6} " + " ".join(f"{f:>{wd}.4f}" for f, wd in
           zip(fr, [10, 10, 8, 8, 9, 9, 8, 8])) +
           f" {u_half:>7.0f} {u_half/N:>8.3f}")

report("")
report("=" * 74)
report("S_N DIAGNOSTIC:  S_N = sum c_k/k  vs  -sum_{k<=N} mu(k)/k")
report("=" * 74)
mu = mobius(2000)
for N in Ns:
    musum = np.sum(mu[:N] / np.arange(1.0, N + 1))
    report(f"  N={N:>5}:  S_N = {profiles[N]['S']:>12.6f}   "
           f"-sum mu/k = {-musum:>12.6f}   d^2 = {profiles[N]['d2']:.3e}")

# ---------------------------------------------------------------------
# figure: per-decade normalized profile, raw u and rescaled u/N
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for N in Ns:
    p = profiles[N]
    n, piece, d2 = p["n"], p["piece"], p["d2"]
    nbins = np.geomspace(1, p["U"], 60)
    idx = np.searchsorted(n, nbins)
    mass = np.add.reduceat(piece, np.clip(idx[:-1], 0, len(piece) - 1)) / d2
    centers = np.sqrt(nbins[:-1] * nbins[1:])
    wlog = np.log(nbins[1:] / nbins[:-1])
    axes[0].plot(centers, mass / wlog, label=f"N={N}")
    axes[1].plot(centers / N, mass / wlog, label=f"N={N}")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("u = 1/t"); axes[0].set_ylabel("d mass / d log u  (/ d^2)")
axes[0].set_title("residual mass density")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("u / N"); axes[1].set_title("rescaled by N")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "residual_profile.png"), dpi=130)
report(f"\nfigure saved: {os.path.join(OUT, 'residual_profile.png')}")
report("Done.")
out.close()
