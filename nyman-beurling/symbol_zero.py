#!/usr/bin/env python3
"""symbol_zero.py -- the symbol-zero test: is the gcd-floor governed by the zeta zeros?

Exact elementary identity (gcd = sum of phi over common divisors):

    x^T G x = sum_{d<=N} phi(d) | sum_{n<=N, d|n} x_n / sqrt(n) |^2        (*)

with G_{mn} = (m,n)/sqrt(mn).  For x_n = W(n) n^{-it} the class-d inner sum is
d^{-1/2-it} times a W-smoothed partial sum of zeta(1/2+it) over k <= N/d, so ALL
divisor classes vanish simultaneously exactly when zeta(1/2+it) = 0: the symbol
of G at the critical exponent is zeta on the critical line.

Tests
  T1  bottom-eigenvector periodograms at N = 1000/2000/4000: peak -> gamma_1?
  T2  Mobius control periodogram (mu itself resonates at zeros -- quantify)
  T3  symbol curve S(t) at N = 2000 (Hann^2 amplitude taper): dips at gamma_j;
      overlay |zeta(1/2+it)|^2
  T4  log-taper constructor via (*) at N up to 10^6 (no matrix needed):
      S(gamma_1) * log^2 N  vs the measured floor constant 0.836
Everything printed is measured; identity (*) is proved (elementary).
"""
import numpy as np, math, time, os
import mpmath as mp

ZEROS = [14.134725142, 21.022039639, 25.010857580, 30.424876126,
         32.935061588, 37.586178159, 40.918719012]
G1 = ZEROS[0]

def phi_sieve(N):
    phi = np.arange(N + 1, dtype=np.int64)
    for p in range(2, N + 1):
        if phi[p] == p:                      # p prime (untouched so far)
            phi[p::p] -= phi[p::p] // p
    return phi

def mobius_sieve(N):
    mu = np.ones(N + 1, dtype=np.int64)
    pr = np.ones(N + 1, dtype=bool); pr[:2] = False
    for p in range(2, int(N ** 0.5) + 1):
        if pr[p]:
            pr[p * p::p] = False
    for p in np.nonzero(pr)[0]:
        mu[p::p] *= -1
        mu[p * p::p * p] = 0
    return mu

def gcd_form(x, phi):
    """x^T G x via (*): x is complex, index n = 1..N at x[n]; x[0] ignored."""
    N = len(x) - 1
    y = x.astype(np.complex128).copy()
    y[1:] /= np.sqrt(np.arange(1, N + 1, dtype=np.float64))
    tot = 0.0
    for d in range(1, N + 1):
        s = y[d::d].sum()
        tot += phi[d] * (s.real * s.real + s.imag * s.imag)
    return tot

def amp(x, r, g):
    """least-squares cos/sin amplitude of residual r at frequency g over abscissa x"""
    c, s = np.cos(g * x), np.sin(g * x)
    M = np.array([[c @ c, c @ s], [c @ s, s @ s]])
    v = np.array([c @ r, s @ r])
    a, b = np.linalg.solve(M, v)
    return math.hypot(a, b)

def peak(x, r, lo=2.0, hi=40.0):
    band = np.arange(lo, hi, 0.05)
    spec = np.array([amp(x, r, g) for g in band])
    i = int(np.argmax(spec))
    fine = np.arange(band[i] - 0.5, band[i] + 0.5, 0.002)
    fspec = np.array([amp(x, r, g) for g in fine])
    j = int(np.argmax(fspec))
    if 0 < j < len(fine) - 1:                # parabolic refine
        y0, y1, y2 = fspec[j - 1], fspec[j], fspec[j + 1]
        j_off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        f = fine[j] + j_off * 0.002
    else:
        f = fine[j]
    return f, fspec[j], fspec[j] / float(np.median(spec))

def detrended(y, logn):
    return y - np.polyval(np.polyfit(logn, y, 3), logn)

t0 = time.time()
print("=" * 72)
zp1 = mp.zeta(mp.mpc(0.5, G1), derivative=1)
print(f"|zeta'(rho_1)| = {float(abs(zp1)):.6f}   |zeta'(rho_1)|^2 = {float(abs(zp1))**2:.6f}")

# ---------------------------------------------------------------- T1 + T2
print("\nT1: bottom-eigenvector peak location vs N (prediction: -> gamma_1 = 14.1347)")
lam_series = {}
for N in (1000, 2000, 4000):
    h = np.arange(1, N + 1)
    G = np.gcd.outer(h, h).astype(np.float64) / np.sqrt(np.outer(h, h))
    ev, vec = np.linalg.eigh(G)
    logn = np.log(h.astype(float))
    lam_series[N] = ev[0]
    r = detrended(vec[:, 0] * np.sqrt(h), logn)
    f, a, snr = peak(logn, r)
    print(f"  N={N:5d}  lam_min={ev[0]:.6f}  lam*log^2N={ev[0]*math.log(N)**2:.4f}"
          f"  peak f={f:8.4f}  (gamma_1{f-G1:+.4f})  SNR={snr:.1f}")
    if N == 2000:
        G2000, vec2000, ev2000, logn2000 = G, vec, ev, logn
    if N == 4000:
        # next three modes at the largest N
        for m in (1, 2, 3):
            rm = detrended(vec[:, m] * np.sqrt(h), logn)
            fm, am_, sm = peak(logn, rm)
            print(f"          mode {m}: lam={ev[m]:.6f}  peak f={fm:8.4f}  SNR={sm:.1f}")

print("\nT2: Mobius control (mu(n), same pipeline, N=4000) -- how much of T1 is mu?")
N = 4000; h = np.arange(1, N + 1); logn = np.log(h.astype(float))
mu = mobius_sieve(N)[1:]
r = detrended(mu.astype(float), logn)
f, a, snr = peak(logn, r)
print(f"  mu control: peak f={f:8.4f}  SNR={snr:.1f}"
      f"   amp@gamma_1={amp(logn, r, G1):.3f}  amp@17.5={amp(logn, r, 17.5):.3f}")

# ---------------------------------------------------------------- T3
print("\nT3: symbol curve S(t), N=2000, Hann^2 taper (prediction: dips at every gamma_j)")
N = 2000; h = np.arange(1, N + 1); logn = logn2000
W = np.cos(math.pi * h / (2 * N)) ** 4
ts = np.arange(1.0, 42.0, 0.1)
S = np.empty(len(ts))
for i, t in enumerate(ts):
    x = W * np.exp(-1j * t * logn)
    S[i] = (np.conj(x) @ (G2000 @ x)).real / (np.conj(x) @ x).real
zz2 = np.array([float(abs(mp.zeta(mp.mpc(0.5, t))) ** 2) for t in ts])
mask = ts > 8
A = float(np.median(S[mask] / np.maximum(zz2[mask], 1e-9)))
corr = float(np.corrcoef(np.log(S[mask]), np.log(zz2[mask] + 1e-12))[0, 1])
print(f"  scale A = {A:.3f}: S(t) ~= A*|zeta(1/2+it)|^2;  corr(log S, log|zeta|^2) [t>8] = {corr:.3f}")
dips = [i for i in range(1, len(ts) - 1) if S[i] < S[i - 1] and S[i] < S[i + 1]]
print("  S(t) local minima: " + ", ".join(f"{ts[i]:.1f}" for i in dips))
print("  zeta zeros:        " + ", ".join(f"{z:.1f}" for z in ZEROS))
for z in ZEROS:
    iz = int(np.argmin(np.abs(ts - z)))
    ib = int(np.argmin(np.abs(ts - (z + 3.0))))
    print(f"    S({z:6.2f}) = {S[iz]:9.4f}   vs S({ts[ib]:5.1f}) = {S[ib]:9.4f}"
          f"   contrast x{S[ib]/S[iz]:.1f}")

# ---------------------------------------------------------------- T4
print("\nT4: log-taper constructor via identity (*) -- upper bound, matrix-free")
print("    x_n = Omega(log n/log N) * n^(-i t);  S*log^2 N vs measured c = 0.836")
# validate (*) against the matrix first
rng = np.random.default_rng(1)
xv = np.zeros(2001, dtype=np.complex128)
xv[1:] = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
phi2000 = phi_sieve(2000)
q_form = gcd_form(xv, phi2000)
q_mat = (np.conj(xv[1:]) @ (G2000 @ xv[1:])).real
print(f"  identity check N=2000: |form-matrix|/|matrix| = {abs(q_form-q_mat)/abs(q_mat):.2e}")

results = {}
for N in (2000, 10**4, 10**5, 10**6):
    tN = time.time()
    phi = phi_sieve(N)
    mu = mobius_sieve(N)
    n = np.arange(0, N + 1, dtype=np.float64); n[0] = 1.0
    u = np.log(n) / math.log(N)
    profs = {
        "sin ":  np.sin(math.pi * u),
        "lin ":  1.0 - u,
        "sinmu": np.sin(math.pi * u) * mu,
    }
    L2 = math.log(N) ** 2
    row = {}
    for name, Om in profs.items():
        x = Om * np.exp(-1j * G1 * np.log(n)); x[0] = 0
        Sg = gcd_form(x, phi) / float(np.sum(np.abs(x) ** 2))
        row[name] = Sg * L2
    # sine profile: minimize over t near gamma_1, plus off-zero control
    tgrid = np.arange(G1 - 0.3, G1 + 0.31, 0.05)
    best, tbest = np.inf, None
    Om = profs["sin "]
    nrm = float(np.sum(np.abs(Om[1:]) ** 2))
    for t in tgrid:
        x = Om * np.exp(-1j * t * np.log(n)); x[0] = 0
        v = gcd_form(x, phi) / nrm
        if v < best: best, tbest = v, t
    x = Om * np.exp(-1j * 17.5 * np.log(n)); x[0] = 0
    ctrl = gcd_form(x, phi) / nrm
    results[N] = (row, best * L2, tbest, ctrl * L2)
    print(f"  N=10^{math.log10(N):.1f}: " +
          "  ".join(f"{k}: {v:7.3f}" for k, v in row.items()) +
          f"  | min_t {best*L2:7.3f} at t={tbest:.2f}  | t=17.5 ctrl {ctrl*L2:9.1f}"
          f"   [{time.time()-tN:.0f}s]")
print(f"  measured floor: lam_min*log^2N = 0.836 (N=2000), "
      f"{lam_series[4000]*math.log(4000)**2:.4f} (N=4000)")

np.savez(os.path.expanduser("~/rh_output/symbol_zero.npz"),
         ts=ts, S=S, zz2=zz2,
         t4_N=np.array(list(results.keys()), dtype=float),
         t4_sin=np.array([results[N][0]["sin "] for N in results]),
         t4_lin=np.array([results[N][0]["lin "] for N in results]),
         t4_sinmu=np.array([results[N][0]["sinmu"] for N in results]),
         t4_min=np.array([results[N][1] for N in results]),
         t4_tbest=np.array([results[N][2] for N in results]))

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].semilogy(ts, S, lw=1.4, label="S(t): twisted Rayleigh quotient of G, N=2000")
ax[0].semilogy(ts, A * np.maximum(zz2, 1e-9), "--", lw=1.0, alpha=0.75,
               label=r"$A\,|\zeta(\frac{1}{2}+it)|^2$ (scaled)")
for z in ZEROS:
    ax[0].axvline(z, color="crimson", alpha=0.35, lw=0.9)
ax[0].set_xlabel("t"); ax[0].set_ylabel("S(t)")
ax[0].set_title("The symbol of the gcd matrix is zeta on the critical line\n"
                "(red lines: Riemann zeros -- no fit, no free parameters)")
ax[0].legend(fontsize=8)
Ns = np.array(list(results.keys()), dtype=float)
for key, lab in (("sin ", r"$\Omega=\sin(\pi u)$"), ("lin ", r"$\Omega=1-u$"),
                 ("sinmu", r"$\Omega=\mu\cdot\sin(\pi u)$")):
    ax[1].semilogx(Ns, [results[int(N)][0][key] for N in Ns], "o-", label=lab)
ax[1].semilogx(Ns, [results[int(N)][1] for N in Ns], "s--",
               label=r"$\min_t$ (sine profile)")
ax[1].axhline(0.836, color="k", lw=1, ls=":", label=r"measured $\lambda_{\min}\log^2N$")
ax[1].set_xlabel("N"); ax[1].set_ylabel(r"$S(\gamma_1)\cdot\log^2 N$")
ax[1].set_title("Constructor upper bound vs the measured floor constant")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.expanduser("~/rh_output/symbol_zero.png"), dpi=130)
print(f"\nfigure + npz saved; total {time.time()-t0:.0f}s")
