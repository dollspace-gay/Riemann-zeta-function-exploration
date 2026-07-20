"""
The out-of-sample test: push the trial-function spectroscopy to N = 10^6.
=========================================================================
The amplitude law (amplitude_theory.md, derived BEFORE this run):

    A_j(N) = 4 log(2pi) N^{-1/2} / (|rho_j|^2 |zeta'(rho_j)|)

predicts gamma_4, gamma_5 become detectable around N ~ 10^5-10^6, with
amplitudes given in advance. This script states the predictions, then
measures.

METHOD (no Gram matrix; near-linear in N): the Plan-3A exact scheme.
In u = 1/t coordinates, chi - sum c_k e_k is piecewise linear with
integer breakpoints: value (1 + T(n) - uS) on [n, n+1), where
T(n) = sum_k c_k floor(n/k) builds by divisor sieve and S = sum c_k/k.
Each unit interval integrates in closed form; the t > 1 part is S^2.
Exact up to the extrapolated tail beyond U = U_MULT * N.

VALIDATION GATES (before the big run):
  G1. Agreement with the Gram-based trial curve (gpu10k data) at
      N = 1000 ... 10^4 to well below the oscillation amplitudes.
  G2. U-convergence: doubling U moves D~^2 by less than the target
      accuracy at 10^4.

MATCHED FILTER: analyze  y = (D~^2 logN - drift) * sqrt(N) / (4 log 2pi);
under the law, each zero contributes a CONSTANT-amplitude cosine with
amplitude 1/(|rho_j|^2 |zeta'(rho_j)|) - the flat-spectrum form.
"""

import numpy as np
import math, os, time, sys

OUT = os.path.expanduser("~/rh_output")
L2PI = math.log(2.0 * math.pi)
EULER_GAMMA = 0.5772156649015328606

def mobius_fast(N):
    """Linear-ish Mobius sieve via smallest prime factor."""
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i][spf[i::i] == 0] = i
    mu = np.ones(N + 1, dtype=np.int64)
    n = np.arange(N + 1)
    # factor by repeated division with spf, vectorized-ish per pass
    rem = n.copy()
    last = np.zeros(N + 1, dtype=np.int64)
    alive = np.ones(N + 1, dtype=bool)
    while True:
        m = alive & (rem > 1)
        if not m.any():
            break
        p = spf[rem[m]]
        sq = p == last[m]
        idx = np.where(m)[0]
        mu[idx[sq]] = 0
        alive[idx[sq]] = False
        keep = idx[~sq]
        mu[keep] *= -1
        last[keep] = p[~sq]
        rem[keep] //= p[~sq]
    mu[0] = 0
    return mu[1:]

def Dtrial_exact(N, mu_all, U_mult=128):
    """D~_N^2 for c_k = -mu(k)(1 - log k/log N), exact head + tail est."""
    ks = np.arange(1, N + 1, dtype=np.float64)
    c = -mu_all[:N] * (1.0 - np.log(ks) / math.log(N))
    S = float(np.sum(c / ks))
    U = U_mult * N
    inc = np.zeros(U + 1)
    for k in range(1, N + 1):
        ck = c[k - 1]
        if ck != 0.0:
            inc[k::k] += ck
    T = np.cumsum(inc)
    # closed-form per-interval integrals, chunked
    head = 0.0
    last_dec = 0.0; prev_dec = 0.0
    CH = 20_000_000
    for lo in range(1, U, CH):
        hi = min(lo + CH, U)
        n = np.arange(lo, hi, dtype=np.float64)
        a = 1.0 + T[lo:hi] - 0.0  # value coefficient; u-slope is S
        piece = (a * a * (1.0 / n - 1.0 / (n + 1.0))
                 - 2.0 * a * S * np.log((n + 1.0) / n)
                 + S * S)
        head += float(np.sum(piece))
        # accumulate last two decades for tail extrapolation
        m1 = n >= U / 10.0
        m2 = (n >= U / 100.0) & (n < U / 10.0)
        last_dec += float(np.sum(piece[m1]))
        prev_dec += float(np.sum(piece[m2]))
    ratio = last_dec / prev_dec if prev_dec > 0 else 0.5
    tail = last_dec * ratio / max(1.0 - ratio, 1e-9)
    return S * S + head + tail, tail

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"

    if mode == "validate":
        mu = mobius_fast(10000)
        g = np.load(os.path.join(OUT, "nb_10k_gpu.npz"))
        Ng, D2g = g["Ngrid"], g["D2"]
        print("G1: sieve-exact vs Gram-based trial curve:")
        for Nt in [1000, 3162, 10000]:
            i = int(np.argmin(np.abs(Ng - Nt)))
            v, tail = Dtrial_exact(int(Ng[i]), mu, U_mult=128)
            print(f"  N={Ng[i]:>6}: sieve {v:.8e}  gram {D2g[i]:.8e}  "
                  f"diff {abs(v-D2g[i]):.2e}  (tail_est {tail:.1e})")
        print("G2: U-convergence at N=10^4:")
        for um in [64, 128, 256]:
            v, tail = Dtrial_exact(10000, mu, U_mult=um)
            print(f"  U={um}N: D~^2 = {v:.9e}  (tail_est {tail:.2e})")

    elif mode == "predict":
        import mpmath as mp
        mp.mp.dps = 25
        print("STATED IN ADVANCE - predicted line amplitudes A_j(N) in D~^2 logN:")
        rows = []
        for j in range(1, 7):
            rho = mp.zetazero(j)
            gam = float(mp.im(rho)); zd = float(abs(mp.zeta(rho, derivative=1)))
            r2 = 0.25 + gam * gam
            rows.append((gam, r2, zd))
            a5 = 4 * L2PI / (math.sqrt(1e5) * r2 * zd)
            a6 = 4 * L2PI / (math.sqrt(1e6) * r2 * zd)
            print(f"  g{j} = {gam:8.4f}  |zeta'| = {zd:.4f}   "
                  f"A(10^5) = {a5:.2e}   A(10^6) = {a6:.2e}")
        np.save(os.path.join(OUT, "predicted_lines.npy"), np.array(rows))

    elif mode == "run":
        NMAX = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000
        mu = mobius_fast(NMAX)
        print(f"mobius sieve to {NMAX} done", flush=True)
        grid = np.unique(np.round(np.geomspace(10000, NMAX, 110)).astype(int))
        vals, tails = [], []
        t0 = time.time()
        for i, N in enumerate(grid):
            v, tl = Dtrial_exact(int(N), mu, U_mult=256)
            vals.append(v); tails.append(tl)
            if i % 10 == 0 or i == len(grid) - 1:
                np.savez(os.path.join(OUT, "bigN_curve_partial.npz"),
                         Ngrid=grid[:i+1], D2=np.array(vals), tails=np.array(tails))
                print(f"  [{i+1}/{len(grid)}] N={N}  D2logN={v*math.log(N):.6f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        np.savez(os.path.join(OUT, "bigN_curve.npz"),
                 Ngrid=grid, D2=np.array(vals), tails=np.array(tails))
        print(f"DONE in {time.time()-t0:.0f}s -> bigN_curve.npz", flush=True)

def Dtrial_blocked(N, mu_all, U_mult=96, BS=250_000_000):
    """Memory-blocked version of Dtrial_exact for N ~ 10^7 (U up to ~1e9;
    the divisor sieve, cumulative sum, and piece integrals stream through
    blocks with a running carry). Identical mathematics."""
    ks = np.arange(1, N + 1, dtype=np.float64)
    c = -mu_all[:N] * (1.0 - np.log(ks) / math.log(N))
    S = float(np.sum(c / ks))
    U = U_mult * N
    T_carry = 0.0
    head = 0.0
    last_dec = 0.0; prev_dec = 0.0
    for blo in range(1, U, BS):
        bhi = min(blo + BS, U)
        Lb = bhi - blo
        inc = np.zeros(Lb)
        for k in range(1, N + 1):
            ck = c[k - 1]
            if ck != 0.0:
                start = ((blo + k - 1) // k) * k
                if start < bhi:
                    inc[start - blo::k] += ck
        T = np.cumsum(inc)
        T += T_carry
        T_carry = float(T[-1])
        n = np.arange(blo, bhi, dtype=np.float64)
        a = 1.0 + T
        piece = (a * a * (1.0 / n - 1.0 / (n + 1.0))
                 - 2.0 * a * S * np.log((n + 1.0) / n)
                 + S * S)
        head += float(np.sum(piece))
        m1 = n >= U / 10.0
        m2 = (n >= U / 100.0) & (n < U / 10.0)
        last_dec += float(np.sum(piece[m1]))
        prev_dec += float(np.sum(piece[m2]))
    ratio = last_dec / prev_dec if prev_dec > 0 else 0.5
    tail = last_dec * ratio / max(1.0 - ratio, 1e-9)
    return S * S + head + tail, tail
