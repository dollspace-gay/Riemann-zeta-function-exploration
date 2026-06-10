"""
Session 6: verify the Mellin identity numerically, and the spectral story.
===========================================================================
IDENTITY (Plancherel + Mellin transform of e_k = -k^{-s} zeta(s)/s):

    x^T A x = (1/pi) * int_0^inf  |zeta(1/2+it)|^2/(1/4+t^2) * |X(t)|^2 dt,
    X(t) = sum_k x_k k^{-1/2-it}.

Checks:
  1. Identity vs Gram data for random vectors and chain vectors (N small).
  2. Chain difference f_{K-1} - f_K across K: Mellin integral vs exact P/8.
  3. The spectral profile: where the chain's weighted mass lives in t.
Tail beyond T_MAX estimated via the zeta mean value |zeta|^2 ~ log(t/2pi)
and reported as an uncertainty, not silently dropped.
"""

import numpy as np
import os, math, time

OUTPUT_DIR = os.path.expanduser("~/rh_output")
GRID_CACHE = os.path.join(OUTPUT_DIR, "zeta_grid.npz")

out = open(os.path.join(OUTPUT_DIR, "nb_mellin.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()

# ---------------------------------------------------------------------
# zeta grid (cached): |zeta(1/2+it)|^2 on a graded grid to T_MAX
# ---------------------------------------------------------------------
T_MAX = 2000.0
if os.path.exists(GRID_CACHE):
    d = np.load(GRID_CACHE)
    ts, zeta2 = d["ts"], d["zeta2"]
    report(f"[loaded zeta grid: {len(ts)} points to T={ts[-1]:.0f}]")
else:
    from mpmath import mp, zeta
    mp.dps = 15
    t1 = np.arange(0.0, 500.0, 0.02)
    t2 = np.arange(500.0, T_MAX + 1e-9, 0.05)
    ts = np.concatenate([t1, t2])
    report(f"computing |zeta(1/2+it)|^2 on {len(ts)} points (several minutes)...")
    t0 = time.time()
    zeta2 = np.empty(len(ts))
    for i, t in enumerate(ts):
        z = zeta(complex(0.5, float(t)))
        zeta2[i] = abs(complex(z))**2
        if i % 5000 == 0:
            report(f"  {i}/{len(ts)} ({time.time()-t0:.0f}s)")
    np.savez_compressed(GRID_CACHE, ts=ts, zeta2=zeta2)
    report(f"[grid done in {(time.time()-t0)/60:.1f} min, cached]")

w = zeta2 / (0.25 + ts**2)   # the weight

def mellin_quadform(x_idx, x_val):
    """(1/pi) * int w(t) |X(t)|^2 dt + tail estimate. Returns (value, tail)."""
    ks = np.asarray(x_idx, dtype=np.float64)
    cs = np.asarray(x_val, dtype=np.float64)
    # X(t) on grid
    phases = np.exp(-1j * np.outer(ts, np.log(ks)))
    X = phases @ (cs / np.sqrt(ks))
    integrand = w * np.abs(X)**2
    val = np.trapezoid(integrand, ts) / math.pi
    # tail: |X|^2 mean ~ sum c_k^2/k; |zeta|^2 mean ~ log(t/2pi)
    meanX2 = np.sum(cs**2 / ks)
    tail = meanX2 / math.pi * (math.log(T_MAX / (2*math.pi)) + 1.0) / T_MAX
    return val, tail

A = np.load(os.path.join(OUTPUT_DIR, "nb_gram_2500.npz"))["A"]

# ---------------------------------------------------------------------
# 1. identity vs Gram data
# ---------------------------------------------------------------------
report("")
report("=" * 70)
report("1. MELLIN IDENTITY vs GRAM DATA")
report("=" * 70)
rng = np.random.default_rng(3)
tests = []
for trial in range(3):
    idx = np.sort(rng.choice(np.arange(1, 13), size=6, replace=False))
    val = rng.standard_normal(6)
    tests.append((idx, val, f"random {trial}"))
tests.append((np.array([3, 6]), np.array([-0.5, 1.0]), "chain f_3"))
tests.append((np.array([7, 8, 14, 16]), np.array([-0.5, 0.5, 1.0, -1.0]),
              "chain diff f_7 - f_8"))
report(f"{'vector':>20} {'x^T A x':>12} {'Mellin':>12} {'tail est':>9} {'rel diff':>9}")
for idx, val, name in tests:
    lhs = float(val @ A[np.ix_(idx-1, idx-1)] @ val)
    rhs, tail = mellin_quadform(idx, val)
    report(f"{name:>20} {lhs:>12.8f} {rhs:>12.8f} {tail:>9.1e} "
           f"{abs(rhs-lhs)/abs(lhs):>9.2e}")

# ---------------------------------------------------------------------
# 2. chain difference across K vs exact P/8
# ---------------------------------------------------------------------
report("")
report("=" * 70)
report("2. CHAIN DIFFERENCE f_(K-1) - f_K: Mellin vs Gram (= P/8)")
report("=" * 70)
report(f"{'K':>5} {'Gram value':>13} {'Mellin':>13} {'tail':>9} {'rel diff':>9}")
for K in [20, 50, 100, 200]:
    idx = np.array([K-1, 2*(K-1), K, 2*K])
    val = np.array([-0.5, 1.0, 0.5, -1.0])
    lhs = float(val @ A[np.ix_(idx-1, idx-1)] @ val)
    rhs, tail = mellin_quadform(idx, val)
    report(f"{K:>5} {lhs:>13.4e} {rhs:>13.4e} {tail:>9.1e} "
           f"{abs(rhs-lhs)/abs(lhs):>9.2e}")

# ---------------------------------------------------------------------
# 3. spectral profile of the chain difference (K = 100)
# ---------------------------------------------------------------------
report("")
report("=" * 70)
report("3. WHERE THE CHAIN'S WEIGHTED MASS LIVES (K = 100)")
report("=" * 70)
K = 100
idx = np.array([K-1, 2*(K-1), K, 2*K]); val = np.array([-0.5, 1.0, 0.5, -1.0])
ks = idx.astype(float); cs = val
phases = np.exp(-1j * np.outer(ts, np.log(ks)))
X = phases @ (cs / np.sqrt(ks))
integrand = w * np.abs(X)**2
total = np.trapezoid(integrand, ts)
cum = lambda T: np.trapezoid(integrand[ts <= T], ts[ts <= T]) / total
for T in [10, 30, 100, 300, 1000, 2000]:
    report(f"  fraction of mass with t <= {T:>5}: {cum(T):.3f}")
report(f"  (theory: differencing adjacent dilations suppresses |X|^2 by")
report(f"   ~ t^2/K^2 below t ~ K = {K}, so the mass should sit at t ≳ K;")
report(f"   the log in Theorem 1 is the zeta second moment accumulating")
report(f"   between t ~ 1 and t ~ K.)")

report("")
report("Done.")
out.close()
