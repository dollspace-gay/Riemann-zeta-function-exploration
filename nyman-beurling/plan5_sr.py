"""
Plan 5, M1 numerics: the coefficient uncertainty lemma vs the actual
adversaries.
=====================================================================
Lemma (mellin_lower_bound.md M1): for unit x, with s_r = sum_{d|r} x_d,

    Q(x) = sum_{r<=N} tau(r) s_r^2 / r  >=  1/(N(log N + 1)).

Saturation := Q(x) * N (log N + 1)  (>= 1; near 1 = lemma tight there).

Tabulated for: the lambda_min eigenvector of A_N (the true adversary),
the Theorem-1 chain witness, a random unit vector, and delta_1.
"""

import numpy as np
import math, os

OUT = os.path.expanduser("~/rh_output")
A = np.load(os.path.join(OUT, "nb_gram_2000_np.npz"))["A"]

out = open(os.path.join(OUT, "plan5_sr.txt"), "w")
def report(msg=""):
    print(msg, flush=True)
    out.write(msg + "\n"); out.flush()

def tau_sieve(N):
    t = np.zeros(N + 1, dtype=np.int64)
    for k in range(1, N + 1):
        t[k::k] += 1
    return t

def s_vec(x):
    """s_r = sum_{d|r} x_d for r <= N (divisor-lattice partial sums)."""
    N = len(x)
    s = np.zeros(N + 1)
    for d in range(1, N + 1):
        s[d::d] += x[d - 1]
    return s[1:]

def Q(x):
    N = len(x)
    tau = tau_sieve(N)[1:]
    s = s_vec(x)
    return np.sum(tau * s * s / np.arange(1.0, N + 1))

report("=" * 74)
report("UNCERTAINTY-LEMMA SATURATION  Q(x) * N(log N + 1)   [>= 1 always]")
report("=" * 74)
report(f"{'N':>6} {'lmin eigvec':>12} {'chain witness':>14} "
       f"{'random':>10} {'delta_1':>12} {'lambda_min':>12}")

rng = np.random.default_rng(11)
for N in [200, 500, 1000, 2000]:
    AN = A[:N, :N]
    evals, evecs = np.linalg.eigh(AN)
    v = evecs[:, 0]
    lam = evals[0]
    bound = 1.0 / (N * (math.log(N) + 1.0))
    # chain witness (Theorem 1): K = N//2
    K = N // 2
    w = np.zeros(N)
    w[K - 2] = -0.5; w[2 * K - 3] = 1.0   # indices K-1, 2K-2 (1-based)
    w[K - 1] = 0.5;  w[2 * K - 1] = -1.0  # indices K,   2K
    w /= np.linalg.norm(w)
    r = rng.standard_normal(N); r /= np.linalg.norm(r)
    d1 = np.zeros(N); d1[0] = 1.0
    report(f"{N:>6} {Q(v)/bound:>12.2f} {Q(w)/bound:>14.2f} "
           f"{Q(r)/bound:>10.1f} {Q(d1)/bound:>12.1f} {lam:>12.3e}")

report("")
report("Reading: the eigenvector column is the adversarial saturation —")
report("how far above the lemma floor the TRUE minimizer sits. Growth of")
report("that column with N measures the lemma's loss against lambda_min.")

# structure of s_r for the adversary: where does its s-mass live?
report("")
report("=" * 74)
report("s_r STRUCTURE OF THE lambda_min EIGENVECTOR (N = 2000)")
report("=" * 74)
N = 2000
evals, evecs = np.linalg.eigh(A[:N, :N])
v = evecs[:, 0]
s = s_vec(v)
contrib = tau_sieve(N)[1:] * s * s / np.arange(1.0, N + 1)
top = np.argsort(contrib)[::-1][:12]
report(f"{'r':>6} {'s_r':>12} {'tau(r)s_r^2/r':>15} {'v_r':>12}")
for r_ in top:
    report(f"{r_+1:>6} {s[r_]:>12.5f} {contrib[r_]:>15.3e} {v[r_]:>12.5f}")
report(f"\n  total Q = {np.sum(contrib):.4e}; top-12 share "
       f"{np.sum(contrib[top])/np.sum(contrib):.2%}")
report("Done.")
out.close()
