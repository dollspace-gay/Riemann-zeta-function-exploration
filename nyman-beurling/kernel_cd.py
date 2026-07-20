"""
Plan 1, Session 1: the kernel constant c(d) derived exactly, and the
zero-sum spectral-floor pre-check.
=====================================================================
THEORY (derived first, validated here):

P_{jk} = 2 * integral over D of u^-2 du,  D = {u : floor(u/j)-floor(u/k) odd}.

1. GCD reduction: P(gj', gk') = P(j', k')/g  (substitute u = g v).
2. EXACT EVALUATION (new): for gcd(j,k)=1 the parity pattern of
   Delta(u) = floor(u/j)-floor(u/k) is periodic with period L = jk when
   d = k-j is even, and 2L when d is odd (Delta(u+L) = Delta(u) + d).
   Hence with Lp the parity period and {[a_i, b_i)} the odd-parity
   segments inside [0, Lp):

       P = 2 * sum_i [ (1/a_i - 1/b_i)
                       + (psi(1 + b_i/Lp) - psi(1 + a_i/Lp)) / Lp ],

   psi = digamma (the shifted-copies tail telescopes to digamma).
   Machine precision, O(K) segments per pair, no truncation.
3. ASYMPTOTIC LAW (derived): Delta has mean lam = u d/(jk); the local
   odd-fraction converges to the triangle wave f(lam) = dist(lam, 2Z).
   The disjoint-interval head is exactly a harmonic sum, the averaged
   tail is int_1^inf f(lam) lam^-2 dlam = 1 + log(2/pi)  (Wallis).
   PREDICTION:
       P_{jk} * jk/(2d) = H_{floor(j/d)} + c* + o(1),
       c* = 1 + log(2/pi) = 0.548417...
   independent of d and of gcd. The June drift c(d): 0.545 -> 0.661 was
   the K^2-vs-jk normalization plus harmonic-vs-log discreteness.

VALIDATION ANCHORS (must pass before anything is believed):
  A1. P(1249,1250) vs the Gram-derived 8*||f_{K-1}-f_K||^2 = 1.05763e-5
      (chains_theory.md par. 0) and the Theorem-1 head 2*H_{K-1}/(K(K-1)).
  A2. Brute-force check of P for small pairs by direct interval
      enumeration to huge cutoff with rigorous tail bracket.
  A3. lambda_chain(K=250, w=20) vs the cached-Gram value 6.481e-6
      (chains.py / RESULTS.md Session 3), M built ONLY from the exact
      decomposition M_jk = (log2/8)(1/j+1/k) - P_jk/16.

PRE-CHECK (Plan 1 Step 2): F(w,K) = K^2 * min eigenvalue of M restricted
to zero-sum coefficient vectors, windows (K-w, K]. The lower-bound lemma
requires F(w,K) bounded below as w grows. Falsify cheaply before proving.
"""

import numpy as np
from scipy.special import digamma
from math import log, gcd, pi
import os, sys, time

EULER_GAMMA = 0.5772156649015328606
C2 = log(2.0) / 4.0
CSTAR = 1.0 + log(2.0 / pi)

OUT = os.path.expanduser("~/rh_output")
os.makedirs(OUT, exist_ok=True)
out = None
def report(msg=""):
    global out
    if out is None:
        out = open(os.path.join(OUT, "kernel_cd.txt"), "w")
    print(msg, flush=True)
    out.write(msg + "\n"); out.flush()

# ---------------------------------------------------------------------
# Exact P via parity-period + digamma tail
# ---------------------------------------------------------------------
def P_exact(j, k):
    """P_{jk} = 2 int_D u^-2, D the parity-disagreement set. Exact."""
    if j == k:
        return 0.0
    if j > k:
        j, k = k, j
    g = gcd(j, k)
    j, k = j // g, k // g
    d = k - j
    L = j * k
    Lp = L if d % 2 == 0 else 2 * L
    # events: +1 at multiples of j, -1 at multiples of k, in (0, Lp)
    mj = np.arange(j, Lp + 1, j, dtype=np.int64)      # include Lp boundary
    mk = np.arange(k, Lp + 1, k, dtype=np.int64)
    pos = np.concatenate([mj, mk])
    dlt = np.concatenate([np.ones_like(mj), -np.ones_like(mk)])
    order = np.argsort(pos, kind="stable")
    pos, dlt = pos[order], dlt[order]
    # merge ties (common multiples)
    upos, inv = np.unique(pos, return_inverse=True)
    ud = np.zeros(len(upos), dtype=np.int64)
    np.add.at(ud, inv, dlt)
    delta = np.cumsum(ud)                              # Delta on [upos_i, upos_{i+1})
    # segments [a,b) with odd Delta, within (0, Lp); Delta=0 on (0, upos_0)
    a = upos[:-1].astype(np.float64)
    b = upos[1:].astype(np.float64)
    par = (delta[:-1] % 2).astype(bool)
    a, b = a[par], b[par]
    head = np.sum(1.0 / a - 1.0 / b)
    tail = np.sum(digamma(1.0 + b / Lp) - digamma(1.0 + a / Lp)) / Lp
    return 2.0 * (head + tail) / g

def M_entry(j, k):
    """<f_j, f_k> from the exact decomposition."""
    if j == k:
        return C2 / j
    return (log(2.0) / 8.0) * (1.0 / j + 1.0 / k) - P_exact(j, k) / 16.0

def _main():
    # ---------------------------------------------------------------------
    # A2 first: brute force on small pairs (rigorous bracket)
    # ---------------------------------------------------------------------
    report("=" * 74)
    report("A2. BRUTE-FORCE VALIDATION OF P_exact (small pairs)")
    report("=" * 74)

    def P_brute(j, k, U_over_jk=2000):
        """Direct enumeration to U = U_over_jk * jk; tail bracketed in [0, 2/U]."""
        U = U_over_jk * j * k
        mj = np.arange(j, U, j, dtype=np.int64)
        mk = np.arange(k, U, k, dtype=np.int64)
        pos = np.concatenate([mj, mk, [U]])
        dlt = np.concatenate([np.ones_like(mj), -np.ones_like(mk), [0]])
        order = np.argsort(pos, kind="stable")
        pos, dlt = pos[order], dlt[order]
        upos, inv = np.unique(pos, return_inverse=True)
        ud = np.zeros(len(upos), dtype=np.int64)
        np.add.at(ud, inv, dlt)
        delta = np.cumsum(ud)
        a = upos[:-1].astype(np.float64); b = upos[1:].astype(np.float64)
        par = (delta[:-1] % 2).astype(bool)
        P0 = 2.0 * np.sum(1.0 / a[par] - 1.0 / b[par])
        return P0, P0 + 2.0 / U          # [lower, upper] rigorous bracket

    ok_all = True
    for (j, k) in [(1, 2), (2, 3), (3, 4), (5, 7), (4, 6), (6, 10), (9, 12), (7, 12)]:
        pe = P_exact(j, k)
        lo, hi = P_brute(j, k)
        inside = (lo - 1e-12 <= pe <= hi + 1e-12)
        ok_all &= inside
        report(f"  P({j},{k}) exact {pe:.12f}  brute bracket "
               f"[{lo:.12f}, {hi:.12f}]  {'OK' if inside else 'FAIL'}")
    if not ok_all:
        report("  *** BRUTE-FORCE VALIDATION FAILED — STOP ***"); sys.exit(1)

    # ---------------------------------------------------------------------
    # A1: the Theorem-1 pair (K-1, K) at K = 1250
    # ---------------------------------------------------------------------
    report("")
    report("=" * 74)
    report("A1. THEOREM-1 PAIR ANCHOR, K = 1250")
    report("=" * 74)
    K = 1250
    pe = P_exact(K - 1, K)
    H = lambda m: np.sum(1.0 / np.arange(1, m + 1))
    head_thm = 2.0 * H(K - 1) / (K * (K - 1.0))
    report(f"  P(1249,1250) exact       = {pe:.6e}")
    report(f"  Gram-derived (Session 4) = 1.05763e-05   "
           f"rel diff {abs(pe - 1.05763e-5)/1.05763e-5:.2e}")
    report(f"  Theorem-1 head 2H/(K(K-1)) = {head_thm:.6e}  "
           f"(tail = {pe - head_thm:.3e}, bound 2/(K(K-1)) = {2.0/(K*(K-1)):.3e})")
    report(f"  c_nat = P*jk/(2d) - H_floor(j/d) = "
           f"{pe * (K-1) * K / 2.0 - H(K-1):.6f}   vs c* = {CSTAR:.6f}")

    # ---------------------------------------------------------------------
    # c(d, K): June normalization reproduced, natural normalization vs c*
    # ---------------------------------------------------------------------
    report("")
    report("=" * 74)
    report("c(d,K): JUNE NORMALIZATION vs THE DERIVED LAW")
    report("=" * 74)
    report(f"  c* = 1 + log(2/pi) = {CSTAR:.6f}")
    report("")
    Ks = [156, 312, 625, 1250, 2500, 5000]
    ds = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    report(f"{'K':>6} " + " ".join(f"{'d='+str(d):>9}" for d in ds))
    report("June def:  c_J = P*K^2/(2d) - log(K/d) - gamma      [pair (K-d, K)]")
    cJ_table = {}
    for Kv in Ks:
        row = []
        for d in ds:
            p = P_exact(Kv - d, Kv)
            cJ = p * Kv * Kv / (2.0 * d) - log(Kv / d) - EULER_GAMMA
            cJ_table[(Kv, d)] = cJ
            row.append(f"{cJ:>9.4f}")
        report(f"{Kv:>6} " + " ".join(row))
    report("")
    report("Natural def:  c_nat = P*jk/(2d) - H_floor(j/d)   [prediction: -> c*]")
    cN_table = {}
    for Kv in Ks:
        row = []
        for d in ds:
            j = Kv - d
            p = P_exact(j, Kv)
            cN = p * j * Kv / (2.0 * d) - H(j // d)
            cN_table[(Kv, d)] = cN
            row.append(f"{cN:>9.4f}")
        report(f"{Kv:>6} " + " ".join(row))
    report("")
    report("Deviation c_nat - c*  (should -> 0 with K at fixed d):")
    for Kv in Ks:
        row = [f"{cN_table[(Kv,d)] - CSTAR:>9.5f}" for d in ds]
        report(f"{Kv:>6} " + " ".join(row))
    # empirical convergence rate at d = 1: fit |c_nat - c*| ~ K^-q
    devs = np.array([abs(cN_table[(Kv, 1)] - CSTAR) for Kv in Ks])
    Karr = np.array(Ks, dtype=float)
    mask = devs > 0
    if mask.sum() >= 3:
        q = np.polyfit(np.log(Karr[mask]), np.log(devs[mask]), 1)[0]
        report(f"\n  d=1 deviation decay exponent: |c_nat - c*| ~ K^{q:.2f}")

    # gcd-independence spot check (the law should not care about gcd(j,k))
    report("")
    report("gcd-independence spot check (d=4, K=1250: gcd(j,k) = 1,2,4 cases):")
    for j in [1246, 1247, 1249]:            # wait: need k-j=4 => j = 1246; vary gcd via j choice
        pass
    for (j, k) in [(1246, 1250), (1247, 1251), (1244, 1248)]:
        d = k - j
        p = P_exact(j, k)
        cN = p * j * k / (2.0 * d) - H(j // d)
        report(f"  (j,k)=({j},{k}) gcd={gcd(j,k)}:  c_nat = {cN:.5f}")

    # ---------------------------------------------------------------------
    # A3: chain-subspace lambda vs the cached-Gram anchor (K=250, w=20)
    # ---------------------------------------------------------------------
    report("")
    report("=" * 74)
    report("A3. CHAIN WINDOW FROM EXACT DECOMPOSITION vs chains.py ANCHOR")
    report("=" * 74)

    def M_window(K, w):
        ks = np.arange(max(2, K - w + 1), K + 1)
        n = len(ks)
        M = np.empty((n, n))
        for a in range(n):
            M[a, a] = C2 / ks[a]
            for b_ in range(a + 1, n):
                M[a, b_] = M[b_, a] = M_entry(int(ks[a]), int(ks[b_]))
        return M, ks

    def lam_chain(K, w):
        M, _ = M_window(K, w)
        return np.linalg.eigvalsh(M)[0] / 1.25       # S = 1 + 1/4

    t0 = time.time()
    lc = lam_chain(250, 20)
    report(f"  lambda_chain(K=250, w=20) = {lc:.4e}   "
           f"(chains.py cached-Gram value: 6.481e-06; rel diff "
           f"{abs(lc - 6.481e-6)/6.481e-6:.1e})   [{time.time()-t0:.1f}s]")
    for (Kv, w, ref) in [(500, 20, 1.735e-6), (1000, 20, 4.642e-7), (1250, 20, 3.036e-7)]:
        lc = lam_chain(Kv, w)
        report(f"  lambda_chain(K={Kv:>4}, w={w}) = {lc:.4e}   "
               f"(cached-Gram: {ref:.3e}; rel diff {abs(lc-ref)/ref:.1e})")

    # ---------------------------------------------------------------------
    # PRE-CHECK: zero-sum spectral floor vs window width
    # ---------------------------------------------------------------------
    report("")
    report("=" * 74)
    report("STEP-2 PRE-CHECK: K^2 * zero-sum floor of M vs window width w")
    report("=" * 74)
    report("Lemma requires: bounded below (no decay) as w grows at fixed K.")
    report("")

    def zero_sum_floor(K, w):
        M, ks = M_window(K, w)
        n = len(ks)
        # orthonormal basis of the zero-sum subspace: complement of ones/sqrt(n)
        Q, _ = np.linalg.qr(np.eye(n) - np.full((n, n), 1.0 / n))
        Q = Q[:, : n - 1]
        Mz = Q.T @ M @ Q
        return np.linalg.eigvalsh(Mz)[0]

    report(f"{'w':>5} | " + " ".join(f"{'K='+str(Kv):>12}" for Kv in [250, 625, 1250]))
    floors = {}
    for w in [4, 8, 16, 32, 64, 128, 200]:
        row = []
        for Kv in [250, 625, 1250]:
            if w >= Kv - 2:
                row.append(f"{'--':>12}"); continue
            F = Kv * Kv * zero_sum_floor(Kv, w)
            floors[(w, Kv)] = F
            row.append(f"{F:>12.4f}")
        report(f"{w:>5} | " + " ".join(row))

    report("")
    report("Unconstrained floor K^2*eigmin(M) for contrast (same windows):")
    report(f"{'w':>5} | " + " ".join(f"{'K='+str(Kv):>12}" for Kv in [250, 625, 1250]))
    for w in [4, 8, 16, 32, 64, 128, 200]:
        row = []
        for Kv in [250, 625, 1250]:
            if w >= Kv - 2:
                row.append(f"{'--':>12}"); continue
            M, _ = M_window(Kv, w)
            row.append(f"{Kv*Kv*np.linalg.eigvalsh(M)[0]:>12.4f}")
        report(f"{w:>5} | " + " ".join(row))

    report("")
    report("Done.")
    out.close()

if __name__ == "__main__":
    _main()
