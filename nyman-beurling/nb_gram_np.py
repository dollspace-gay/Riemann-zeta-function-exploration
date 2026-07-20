"""
NumPy port of the exact NB Gram builder (no torch/GPU dependency).
Same scheme as nb_gram.py: exact piecewise head on [1, 1+L], periodized
trigamma tail, coprime pairs only + gcd scaling. Differences:
  - psi1 via Euler-Maclaurin with J=8 and corrections through y^-7
    (arguments are in [1, 2.1], absolute accuracy ~1e-12);
  - Gauss-Legendre 4 on the tail segments (integrand is a quadratic in w
    times a smooth trigamma factor; GL-4 is exact through degree 7).
Both shortcuts are validated against the closed-form anchors below and
must reproduce the June session values before any use.
"""

import numpy as np
import math, time

EULER_GAMMA = 0.5772156649015328606
LOG_2PI = math.log(2.0 * math.pi)
BURNOL_C = 2.0 + EULER_GAMMA - math.log(4.0 * math.pi)

GL4_X = np.array([-0.8611363115940526, -0.3399810435848563,
                   0.3399810435848563,  0.8611363115940526])
GL4_W = np.array([0.3478548451374538, 0.6521451548625461,
                  0.6521451548625461, 0.3478548451374538])

def psi1(x):
    """Trigamma, vectorized, for x >= 1 (used on [1, 2.1])."""
    s = np.zeros_like(x)
    for j in range(8):
        s += (x + j) ** -2
    y = x + 8.0
    y2 = y * y
    return (s + 1.0/y + 0.5/y2 + 1.0/(6.0*y2*y)
            - 1.0/(30.0*y2*y2*y) + 1.0/(42.0*y2*y2*y2*y))

def acop_batch(ms, ns):
    """Acop(m,n) = <e_m,e_n> for arrays of coprime pairs (m <= n)."""
    m = np.asarray(ms, dtype=np.int64)
    n = np.asarray(ns, dtype=np.int64)
    P = len(m)
    L = m * n
    cnt = n + m + 2
    tot = int(cnt.sum())
    pair_of = np.repeat(np.arange(P, dtype=np.int64), cnt)
    start = np.cumsum(cnt) - cnt
    local = np.arange(tot, dtype=np.int64) - start[pair_of]
    mm, nn, LL = m[pair_of], n[pair_of], L[pair_of]
    v = np.where(local == 0, 1,
        np.where(local <= nn, local * mm,
        np.where(local <= nn + mm, (local - nn) * nn, LL + 1)))
    key = pair_of * (int(L.max()) + 3) + v
    order = np.argsort(key, kind="stable")
    v_sorted = v[order].astype(np.float64)
    p_sorted = pair_of[order]
    same = p_sorted[1:] == p_sorted[:-1]
    v0 = v_sorted[:-1][same]
    v1 = v_sorted[1:][same]
    pidx = p_sorted[:-1][same]
    mmf = m[pidx].astype(np.float64)
    nnf = n[pidx].astype(np.float64)
    LLf = L[pidx].astype(np.float64)
    mid = 0.5 * (v0 + v1)
    a = np.floor(mid / mmf)
    bb = np.floor(mid / nnf)
    nonzero = v1 > v0
    def H(u):
        return u / (mmf * nnf) - (bb / mmf + a / nnf) * np.log(u) - a * bb / u
    head = np.where(nonzero, H(v1) - H(v0), 0.0)
    w0 = v0 - 1.0; w1 = v1 - 1.0
    half = 0.5 * (w1 - w0); cen = 0.5 * (w0 + w1)
    tail = np.zeros_like(v0)
    # large pairs (n >= 64): trigamma varies by <= 1/64 per segment, GL-4 ample
    big = nnf >= 64.0
    for q in range(4):
        wq = cen + half * GL4_X[q]
        u = 1.0 + wq
        f = (u / mmf - a) * (u / nnf - bb)
        contrib = GL4_W[q] * half * f * psi1((1.0 + LLf + wq) / LLf) / LLf**2
        tail += np.where(big, contrib, 0.0)
    # small pairs: subdivide each segment into 8, GL-8 on each piece
    sm = ~big
    if sm.any():
        GL8_X = np.array([-0.9602898564975363, -0.7966664774136267,
                          -0.5255324099163290, -0.1834346424956498,
                           0.1834346424956498,  0.5255324099163290,
                           0.7966664774136267,  0.9602898564975363])
        GL8_W = np.array([0.1012285362903763, 0.2223810344533745,
                          0.3137066458778873, 0.3626837833783620,
                          0.3626837833783620, 0.3137066458778873,
                          0.2223810344533745, 0.1012285362903763])
        w0s, w1s = w0[sm], w1[sm]
        mms, nns, LLs = mmf[sm], nnf[sm], LLf[sm]
        as_, bbs = a[sm], bb[sm]
        acc = np.zeros_like(w0s)
        for piece in range(8):
            p0 = w0s + (w1s - w0s) * piece / 8.0
            p1 = w0s + (w1s - w0s) * (piece + 1) / 8.0
            h2 = 0.5 * (p1 - p0); c2 = 0.5 * (p0 + p1)
            for q in range(8):
                wq = c2 + h2 * GL8_X[q]
                u = 1.0 + wq
                f = (u / mms - as_) * (u / nns - bbs)
                acc += GL8_W[q] * h2 * f * psi1((1.0 + LLs + wq) / LLs) / LLs**2
        tail[sm] = acc
    tail = np.where(nonzero, tail, 0.0)
    A = np.zeros(P)
    np.add.at(A, pidx, head + tail)
    return A + 1.0 / (m.astype(np.float64) * n.astype(np.float64))

def build_gram(N, chunk_cost=4_000_000, verbose=True):
    """Full A matrix (N x N) in L2(0, inf)."""
    cm, cn = [], []
    for mv in range(1, N + 1):
        ns = np.arange(mv, N + 1)
        mask = np.gcd(mv, ns) == 1
        cm.append(np.full(int(mask.sum()), mv)); cn.append(ns[mask])
    cm = np.concatenate(cm); cn = np.concatenate(cn)
    cost = cm + cn
    orderc = np.argsort(cost)
    cm, cn, cost = cm[orderc], cn[orderc], cost[orderc]
    Acop_tab = np.zeros((N + 1, N + 1))
    csum = np.cumsum(cost)
    i = 0
    t0 = time.time()
    while i < len(cm):
        j = int(np.searchsorted(csum, (csum[i - 1] if i else 0) + chunk_cost)) + 1
        j = min(j, len(cm))
        Acop_tab[cm[i:j], cn[i:j]] = acop_batch(cm[i:j], cn[i:j])
        i = j
    if verbose:
        print(f"  [coprime Gram: {len(cm)} pairs in {time.time()-t0:.1f}s]",
              flush=True)
    mg, ng = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij")
    gcds = np.gcd(mg, ng)
    mp = mg // gcds; np_ = ng // gcds
    lo = np.minimum(mp, np_); hi = np.maximum(mp, np_)
    return Acop_tab[lo, hi] / gcds

def b_vector(N):
    ks = np.arange(1, N + 1)
    return (np.log(ks) + 1.0 - EULER_GAMMA) / ks

def mobius(N):
    mu = np.ones(N + 1, dtype=np.int64)
    for p in range(2, N + 1):
        if all(p % q for q in range(2, int(p**0.5) + 1)):
            mu[p::p] *= -1
            mu[p*p::p*p] = 0
    return mu[1:]

if __name__ == "__main__":
    # Validation at N = 500 against closed forms and June session values
    t0 = time.time()
    A = build_gram(500)
    print(f"  build N=500: {time.time()-t0:.1f}s")
    checks = [
        ("A(1,1) vs log2pi-gamma", A[0, 0], LOG_2PI - EULER_GAMMA),
        ("A(1,2) vs closed form", A[0, 1], 0.75*(LOG_2PI-EULER_GAMMA) - math.log(2)/4),
        ("A(2,3) vs June", A[1, 2], 0.441103509),
        ("A(2,4)=A(1,2)/2 gcd", A[1, 3], A[0, 1] / 2.0),
    ]
    for name, got, ref in checks:
        print(f"  {name}: {got:.11f} vs {ref:.11f}  diff {abs(got-ref):.2e}")
    b = b_vector(500)
    Lc = np.linalg.cholesky(A)
    y = np.linalg.solve_triangular(Lc, b, lower=True) if hasattr(np.linalg, "solve_triangular") else None
    if y is None:
        from scipy.linalg import solve_triangular
        y = solve_triangular(Lc, b, lower=True)
    d2 = 1.0 - np.cumsum(y**2)
    for N, ref in [(10, 0.0237719), (100, 0.0101888), (500, 0.0074199)]:
        print(f"  d_{N}^2 = {d2[N-1]:.7f} vs June {ref:.7f}  "
              f"diff {abs(d2[N-1]-ref):.1e}")
    print(f"  d2*logN at 500: {d2[499]*math.log(500):.5f} (June: 0.04611)")
