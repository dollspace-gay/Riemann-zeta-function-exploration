"""
Shared Nyman-Beurling Gram machinery (see README.md for derivations).
A_{mn} = <e_m, e_n>_{L2(0,inf)}, e_k(t) = rho(1/(kt)).
Exact head + trigamma-tail scheme; only coprime pairs computed directly.
"""

import torch
import numpy as np
import math, time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

EULER_GAMMA = 0.5772156649015328606
LOG_2PI = math.log(2.0 * math.pi)
BURNOL_C = 2.0 + EULER_GAMMA - math.log(4.0 * math.pi)

def psi1(x):
    """Trigamma for x >= 0.5, vectorized torch float64 (Euler-Maclaurin)."""
    J = 16
    s = torch.zeros_like(x)
    for j in range(J):
        s = s + (x + j) ** -2
    y = x + J
    return s + 1.0/y + 0.5/y**2 + 1.0/(6.0*y**3) - 1.0/(30.0*y**5)

GL_X = torch.tensor([-0.9602898564975363, -0.7966664774136267,
                     -0.5255324099163290, -0.1834346424956498,
                      0.1834346424956498,  0.5255324099163290,
                      0.7966664774136267,  0.9602898564975363],
                    dtype=DTYPE, device=DEVICE)
GL_W = torch.tensor([0.1012285362903763, 0.2223810344533745,
                     0.3137066458778873, 0.3626837833783620,
                     0.3626837833783620, 0.3137066458778873,
                     0.2223810344533745, 0.1012285362903763],
                    dtype=DTYPE, device=DEVICE)

def acop_batch(ms, ns):
    """Acop(m,n) for arrays of coprime pairs (m <= n)."""
    P = len(ms)
    m = torch.tensor(ms, dtype=torch.int64, device=DEVICE)
    n = torch.tensor(ns, dtype=torch.int64, device=DEVICE)
    L = m * n
    cnt = (n + m + 2)
    tot = int(cnt.sum().item())
    pair_of = torch.repeat_interleave(torch.arange(P, device=DEVICE), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(tot, device=DEVICE) - start[pair_of]
    mm, nn, LL = m[pair_of], n[pair_of], L[pair_of]
    v = torch.where(local == 0, torch.ones_like(LL),
        torch.where(local <= nn, local * mm,
        torch.where(local <= nn + mm, (local - nn) * nn, LL + 1)))
    key = pair_of * (int(L.max().item()) + 3) + v
    order = torch.argsort(key)
    v_sorted = v[order].to(DTYPE)
    p_sorted = pair_of[order]
    same = p_sorted[1:] == p_sorted[:-1]
    v0 = v_sorted[:-1][same]
    v1 = v_sorted[1:][same]
    pidx = p_sorted[:-1][same]
    mm = m[pidx].to(DTYPE); nn = n[pidx].to(DTYPE); LL = L[pidx].to(DTYPE)
    mid = 0.5 * (v0 + v1)
    a = torch.floor(mid / mm)
    bb = torch.floor(mid / nn)
    nonzero = (v1 > v0)
    def H(u):
        return u / (mm * nn) - (bb / mm + a / nn) * torch.log(u) - a * bb / u
    head = torch.where(nonzero, H(v1) - H(v0), torch.zeros_like(v0))
    w0 = v0 - 1.0; w1 = v1 - 1.0
    half = 0.5 * (w1 - w0); cen = 0.5 * (w0 + w1)
    tail = torch.zeros_like(v0)
    for q in range(8):
        wq = cen + half * GL_X[q]
        u = 1.0 + wq
        f = (u / mm - a) * (u / nn - bb)
        tail = tail + GL_W[q] * half * f * psi1((1.0 + LL + wq) / LL) / LL**2
    tail = torch.where(nonzero, tail, torch.zeros_like(tail))
    A = torch.zeros(P, dtype=DTYPE, device=DEVICE)
    A.index_add_(0, pidx, head + tail)
    return A + 1.0 / (m.to(DTYPE) * n.to(DTYPE))

def build_gram(N, chunk_cost=4_000_000, verbose=True):
    """Full A matrix (N x N) in L2(0,inf)."""
    cm, cn = [], []
    for mv in range(1, N+1):
        ns = np.arange(mv, N+1)
        mask = np.gcd(mv, ns) == 1
        cm.append(np.full(mask.sum(), mv)); cn.append(ns[mask])
    cm = np.concatenate(cm); cn = np.concatenate(cn)
    cost = cm + cn
    orderc = np.argsort(cost)
    cm, cn, cost = cm[orderc], cn[orderc], cost[orderc]
    Acop_tab = np.zeros((N+1, N+1))
    i = 0
    t0 = time.time()
    csum = np.cumsum(cost)
    while i < len(cm):
        j = int(np.searchsorted(csum, (csum[i-1] if i else 0) + chunk_cost)) + 1
        j = min(j, len(cm))
        vals = acop_batch(cm[i:j], cn[i:j]).cpu().numpy()
        Acop_tab[cm[i:j], cn[i:j]] = vals
        i = j
    if verbose:
        print(f"  [coprime Gram: {len(cm)} pairs in {time.time()-t0:.1f}s]")
    mg, ng = np.meshgrid(np.arange(1, N+1), np.arange(1, N+1), indexing="ij")
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
