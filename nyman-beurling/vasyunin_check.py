"""
Plan 3B: the Vasyunin cross-check.
=====================================================================
SOURCE (obtained verbatim, not recalled): Darses-Hillion,
"An exponentially-averaged Vasyunin formula", arXiv:2004.10086, p.2,
quoting Vasyunin [Vas95] and Baez-Duarte-Balazard-Landreau-Saias
[BDBLS00, p.141]. For COPRIME m, n >= 1:

  mn * int_0^inf {1/(nt)}{1/(mt)} dt
      = C(n+m) + (m-n)/2 * log(n/m)
        - (pi/2) * sum_{k=1}^{n-1} {mk/n} cot(k pi/n)
        - (pi/2) * sum_{l=1}^{m-1} {ln/m} cot(l pi/m),
  C = (1/2)(log 2pi - gamma).

Hand-checks done before this script: (1,1) gives A(1,1) = 2C =
log 2pi - gamma; (2,1) gives A(1,2) = (3/2)C - log2/4 — exactly our
Session-3 closed form, so A(1,2) IS a Vasyunin special case (LITERATURE
flag resolved: 'presumably recoverable' upgraded to 'verified').

THIS SCRIPT: three-way cross-validation.
  1. Vasyunin formula vs our exact-head+trigamma A entries (acop_batch)
     on a spread of coprime pairs.
  2. Vasyunin formula vs gcd-scaled entries A(gm,gn) = A(m,n)/g.
  3. The parity-disagreement integral P(j,k) of kernel_cd.py vs the
     Vasyunin route: P = 8||f_j - f_k||^2 expanded in A entries at
     (j, k, 2j, 2k) — a third independent method for P.
"""

import numpy as np
import math
from math import gcd, pi, log
from nb_gram_np import acop_batch, EULER_GAMMA, LOG_2PI
from kernel_cd import P_exact

CVAS = 0.5 * (LOG_2PI - EULER_GAMMA)

def vasyunin_coprime(m, n):
    """A(m,n) for coprime m,n via the Vasyunin cotangent formula."""
    total = CVAS * (n + m) + 0.5 * (m - n) * log(n / m)
    if n > 1:
        k = np.arange(1, n)
        total -= (pi / 2) * np.sum(np.mod(m * k / n, 1.0) / np.tan(k * pi / n))
    if m > 1:
        l = np.arange(1, m)
        total -= (pi / 2) * np.sum(np.mod(l * n / m, 1.0) / np.tan(l * pi / m))
    return total / (m * n)

def vasyunin_A(m, n):
    g = gcd(m, n)
    return vasyunin_coprime(m // g, n // g) / g

print("=" * 74)
print("1. VASYUNIN vs EXACT-SCHEME A ENTRIES (coprime pairs)")
print("=" * 74)
pairs = [(1, 1), (1, 2), (2, 3), (3, 4), (5, 7), (1, 100), (7, 100),
         (99, 100), (13, 977), (500, 501), (999, 1000), (1249, 1250),
         (3, 1000), (29, 30), (127, 128), (100, 101), (250, 251),
         (17, 19), (123, 124), (1, 1999)]
ms = np.array([p[0] for p in pairs]); ns = np.array([p[1] for p in pairs])
ours = acop_batch(ms, ns)
worst = 0.0
for (m, n), av in zip(pairs, ours):
    vv = vasyunin_A(m, n)
    diff = abs(vv - av)
    worst = max(worst, diff)
    print(f"  A({m:>4},{n:>4}): vasyunin {vv:.12f}  ours {av:.12f}  "
          f"diff {diff:.2e}")
print(f"  WORST: {worst:.2e}")

print()
print("=" * 74)
print("2. GCD SCALING (non-coprime pairs)")
print("=" * 74)
pairs_nc = [(2, 4), (6, 10), (12, 18), (100, 150), (500, 750), (624, 1248)]
ms = np.array([p[0] // gcd(*p) for p in pairs_nc])
ns = np.array([p[1] // gcd(*p) for p in pairs_nc])
red = acop_batch(ms, ns)
for (m, n), av in zip(pairs_nc, red):
    g = gcd(m, n)
    vv = vasyunin_A(m, n)
    print(f"  A({m:>4},{n:>4}) = A({m//g},{n//g})/{g}: vasyunin {vv:.12f}  "
          f"ours {av/g:.12f}  diff {abs(vv - av/g):.2e}")

print()
print("=" * 74)
print("3. P(j,k) THREE WAYS: digamma-period vs Vasyunin-combination")
print("=" * 74)
print("   P = 8||f_j - f_k||^2,  f_k = e_2k - e_k/2  (expanded in A):")
for (j, k) in [(3, 4), (5, 7), (20, 21), (99, 100), (249, 250), (1249, 1250)]:
    # <f_a, f_b> = A(2a,2b) - A(2a,b)/2 - A(a,2b)/2 + A(a,b)/4
    def ip(a, bb):
        return (vasyunin_A(2*a, 2*bb) - vasyunin_A(2*a, bb)/2.0
                - vasyunin_A(a, 2*bb)/2.0 + vasyunin_A(a, bb)/4.0)
    Pv = 8.0 * (ip(j, j) + ip(k, k) - 2.0 * ip(j, k))
    Pd = P_exact(j, k)
    print(f"  P({j:>4},{k:>4}): vasyunin-route {Pv:.10e}  "
          f"digamma-route {Pd:.10e}  rel diff {abs(Pv-Pd)/Pd:.2e}")

print()
print("Done.")
