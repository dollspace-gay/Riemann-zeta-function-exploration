"""
Tail Variation Lemma: numerical validation of the proof's three new
identities BEFORE the writeup is trusted.
=====================================================================
The proof (chains_theory.md section 7) claims, for coprime j < k,
d = k - j, m0 = floor(j/d), U0 = j(m0+1):

 (I1) COLLAPSE. tail := 2 * int_{D cap [U0,inf)} u^-2 du equals the
      (l,t)-expansion
        tail = -(2) * [ -2 sum_l w(l,0) + 4 sum_l sum_{t>=1} (-1)^{t-1} w(l,t) ] / 2 ...
      concretely:  tail = -sum_{l>m0} c_t * w(l,t) summed with
      c_0 = -2, c_t = 4(-1)^{t-1}, w(l,t) = 1/(j(l+t)) - 1/(k l) > 0
      restricted to t < d l / j,  PLUS the identity
      int_{U0}^inf u^-2 = 1/U0 entering with opposite sign; i.e.
        tail = 1/U0 - [ 1/U0 + sum_{l,t} c_t w(l,t) ] = -sum c_t w.
      (No intersection reaches below U0: k*m0 <= U0 exactly.)
 (I2) TENT. With G(T) = sum_{1<=t<T} (-1)^{t-1} (1 - t/T):
        2 G(T) - 1 = -dist(T, 2Z)/T   exactly (T not an integer).
 (I3) LIMIT. tail * jk/(2d) = c* + O(d/j), c* = 1 + log(2/pi);
      measured slope A_eff of the deviation vs d/j.
"""

import numpy as np
import math
from math import gcd, log, pi
from kernel_cd import P_exact

CSTAR = 1.0 + log(2.0 / pi)

def head_tail(j, k):
    d = k - j
    m0 = j // d
    H = np.sum(1.0 / np.arange(1, m0 + 1))
    head = (2.0 * d / (j * k)) * H
    return head, P_exact(j, k) - head, m0

def tail_expansion(j, k, LMAX_mult=2000):
    """(l,t)-expansion of the tail, truncated at l <= LMAX with a
    rigorous truncation bracket [-(2d/jk)*(j/d)/LMAX-ish, +same]."""
    d = k - j
    m0 = j // d
    total = 0.0
    LMAX = LMAX_mult * max(m0, 1)
    for l in range(m0 + 1, LMAX + 1):
        Tl = d * l / j
        # t = 0 term
        w0 = 1.0 / (j * l) - 1.0 / (k * l)
        s = -2.0 * w0
        t = 1
        while t < Tl:
            w = 1.0 / (j * (l + t)) - 1.0 / (k * l)
            s += 4.0 * (-1.0) ** (t - 1) * w
            t += 1
        total += s
    # truncation: remaining sum bounded by ~ (2d/jk) * sum_{l>LMAX} 1/l * O(1/T)
    trunc = (2.0 * d / (j * k)) * (j / d) / LMAX
    return -total, trunc

print("=" * 72)
print("(I1) COLLAPSE IDENTITY: tail via exact P vs (l,t)-expansion")
print("=" * 72)
for (j, k) in [(40, 41), (40, 43), (81, 85), (200, 203), (500, 503), (711, 715)]:
    if gcd(j, k) != 1:
        continue
    head, tail_ex, m0 = head_tail(j, k)
    tail_sum, trunc = tail_expansion(j, k)
    ok = abs(tail_sum - tail_ex) <= 3 * trunc + 1e-14
    print(f"  (j,k)=({j},{k}): exact {tail_ex:.6e}  expansion {tail_sum:.6e}  "
          f"diff {abs(tail_sum-tail_ex):.1e} (trunc bound {trunc:.1e})  "
          f"{'OK' if ok else 'FAIL'}")

print()
print("=" * 72)
print("(I2) TENT IDENTITY: 2G(T) - 1 = -dist(T, 2Z)/T")
print("=" * 72)
rng = np.random.default_rng(5)
worst = 0.0
for T in np.concatenate([rng.uniform(1.01, 40, 60), [1.5, 2.5, 7.999, 8.001]]):
    t = np.arange(1, int(np.ceil(T)))
    t = t[t < T]
    G = float(np.sum((-1.0) ** (t - 1) * (1 - t / T)))
    lhs = 2 * G - 1
    rhs = -min(T % 2.0, 2.0 - T % 2.0) / T
    worst = max(worst, abs(lhs - rhs))
print(f"  64 random + boundary T in (1, 40): max |LHS - RHS| = {worst:.2e}")

print()
print("=" * 72)
print("(I3) THE LIMIT AND ITS RATE: tail*jk/(2d) - c* vs d/j")
print("=" * 72)
print(f"  c* = {CSTAR:.6f}")
print(f"  {'j':>6} {'d':>4} {'tail*jk/2d':>11} {'dev':>10} {'dev/(d/j)':>10}")
rows = []
for (j, d) in [(200, 1), (500, 1), (2000, 1), (200, 3), (500, 3), (2000, 3),
               (301, 5), (1001, 5), (3001, 5), (401, 7), (2003, 7),
               (500, 9), (2000, 9), (1000, 13), (4000, 13)]:
    k = j + d
    if gcd(j, k) != 1:
        continue
    head, tail_ex, m0 = head_tail(j, k)
    v = tail_ex * j * k / (2.0 * d)
    dev = v - CSTAR
    rows.append(abs(dev) / (d / j))
    print(f"  {j:>6} {d:>4} {v:>11.6f} {dev:>10.6f} {abs(dev)/(d/j):>10.3f}")
print(f"  measured A_eff = max |dev|/(d/j) over the table: {max(rows):.3f}")
