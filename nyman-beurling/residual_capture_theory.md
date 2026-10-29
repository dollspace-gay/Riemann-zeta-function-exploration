# Möbius corrections against the actual optimal residual

September 23, 2026. **The uniform residual-gain estimate remains unproved.**
This investigation extends the numerical range, derives an exact arithmetic
expression for the correlation that must be bounded, proves that removing an
old-span component improves the correction, and tests a proposed transfer from
the tapered trial approximation. The transfer estimate is too weak in every
tested case. No asymptotic or novelty claim is made.

The September 24 [direct noncancellation attempt](mobius_correlation_identity.md)
proves a fixed-residual limit from classical L1 convergence, derives the chain
limit separately, and identifies the remaining uniformity and norm-cost gap.


## 1. The recipe and the relevant quantity

Use the spaces, projections, and squared errors from
[the refinement note](refinement_theory.md):

\[
 V_N=\operatorname{span}(e_1,\ldots,e_N),\quad
 p_N=P_N\chi,\quad r_N=\chi-p_N,\quad E_N=\|r_N\|^2.
\]

Among the five explicit directions previously screened, the Möbius direction
had the largest minimum normalized gain over all integers \(16\le N\le512\).
Put \(K=\lfloor N/2\rfloor\), and investigate

\[
 h_N=\sum_{K<k\le N}\mu(k)(e_{2k}-e_k/2),\qquad
 C_N=\langle r_N,h_N\rangle,\quad D_N=\|h_N\|^2.
\]

The best scalar correction gains \(C_N^2/D_N\). We report
\(\eta_N=\log(N)C_N^2/(E_ND_N)\), since a uniform positive lower bound
along sufficiently large dyadic scales would suffice for RH. The observed
minimum over a finite collection of sizes is not such a bound.

## 2. Exact arithmetic description, with no trial-residual substitution

Write \(p_N=\sum_{k\le N}c_ke_k\), with the **optimal** coefficients
\(c=A_N^{-1}b_N\). In \(u=1/t\) coordinates, put

\[
 a_N=\sum_{k\le N}c_k/k,\qquad
 B_j=1+\sum_{k\le N}c_k\lfloor j/k\rfloor.
\]

On \(j<u<j+1\), \(j\ge1\), the actual residual is
\(R_N(u)=B_j-a_Nu\). Define the truncated divisor sum

\[
 F_m(x)=\sum_{k\le m}\mu(k)\lfloor x/k\rfloor,\qquad
 M(x)=\sum_{k\le x}\mu(k).
\]

The square-wave identity for each chain gives

\[
 H_N(u):=h_N(1/u)
 =\tfrac12(F_N(u)-F_K(u))-(F_N(u/2)-F_K(u/2)).                 \tag{1}
\]

Reversing a finite sum also gives

\[
 F_N(x)-F_K(x)
 =\sum_{q\le x/(K+1)}
  \big[M(\min(N,\lfloor x/q\rfloor))-M(\min(K,\lfloor x/q\rfloor))\big]. \tag{2}
\]

These follow simply by counting the integers \(q\) with \(kq\le x\).
In particular \(H_N=0\) on \([0,K+1)\), and

\[
 H_N(u)=\tfrac12(M(\lfloor u\rfloor)-M(K))
 \quad(K+1\le u\le N).                                      \tag{3}
\]

Let \(H_j\) be the value on \((j,j+1)\). Integrating the affine residual
on each unit cell yields the exact, absolutely convergent identities

\[
 C_N=\sum_{j\ge K+1}H_j
 \left[\frac{B_j}{j(j+1)}-a_N\log(1+1/j)\right],\qquad
 D_N=\sum_{j\ge K+1}\frac{H_j^2}{j(j+1)}.                    \tag{4}
\]

Absolute convergence follows from boundedness of the finite syntheses on
\(u\ge1\) and integrability of \(u^{-2}\). Thus (1)–(4) identify the
Möbius cancellation against the actual residual, not against a convenient
approximation to it. The coefficients \(c_k\) are still present: these
identities do not supply bounds on their signed correlation with \(H_j\).

## 3. A provably better version of every correction

Decompose \(h_N=P_Nh_N+\widetilde h_N\), where
\(\widetilde h_N=(I-P_N)h_N\). Since \(r_N\perp V_N\),

\[
 \langle r_N,\widetilde h_N\rangle=C_N,\qquad
 \widetilde D_N=\|\widetilde h_N\|^2
 =D_N-\|P_Nh_N\|^2\le D_N.
\]

When \(\widetilde h_N\ne0\), its best scalar gain is therefore at least
that of \(h_N\). The correction still belongs to \(V_{2N}\); removing
its old component just refits old coefficients. In matrix form, for
\(w_i=\langle e_i,h_N\rangle\),

\[
 \widetilde D_N=D_N-w^TA_N^{-1}w,\qquad
 \text{gain}=C_N^2/\widetilde D_N.                            \tag{5}
\]

The general Hilbert-space inequality is now Lean-checked as
`Refinement.orthogonal_cleanup_gain`, with explicit orthogonality and
nonzero-direction hypotheses. This is a standard projection fact, not a
new arithmetic estimate. It avoids spending correction norm on a component
that has zero correlation with the residual.

The computations also examine the tapered chain weights
\(\mu(k)\log(2N/k)\) and the full-block direction
\(\sum_{N<k\le2N}\mu(k)e_k\), before and after the same projection.
None replaces the fully optimized baseline by a chain-only approximation.

## 4. The proposed trial-to-optimal transfer does not yet help

For any trial \(p_N^*\in V_N\), write \(r_N^*=r_N+\delta_N\), where
\(\delta_N=p_N-p_N^*\in V_N\). Orthogonality and Cauchy–Schwarz give

\[
 E_N^*-E_N=\|\delta_N\|^2,
\]
\[
 |\langle r_N^*,h_N\rangle-C_N|
 =|\langle\delta_N,P_Nh_N\rangle|
 \le\sqrt{(E_N^*-E_N)(D_N-\widetilde D_N)}.                  \tag{6}
\]

The excess-energy identity and transfer inequality are Lean-checked as
`Refinement.trial_excess_energy` and `Refinement.trial_correlation_transfer`.
For the earlier tapered-Möbius trial, its Chebyshev identity is useful only
if this discrepancy can be controlled sufficiently sharply.

**Test result:** in all 946 distinct tested sizes, subtracting the right
side of (6) from the magnitude of the trial correlation gives no positive
lower bound for \(|C_N|\). At \(N=4096\), for example:

| Quantity | Value |
|---|---:|
| Optimal energy \(E_N\) | 0.005400479114 |
| Trial energy \(E_N^*\) | 0.056879746387 |
| Actual correlation \(C_N\) | 0.001842769696 |
| Trial correlation | 0.001159448572 |
| Transfer error bound (6) | 0.033572911255 |

The actual discrepancy is much smaller than its bound, but the bound is
what a proof would need. This rejects the **current Cauchy–Schwarz transfer
argument**, not every possible use of the trial or Chebyshev identities.
The conditional tapered-Möbius result of
[Bettin–Conrey–Farmer](https://arxiv.org/abs/1211.5191) assumes RH and a
zeta-derivative moment bound, so it cannot supply the missing unconditional
step here.

## 5. What survives at larger sizes

The main scan covers every integer \(16\le N\le512\), then every
sixteenth integer through 4096: 721 sizes. Follow-up scans cover every
integer in 1104–1168, 1344–1456, and 3472–3536, around weak observed gains.
Together these give 946 distinct sizes, **not every integer through 4096**.

| Recipe | Smallest raw \(\eta_N\) observed | Smallest projected \(\eta_N\) observed |
|---|---:|---:|
| Möbius chains | 0.01785777 at N=1132 | 0.04113155 at N=1132 |
| Tapered Möbius chains | 0.01673178 at N=1369 | 0.05660692 at N=1418 |
| Full-block Möbius | 0.00640824 at N=3514 | 0.06820084 at N=3514 |

The previous raw Möbius minimum, 0.0379382 through N=512, does not persist
over this larger finite range. This does not refute the existence of a
smaller eventual dyadic bound, but it prevents promoting that earlier
observed constant to a universal one.

Selected dyadic values:

| N | Raw Möbius-chain \(\eta_N\) | Projected Möbius-chain \(\eta_N\) | Projected full-block Möbius \(\eta_N\) | Full optimal refinement \(\eta_N\) |
|---:|---:|---:|---:|---:|
| 512 | 0.172299 | 0.293268 | 0.584131 | 0.724450 |
| 1024 | 0.074568 | 0.129741 | 0.255102 | 0.448304 |
| 2048 | 0.189518 | 0.275871 | 0.474832 | 0.881504 |
| 4096 | 0.109886 | 0.203499 | 0.216239 | 0.517487 |

The squared errors are \(E_{4096}\approx0.005400479114\) and
\(E_{8192}\approx0.005064489997\). Finite improvement does not determine
the limiting error.

The arithmetic alignment ratio
\(|\sum\mu(k)q_k|/\sum|\mu(k)q_k|\) is about 0.926 at N=32, 0.275 at
512, 0.102 at 1024, and 0.136 at 4096. A proof cannot treat these signed
contributions as if they were all positive. The available support energy
times \(\log N/E_N\) stays around 2.4–2.7 at the profiled scales, while
the chosen direction captures only a small fraction of that energy. These
measurements distinguish energy availability from directional cancellation;
neither is a uniform theorem.

## 6. A sharper, explicit sufficient arithmetic target

For an integer \(J\), let \(C_N(J)\) be (4) restricted to \(j<J\), and
let \(T_N(J)\), \(D_N(J)\) be the residual and correction squared energies
over \(u\ge J\). Cauchy–Schwarz on that tail proves

\[
 |C_N|\ge L_N(J):=
 \big(|C_N(J)|-\sqrt{T_N(J)D_N(J)}\big)_+.                   \tag{7}
\]

This is an exact written inequality. Its evaluation here is floating point,
not an interval certificate. Using the actual residual and \(J=64N\), the
computed expression \(\log(N)L_N(J)^2/(E_ND_N)\) is:

| N | Conservative expression from (7) | Direct \(\eta_N\) |
|---:|---:|---:|
| 32 | 0.068977 | 0.077975 |
| 128 | 0.198763 | 0.216340 |
| 512 | 0.150209 | 0.172299 |
| 1024 | 0.062358 | 0.074568 |
| 2048 | 0.168268 | 0.189518 |
| 4096 | 0.095457 | 0.109886 |

Thus the positive computed overlap survives subtracting a bound on the
entire omitted tail. For the projected recipe, the more favorable denominator
\(\widetilde D_N\) may be used without changing the numerator in (7).

**Unproved target:** establish from arithmetic and the optimality equations,
with constants independent of sufficiently large dyadic N,

\[
 L_N(64N)^2\ge\frac{c}{\log N}E_N\widetilde D_N,
 \qquad c>0,\quad \widetilde D_N>0.                         \tag{8}
\]

The particular factor 64 is an experimental choice, not necessary for a proof.
Another cutoff or nonsummable sequence of guaranteed fractional gains also
suffices. Equation (8) isolates a finite signed Möbius/residual sum and a
controlled tail. **No lower estimate for that signed sum uniform in N has
been obtained.** This is a sufficient target for this recipe; it might fail
even if RH is true. The general optimized residual-gain target remains open.

## 7. Numerical and formal verification

The new bulk builder uses the established finite Vasyunin formula, already
checked on individual pairs in `vasyunin_check.py`. For coprime m,n it is

\[
 A_{mn}=\frac{(m+n)(\log(2\pi)-\gamma)
 -(m-n)\log(m/n)-\pi(V(m,n)+V(n,m))}{2mn},
 \quad V(m,n)=\sum_{k=1}^{n-1}\{km/n\}\cot(\pi k/n).
\]

General indices reduce by gcd scaling. The formula is sourced in
[Ehm, equations (17) and (30)](https://arxiv.org/pdf/2405.06349), and also
in [Darses–Hillion, p. 2](https://arxiv.org/pdf/2004.10086). This is not a
new Gram formula. The new C++ implementation uses integer modular residues,
paired cotangents, compensated long-double summation, and cached finite sums.

- The 1024-square prefix differs from the independently integrated earlier
  matrix by at most \(5.773\times10^{-12}\). The 4096 prefix agrees exactly
  between separate new builds at 4096 and 8192.
- Builds took 3.22 s and 23.66 s, respectively, on this machine with four
  OpenMP threads. Timing is machine dependent.
- Twelve Python tests pass: the previous six tests plus comparisons with
  independent piecewise integration, high-precision cotangent sums, ambient
  least squares, the prior screen, and the Mertens/unit-cell identities.
- The pinned Lean project builds successfully. The axiom audit covers 196
  declarations across eight modules and permits only `propext`,
  `Classical.choice`, and `Quot.sound`. Three general projection/transfer
  lemmas were added; equations (1)–(4), (7), and the numerical computations
  are not claimed as Lean formalizations.

The matrix evaluations, projections, gain minima, and tail diagnostics are
**not interval-certified**. Solving normal equations accurately and agreeing
across implementations are checks, not rigorous floating-point error bounds.
The new prime–zero interval certificate belongs to a different matrix family.

## 8. Reproduction

From the repository root, with a C++17/OpenMP compiler, NumPy, SciPy, and
mpmath installed:

```bash
mkdir -p Riemann_Continuation_Verification/residual_capture
g++ -std=c++17 -O3 -fopenmp -shared -fPIC -Wall -Wextra -Werror \
  nyman-beurling/nb_vasyunin.cpp \
  -o Riemann_Continuation_Verification/residual_capture/nb_vasyunin.so
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python nyman-beurling/nb_vasyunin.py \
  --size 8192 \
  --library Riemann_Continuation_Verification/residual_capture/nb_vasyunin.so \
  --output Riemann_Continuation_Verification/residual_capture/gram_8192.npy
OPENBLAS_NUM_THREADS=1 python nyman-beurling/residual_capture.py \
  --gram Riemann_Continuation_Verification/residual_capture/gram_8192.npy \
  --output Riemann_Continuation_Verification/residual_capture/capture_4096.json
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s nyman-beurling -p 'test_*.py' -v
```

`--sizes` accepts a comma-separated list for focused follow-ups. The follow-up
JSON records every checked N in the three neighborhoods specified above.
Matrices, compiled libraries, and generated reports stay in the ignored
directory. Reports record source and matrix hashes. For Lean, run `lake build`
and `lake env lean check_axioms.lean` in `lean/`, preserving its pinned
toolchain and manifest.
