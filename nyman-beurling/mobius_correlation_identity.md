# Noncancellation with the residual fixed: proof and uniformity gap

September 24, 2026. **The uniform residual-gain estimate is not proved.**
This is the next direct attempt at the signed sums in
[the residual-capture investigation](residual_capture_theory.md).
It proves a weaker, unconditional noncancellation statement using a known
L1 convergence theorem, derives the corresponding limit for the original
chain recipe, and identifies why these limits do not establish RH.
No novelty claim is made.

## 1. A signed correlation that cannot disappear when N is fixed

Keep the actual optimal residual

\[
 r_N=\chi-P_N\chi,\quad E_N=\|r_N\|_2^2,\quad
 P_N\chi=\sum_{k\le N}c_ke_k,\quad
 a_N=\sum_{k\le N}c_k/k.
\]

Define the natural Möbius sum, its slope, and its error on (0,1):

\[
 S_M=\sum_{k\le M}\mu(k)e_k,\qquad
 m_M=\sum_{k\le M}\mu(k)/k,\qquad
 \delta_M=\int_0^1|\chi+S_M|\,dt.
\]

The external input is the established unconditional theorem
\(\delta_M\to0\). It is stated in the introduction, p. 2, of
[Báez-Duarte, *A strengthening of the Nyman–Beurling criterion*](https://arxiv.org/pdf/math/0202141),
which attributes it to his 1993 paper. The same passage distinguishes this
L1 convergence on (0,1) from convergence in L2(0,∞); the natural sums do
not converge in the latter norm. We use only the L1 assertion here.

For M ≥ N, let

\[
 C_{N,M}=\sum_{N<k\le M}\mu(k)\langle r_N,e_k\rangle.
\]

**Proposition.** For each fixed finite N,

\[
 \boxed{\lim_{M\to\infty}C_{N,M}=-E_N.}                    \tag{1}
\]

More precisely, with \(\mathcal B_N=1+\sum_{k\le N}|c_k|\),

\[
 |C_{N,M}+E_N|
 \le\mathcal B_N\delta_M+|a_Nm_M|
 \le\left(\mathcal B_N+\frac{|a_N|}{\log2}\right)\delta_M. \tag{2}
\]

**Proof.** Orthogonality to V_N gives
\(C_{N,M}=\langle r_N,S_M\rangle\) and
\(\langle r_N,\chi\rangle=E_N\). Thus

\[
 C_{N,M}+E_N=\langle r_N,\chi+S_M\rangle.                  \tag{3}
\]

On (0,1), \(|r_N|\le\mathcal B_N\), so that portion of (3) is
bounded in absolute value by \(\mathcal B_N\delta_M\). On t > 1,
\(r_N(t)=-a_N/t\) and \(S_M(t)=m_M/t\), so the remaining integral
is exactly \(-a_Nm_M\). Finally, on 1/2 < t < 1,
\(\chi+S_M=m_M/t\), giving \(\delta_M\ge|m_M|\log2\).
This proves (2), and the external L1 theorem proves (1). ∎

In particular, if E_N > 0, then for all sufficiently large M (depending
on N), \(C_{N,M}<-E_N/2\). Thus the contributions cannot cancel their
entire signal in this fixed-N limit. Removing P_N S_M preserves that
correlation and gives a useful finite correction in V_M.

For completeness, E_N is positive for every finite N. An exact finite
representation of χ would have slope a_N = 0 from t > 1, then c_1 = −1
from 1/2 < t < 1. At any prime p > N its synthesis, in u = 1/t
coordinates, would have a nonzero jump contributed by c_1 alone, whereas
χ is constant there. This is impossible.

## 2. What the original chain recipe tends to

Equation (1) concerns full dilation blocks, not just the doubling chains.
For the original family, put

\[
 f_k=e_{2k}-e_k/2,\qquad
 T_M=\sum_{k\le M}\mu(k)f_k=S_M(2t)-S_M(t)/2.
\]

Its L1 limit is \(\chi(t)/2-\chi(2t)\). To justify the change of
scale, the L1 error of S_M on (1,2) is \(|m_M|\log2\). Therefore

\[
 \|T_M-(\chi/2-\chi(2\cdot))\|_{L^1(0,1)}
 \le\delta_M+\tfrac12|m_M|\log2\le\tfrac32\delta_M.
\]

Both T_M and its limit vanish on t > 1. Consequently, for fixed N,

\[
 \lim_{M\to\infty}\langle r_N,T_M\rangle
 =\Lambda_N:=\tfrac12E_N-\langle r_N,\chi(2\cdot)\rangle
 =\frac{1-E_N+c_1}{2}-(\log2)a_N.                         \tag{4}
\]

The error is at most \(\tfrac32\mathcal B_N\delta_M\).
For the final equality, directly integrate e_k on (0,1/2): the answer is
\(b_k-(\log2)/k+\tfrac12 1_{k=1}\), and use
\(\sum c_kb_k=1-E_N\).

When M = N, all terms k ≤ floor(N/2) vanish against r_N by
orthogonality, leaving precisely the Möbius-chain correlation investigated
earlier. **Taking M → ∞ with N fixed does not establish a bound at M=N.**
Also, (4) supplies Λ_N rather than E_N: no uniform lower bound for
\(|\Lambda_N|/E_N\) has been proved. The two recipes must not be conflated.

## 3. The two estimates still needed at the doubling scale

For the full-block recipe define

\[
 g_{N,M}=(I-P_N)S_M,\qquad D_{N,M}=\|g_{N,M}\|^2.
\]

Its optimal scalar gain is \(C_{N,M}^2/D_{N,M}\) when D > 0.
A concrete sufficient pair of statements is, for fixed constants
0 ≤ θ < 1 and B > 0, at every sufficiently large dyadic N,

\[
 |C_{N,2N}+E_N|\le\theta E_N,\qquad
 D_{N,2N}\le B E_N\log N.                                 \tag{5}
\]

They would imply

\[
 E_N-E_{2N}\ge\frac{(1-\theta)^2}{B\log N}E_N,
\]

which is the sufficient gain estimate. **Neither uniform assertion in
(5) is proved here.** They are stronger than necessary; a bound on their
combined gain, or another nonsummable fractional-gain bound, could suffice.

The exact obstruction to using (2) as that proof is its dependence on N.
The established synthesis bound gives only

\[
 \mathcal B_N\le1+6N^{3/2}\sqrt{H_N},\qquad |a_N|\le\sqrt{E_N}.
\]

Indeed, the squared norm of P_Nχ is at most 1, the coefficient bound gives
\(\sum c_k^2\le36N^2H_N\), and the t > 1 residual energy is a_N².
Thus δ_M → 0 for fixed N cannot be substituted into (2) with M=2N to
obtain an error small relative to E_N. No interchange of these limits is
justified. Even choosing a much larger M to secure correlation leaves its
norm cost to control. A uniformly bounded cost along suitable choices of
M would itself be sufficient for convergence, but is not established.

## 4. A rigorous norm-cost obstruction

Let W be the closed span of all e_k, and write

\[
 r_N=r_\infty+s_N,\quad r_\infty=\chi-P_W\chi,\quad
 E_\infty=\|r_\infty\|^2,\quad\|s_N\|^2=E_N-E_\infty.
\]

Every g ∈ W is orthogonal to r_∞. Cauchy–Schwarz therefore gives

\[
 \frac{|\langle r_N,g\rangle|^2}{\|g\|^2}
 \le E_N-E_\infty.                                        \tag{6}
\]

In particular, if \(|\langle r_N,g\rangle|\ge E_N/2\), then

\[
 \boxed{\|g\|^2\ge\frac{E_N^2}{4(E_N-E_\infty)}.}          \tag{7}
\]

The denominator is positive whenever such a g exists. The general
Hilbert-space statements behind (6) and (7) are Lean-checked as
`Refinement.gain_le_visible_energy` and
`Refinement.cost_lower_of_half_capture`, with explicit hypotheses.
The arithmetic limit (1) and its external L1 input are written proofs,
not Lean formalizations.

If E_∞ were positive, (1) would remain valid at every fixed N, but the
cost required for half-capture would diverge as N → ∞. Thus (1) does
not already exclude the very obstruction RH asks us to exclude.

An explicit Hilbert-space example verifies this failure mode. In an
orthonormal basis u_0,u_1,..., take

\[
 x=u_0+\sum_{j\ge1}2^{-j}u_j,\qquad
 V_N=\operatorname{span}(u_1,\ldots,u_N),\qquad
 w_M=-P_Mx-2^M u_M.
\]

For r_N=x−P_Nx and g=(I−P_N)w_M, with M > N,

\[
 E_N=1+4^{-N}/3\longrightarrow1,
\]
\[
 \langle r_N,g\rangle=-1-(4^{-N}-4^{-M})/3\longrightarrow-E_N
 \quad\text{for each fixed N},
\]
\[
 \|g\|^2=4^M+2+(4^{-N}-4^{-M})/3.
\]

Its scalar gain is at most \(16/(9\,4^M)\), despite the nonvanishing
correlation. This is a counterexample to the proposed *logical inference*
from fixed-residual noncancellation to convergence, not a counterexample
to RH or to (5) for the actual Nyman–Beurling family.

## 5. Numerical checks and reproduction

The probe evaluates 24 (N,M) pairs using the existing 8192 Gram matrix,
independently checks finite identity (3), and keeps correlation and cost
separate. Selected full-block results follow. All are floating point,
not interval certificates.

| N | M | −C/E | D/(E log N) | log(N) C²/(E D) |
|---:|---:|---:|---:|---:|
| 32 | 64 | 0.671466 | 0.811576 | 0.555544 |
| 32 | 1024 | 0.992519 | 0.994727 | 0.990317 |
| 32 | 8192 | 1.009528 | 2.610867 | 0.390348 |
| 1024 | 2048 | 0.447937 | 0.786539 | 0.255102 |
| 1024 | 8192 | 0.976749 | 1.905096 | 0.500782 |
| 4096 | 8192 | 0.418955 | 0.811712 | 0.216239 |

At fixed N=32, near-complete correlation at M=8192 produces less gain
than at M=1024 because the norm cost increased. At the largest available
doubling step, only about 42% of the fixed-N limiting correlation is
captured. Neither observation determines the asymptotic behavior.

Three additional tests check the projected correlation and norm using
independent periodic quadrature, the chain-limit formula using direct
target integration, and the explicit Hilbert-space counterexample using
orthonormal coordinates. Together with the previous tests, all 15 pass.
The pinned Lean build succeeds, and the axiom audit passes for 198 declarations
across eight modules, permitting only `propext`, `Classical.choice`, and
`Quot.sound`.

```bash
OPENBLAS_NUM_THREADS=1 python nyman-beurling/mobius_correlation_probe.py \
  --gram Riemann_Continuation_Verification/residual_capture/gram_8192.npy \
  --output Riemann_Continuation_Verification/residual_capture/fixed_residual_correlation.json
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s nyman-beurling -p 'test_*.py' -v
```

The report records source/matrix hashes and stays ignored. In `lean/`,
run `lake build` and `lake env lean check_axioms.lean` with the pinned
toolchain and manifest.

**Proof status:** noncancellation is established with N fixed and M → ∞,
using a classical unconditional L1 theorem. Uniform control of correlation
relative to its correction cost as N grows remains missing. This attempt
does not prove the requested uniform bound or RH.
