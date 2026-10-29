# Residual reduction: the next proof target

September 10, 2026. This note distinguishes exact identities, a machine-checked
unconditional real coefficient bound, finite experiments, and the still
unproved estimate that would imply RH. Numerical outputs are floating-point
diagnostics, not interval certificates.

## 1. The quantity that must decrease

Work in real L²(0,∞), with e_k(t) = fract(1/(kt)) and χ = 1_(0,1]. Let

    V_N = span(e_1,...,e_N),  r_N = χ − P_N χ,
    E_N = ‖r_N‖² = 1 − b_Nᵀ A_N⁻¹ b_N.

The use of real coefficients loses nothing for this real target and family.
The integer-dilation criterion says E_N → 0 is equivalent to RH
([Báez-Duarte, Theorem 1](https://arxiv.org/pdf/math/0202141)).
Positive definiteness of each A_N ensures finite projections exist; its
smallest eigenvalue alone says nothing about the remaining error.

For the block decomposition

    A_2N = [ A  B ] ,    b_2N = [ b ] ,
           [ Bᵀ D ]            [ d ]

put S = D − Bᵀ A⁻¹ B and g = d − Bᵀ A⁻¹ b. Completing the quadratic
form, or eliminating the old coefficients, gives the exact identity

    E_N − E_2N = gᵀ S⁻¹ g.

This is where the missing information resides: correlations of the actual
residual with the new directions. A lower bound on S controls the cost of
coefficients; it does not provide a lower bound on this gain.

## 2. A simpler explicit correction

Use parents k = floor(N/2)+1,...,N and chains f_k = e_2k − e_k/2.
Every child 2k is new, and every parent is already available. Define

    M_kl = ⟨f_k,f_l⟩,
    V_ik = ⟨e_i,f_k⟩,
    h_k = ⟨χ,f_k⟩ = log(2)/(2k),
    c = A_N⁻¹ b_N,    q = h − Vᵀc.

All these entries are explicit combinations of the original Gram kernel.
For example,

    M_kl = A_2k,2l − A_2k,l/2 − A_k,2l/2 + A_k,l/4.

With the old approximation fixed, add Σ α_k f_k. Its error is exactly

    E_N − 2αᵀq + αᵀMα.

Taking α = M⁻¹q gives the direct gain G_N = qᵀM⁻¹q. If the old
coefficients are refitted as well, the gain is

    G_N^proj = qᵀ(M − VᵀA_N⁻¹V)⁻¹q.

The projected matrix is the principal submatrix of S indexed by the new
even dilations. Thus, unconditionally,

    0 ≤ G_N ≤ G_N^proj ≤ E_N − E_2N.

M and the projected matrix are positive definite because the relevant
families are independent. The jump argument below supplies that independence.
No zeros, Möbius asymptotics, or RH enter these finite identities.

## 3. Separate the location of the error from its capture

In u = 1/t coordinates,

    R_N(u) = 1_[1,∞)(u) − Σ c_k fract(u/k),
    E_N = ∫₀∞ R_N(u)² du/u²,
    f_k(1/u) = (floor(u/k) mod 2)/2.

The new chains vanish for u < K = floor(N/2)+1. Let

    T_N = ∫_K∞ R_N(u)² du/u².

Restricting the residual to this tail does not change its correlations
with the chains. Orthogonal projection onto their span therefore gives
G_N ≤ T_N, and

    log(N) G_N/E_N = [log(N) T_N/E_N] [G_N/T_N].

The first factor measures available tail energy. The second is the
fraction of that energy removed by the chains. This isolates two possible
failures of a uniform bound; measuring total gain alone does not distinguish
them. The identity is used only when E_N and T_N are positive; an exact
zero residual has already solved the approximation problem at that stage.

The head integral is elementary. Set a = Σ c_k/k and
b_j = 1 + Σ c_k floor(j/k). On j < u < j+1, j ≥ 1,

    R_N(u) = b_j − au,
    ∫_j^(j+1) R_N(u)² du/u²
      = b_j²/[j(j+1)] − 2ab_j log(1+1/j) + a².

The contribution from 0 < u < 1 is a². This formula is evaluated in
long double, and tested against independent piecewise quadrature. It is
an exact formula evaluated approximately, not an exact numerical enclosure.

## 4. The unproved lemma, and a weaker sufficient version

One concrete sufficient target is: there are c > 0 and N₀ such that,
for every dyadic N ≥ N₀,

    q_Nᵀ M_N⁻¹ q_N ≥ (c/log N) E_N.                 [OPEN]

For example, uniform lower bounds log(N)T_N/E_N ≥ a > 0 and
G_N/T_N ≥ b > 0 would imply this with c = ab. Neither bound is proved.
The pair is stronger than necessary: their product is what matters.

There is no need to demand that particular rate. Put F_j = E_(2^j).
If a sequence a_j gives

    F_(j+1) ≤ (1−a_j) F_j,    Σ_j a_j = ∞,

then 1−x ≤ exp(−x) implies

    0 ≤ F_m ≤ F_0 exp(−Σ_(j<m) a_j) → 0.

Starting after a finite initial segment is handled by reindexing. Monotonicity
of E_N then gives convergence for all N. The sufficient analytic implication
is formalized in `lean/RHCrystal/Refinement.lean`; the hypotheses asserting
gain and divergence remain explicit. The library does not formalize the
external RH-equivalence theorem or establish those hypotheses for E_N.

For an individual direction v in coefficient space one can seek the even
more explicit sufficient inequality

    (qᵀv)² ≥ (c/log N) E_N (vᵀMv).

Optimizing the scalar multiple of v then proves the desired gain. Choosing
v by an arithmetic rule would remove the inverse from the proof target.
The scalar square-completion identity is also machine checked.

## 5. An unconditional jump bound: full written argument

For N ≥ 1, arbitrary complex coefficients c_k, and

    F(u) = Σ_(k≤N) c_k fract(u/k),
    Q = ∫₀∞ |F(u)|² du/u²,
    s_n = Σ_(k|n) c_k,    H_N = Σ_(d=1)^N 1/d,

one has

    Q ≥ (Σ_(k=1)^N |c_k|²)/(36 N² H_N).

The integral is finite: F(u) = au on 0 < u < 1, and F is bounded
on u ≥ 1. All jumps occur at integers. For 1 ≤ n ≤ N, on the cell
(n−1/2,n+1/2), apart from its centre,

    F(n+v) = b + av − s_n 1_(v>0),
    a = Σ c_k/k.

Direct integration and square completion give

    ∫_(-1/2)^(1/2) |b+av−s 1_(v>0)|² dv
      = |b−s/2|² + |a−3s/2|²/12 + |s|²/16
      ≥ |s|²/16.

On that cell u ≤ n+1/2 ≤ 3n/2. Therefore its weighted energy is at
least |s_n|²/(36n²). The cells are disjoint up to endpoints, so

    Σ_(n≤N) |s_n/n|² ≤ 36Q.

Let y_d = s_d/d. Möbius inversion gives

    c_n = Σ_(d|n) μ(n/d) d y_d.

The matrix M_nd = 1_(d|n) μ(n/d)d has absolute column sum
at most d floor(N/d) ≤ N. Its absolute row sum is at most

    Σ_(d|n) d = n Σ_(d|n) 1/d ≤ N H_N.

Weighted Cauchy–Schwarz followed by interchange of the finite sums
(the Schur test) yields

    Σ |c_n|² ≤ N²H_N Σ |y_d|² ≤ 36 N²H_N Q.

This proves the bound. Substitution u = 1/t identifies Q with
‖Σ c_k e_k‖², and hence gives the same lower bound for λ_min(A_N).
The sharper factor P_N = max_(n≤N) Σ_(d|n) |μ(d)|/d may replace H_N:
the exact absolute row sum is n Σ_(d|n)|μ(d)|/d. The formal arithmetic
module currently uses the simpler harmonic version.

Formalization status: the real integral bound is machine verified end to end
by `NymanBeurling.Jump.synthesis_integral_lower_bound` in `JumpSynthesis.lean`.
Its only hypothesis is N > 0, for an arbitrary real coefficient function.
`JumpLocal.lean` supplies the local integral; `JumpArithmetic.lean` supplies
Möbius recovery; `JumpSynthesis.lean` proves the actual fractional-part jump
identities, integrability, change of variables, weighted cell sum, and final
bound. The sharper P_N constant, complex extension, and named eigenvalue
corollary are written consequences, not separate end-to-end Lean theorems here.

## 6. Reproduced evidence and a control that does not converge

The current code independently checks the block solution against the full
projection, the chain solution against a subset projection, and the direct
correction by evaluating its actual 2N-coefficient vector. Ambient-space QR
supplies a separate check on synthetic data. Cholesky solves are used;
there is no spectral cutoff or matrix inverse.

| N → 2N | E_2N | log(N)G_N/E_N | projected share of full gain | log(N)T_N/E_N | G_N/T_N |
|---:|---:|---:|---:|---:|---:|
| 32 → 64 | 0.011376040 | 0.273187 | 62.95% | 2.366274 | 11.55% |
| 64 → 128 | 0.009658549 | 0.344379 | 76.21% | 2.462517 | 13.98% |
| 128 → 256 | 0.008233716 | 0.442684 | 84.82% | 2.596786 | 17.05% |
| 250 → 500 | 0.007419858 | 0.341641 | 86.99% | 2.499020 | 13.67% |
| 256 → 512 | 0.007386508 | 0.361779 | 88.04% | 2.495438 | 14.50% |
| 384 → 768 | 0.006719444 | 0.518838 | 93.93% | 2.590607 | 20.03% |
| 512 → 1024 | 0.006528720 | 0.444187 | 95.03% | 2.474954 | 17.95% |

The smallest tenth of full Schur eigenmodes account for only 0.287% of
the gain at N=512. These sampled values motivate the direct-chain target;
they do not give a uniform lower bound, monotonicity of capture, or an
asymptotic law.

As a control, in H² approximate 1 by (1−az)z^j, j=0,...,m−1.
The Gram matrix has diagonal 1+a² and adjacent entries −a. Solving its
three-term recurrence gives the exact error

    E_m = 1/(m+1)                            if a = 1,
    E_m = (a²−1)/(a²−a^(−2m))               if a > 1.

For a=2 the smallest eigenvalue is at least 1, but E_m → 3/4;
the fractional gains vanish. Both controls are included so that a finite
positive spectral floor cannot be mistaken for a convergence mechanism.

Reproduction, from `nyman-beurling/`:

```bash
python -m unittest test_block_refinement -v
python refinement_scan.py --build 1024 \
  --gram ../Riemann_Continuation_Verification/nb_gram_refinement_1024.npy \
  --sizes 32 64 128 250 256 384 512 \
  --output ../Riemann_Continuation_Verification/refinement_1024.json
```

Omit `--build 1024` when the matrix already exists. Building the 1024
matrix took 579 seconds on this CPU run. JSON reports record source hashes,
matrix hash, library versions, and base commit. Generated files stay ignored.

## 7. What the new prime–zero files contribute

The supplied Suzuki-function script was extracted and rerun unchanged.
All 64 floating fields reproduce within 6.29e−13; its six 55-digit
reference strings are identical. Five prime-derived peaks, six positive
sampled kernel matrices, and the 2.55e−14 precision comparison reproduce.
These checks validate a finite computation, not global positivity.

For this particular Ψ, Suzuki proves that global kernel positivity,
pointwise Ψ(t) ≥ 0 for all real t, and boundedness of Ψ are each equivalent
to RH (Theorems 1.2, 1.6, 1.7 of
[the original paper](https://arxiv.org/pdf/2206.03682)). Pointwise positivity
is thus a second precise target. It is not a general rule that a kernel's
positive diagonal makes it positive semidefinite.

The supplied separated-bump argument correctly obstructs the proposed
pointwise-positive heat flow, under its stated Weil-form/operator setup.
It does not show that the Weil quadratic form has negative values.
The cited current [spectral-triples PDF](https://alainconnes.org/wp-content/uploads/zeta-spectral-triples-1.pdf)
puts its convergence discussion in §7, “Outlook”; the handoff says §8.

No transformation identifying that indefinite candidate form with the
positive Nyman–Beurling Gram form has been established. A shared appearance
of primes or zeros does not supply one. Continue the explicit residual
correlation estimate above; use the new object as a separate route until a
proved analytic identity connects their missing estimates.

## 8. Shared-conversation finite extrema (September 23, 2026)

The [reviewed addition](../papers/gram_extremum_review.md) establishes a sufficient
integrated-tail dominance criterion for a finite prime–zero Gram condition-number
minimum and reproduces its interval certificate for 20 supplied zero brackets
and 120 primes. It also verifies that the unrestricted minimum claim is false.
The completed-xi vertical-energy minimum is another unconditional statement,
but a symmetric entire counterexample shows that this property does not locate
zeros. Neither statement gives the signed-correlation estimate for the optimized
residual required in this note. They are separate, verified finite/analytic
results, with the uniform residual-gain step still open.

## 9. Actual Möbius residual capture (September 23, 2026)

[The follow-up derivation](residual_capture_theory.md) writes the correlation
against the actual optimal residual as a convergent sum of explicit unit-cell
integrals and Möbius partial sums. Removing the old-span component improves
every correction's scalar gain; the general statement, trial-transfer bound,
and trial excess-energy identity are now Lean verified. Finite experiments
extend to N=4096 → 8192 and expose further cancellation and smaller observed
gain minima. The current Cauchy–Schwarz transfer from the tapered trial gives
no positive lower bound in any tested case. The remaining target is the
uniform signed-correlation estimate in equation (8) of the follow-up note.
