# The hypothetical surviving remainder: arithmetic constraints and rigidity

September 24, 2026. **The unrestricted rigidity theorem, the uniform gain
estimate, and RH remain unproved.** This investigation changes the object of
study to the canonical remainder invisible to every dilation. It derives an
exact sequence-space description, its dilation constraints, and a rigidity
theorem with an additional summability hypothesis. It also identifies where
the attempted unconditional argument fails. No novelty claim is made.

The subsequent [decay investigation](autocorrelation_decay.md) sharpens the
autocorrelation limitation: the identity follows automatically from projection,
and explicit normalized L² controls satisfy it while their primitives fail
absolute summability. Their arithmetic defects are computed exactly; no
unconditional example satisfying all the arithmetic equations is claimed.

## 1. Assume the canonical remainder is nonzero

In real H = L²(0,∞), let

\[
 e_k(t)=\{1/(kt)\},\quad \chi=1_{(0,1]},\quad
 W=\overline{\operatorname{span}\{e_k:k\ge1\}},\quad
 h=\chi-P_W\chi,\quad E=\|h\|^2.
\]

The hypothesis to contradict is E > 0. Projection immediately gives

\[
 \langle h,e_k\rangle=0\ (k\ge1),\qquad
 \chi-h\in W,\qquad \langle h,\chi\rangle=E.                \tag{1}
\]

The last two statements distinguish the canonical remainder from an arbitrary
annihilating function. The finite residuals converge to h in H because their
projection spaces increase to W. None of this establishes E = 0.

The norm bound E ≤ 1 and all finite orthogonality equations are compatible
with E > 0. An unconditional contradiction must exploit more structure.

## 2. Orthogonality becomes a residue-class system

Let

\[
 I_j=(1/(j+1),1/j),\quad w_j=|I_j|=1/[j(j+1)],\quad
 d_j=\int_{I_j}h(t)\,dt,\quad F_n=\int_0^{1/n}h(t)\,dt.
\]

Then

\[
 F_n=\sum_{j\ge n}d_j,\quad d_j=F_j-F_{j+1},\quad
 F_1=E,\quad \sum_{j\ge1}|d_j|<\infty,\quad
 \sum_{j\ge1}j(j+1)|d_j|^2\le E.                           \tag{2}
\]

The last inequality is Cauchy–Schwarz on disjoint intervals. Also
\(F_n=o(n^{-1/2})\), since
\(\sqrt n|F_n|\le\|h\|_{L^2(0,1/n)}\to0\).

For every positive integer n, the fractional-part identity is

\[
 e_n(t)-e_1(t)/n=\frac{j\bmod n}{n}\quad(t\in I_j),
 \qquad e_n(t)-e_1(t)/n=0\quad(t>1).                       \tag{3}
\]

Consequently (1) forces the absolutely convergent equations

\[
 \boxed{\sum_{j\ge1}(j\bmod n)d_j=0\quad(n\ge2).}          \tag{4}
\]

The generalized fractional-part identity and its actual dilation version
are Lean-checked as `Annihilator.fract_div_nat` and
`Annihilator.dilation_residue`.

Finite summation by parts gives, with every boundary term retained,

\[
 \sum_{j=1}^L(j\bmod n)d_j
 =\sum_{j=1}^L F_j-n\sum_{k\le L/n}F_{nk}
  -(L\bmod n)F_{L+1}.                                    \tag{5}
\]

The general finite summation-by-parts identity is Lean-checked as
`Annihilator.finite_abel_boundary`. The final boundary term tends to zero
for fixed n, since its multiplier is at most n−1. Thus

\[
 \boxed{\lim_{L\to\infty}
 \left(\sum_{j\le L}F_j-n\sum_{k\le L/n}F_{nk}\right)=0.}   \tag{6}
\]

Equation (6) is a limit of a difference. It does **not** assert that either
series separately converges.

## 3. This sequence model is exact, including the original tail

Let S₀ be the closed subspace of functions constant on each I_j and zero
on t > 1. Identify it with sequences q_j of squared norm
\(\sum w_j|q_j|^2\). Put

\[
 \gamma_n(j)=\{j/n\},\qquad
 U=\overline{\operatorname{span}\{\gamma_n:n\ge2\}}\subset S_0,
 \qquad Q=U^\perp\cap S_0.
\]

Then q ∈ Q means exactly (4), with d_j=w_jq_j. This is the sequence
family in [Bagchi, Theorem 1](https://arxiv.org/pdf/math/0607733): density
of these periodic fractional-part sequences in the weighted space is
equivalent to RH. Rediscovering these equations is not a new RH criterion.

The full space used by this repository also contains the non-step component
of e₁. Its orthogonal decomposition is explicit:

\[
 v=P_{S_0}e_1,\quad
 v_j=j(j+1)\log(1+1/j)-j,\quad
 z=e_1-v,\quad\kappa=\|z\|^2\ge1.
\]

The inequality follows from z(t)=1/t on t > 1. Equation (3) proves

\[
 W=U+\operatorname{span}\{e_1\}\subset
 S:=S_0\mathbin\oplus\operatorname{span}\{z\}.              \tag{7}
\]

This sum is closed: the z component determines the coefficient of e₁
continuously, after which closedness of U applies. Since χ ∈ S, the
canonical remainder belongs to S. Every element of S orthogonal to W
has exactly the form

\[
 h=q-\frac{\langle q,v\rangle}{\kappa}z,\qquad q\in Q.     \tag{8}
\]

To verify (8), orthogonality to all γ_n forces q ∈ Q; orthogonality to
e₁=v+z fixes the coefficient of z. Conversely, these conditions annihilate
every e_n. In particular h has one common affine slope in u=1/t on
every unit cell and a constant multiple of 1/t on t > 1. This local
regularity still permits unbounded growth as t → 0.

For the canonical remainder specifically, write χ_Q=P_Qχ and v_Q=P_Qv.
The condition χ−h ∈ W yields

\[
 q+\frac{\langle q,v_Q\rangle}{\kappa}v_Q=\chi_Q,
\quad
 q=\chi_Q-\frac{\langle\chi_Q,v_Q\rangle}
                   {\kappa+\|v_Q\|^2}v_Q,                 \tag{9}
\]
\[
 E=\|\chi_Q\|^2-
       \frac{|\langle\chi_Q,v_Q\rangle|^2}
            {\kappa+\|v_Q\|^2}.                           \tag{10}
\]

Thus the full-tail problem is an explicit rank-one modification of the
step-space problem. Dropping the tail without this adjustment would change
the projection. These are written Hilbert-space identities, not additional
Lean formalizations.

## 4. Integer dilation, its adjoint, and an extra canonical constraint

Set D_n f(t)=f(nt). Its norm is n^(−1/2), and
D_n e_k=e_nk, so D_n W ⊆ W. Likewise

\[
 D_n\gamma_k=\gamma_{nk}-\gamma_n/k,
\]

so U is invariant under D_n on S₀. In sequence coordinates,
(D_n q)_j=q_floor(j/n), taking q₀=0. Its adjoint on S₀ is

\[
 (D_n^*q)_k=\frac1{w_k}
          \sum_{j=nk}^{n(k+1)-1}w_jq_j.                   \tag{11}
\]

Consequently Q is invariant under this adjoint. On primitives, the same
operation is simply

\[
 F_k\longmapsto F_{nk}.                                   \tag{12}
\]

This is the explicit link between geometric dilation and arithmetic
subsequences. It is invariance in the indicated direction; invariance under
both an operator and its adjoint has not been established.

There is a further condition for the **canonical** h. Because χ−h ∈ W
and D_nW ⊆ W,

\[
 \boxed{F_n=\langle h,D_n\chi\rangle
              =\langle h,D_n h\rangle,\qquad
          |F_n|\le E/\sqrt n.}                            \tag{13}
\]

Thus these arithmetic primitives are also dilation autocorrelations.
Under the unitary change of variable a(y)=e^(y/2)h(e^y), let
C(x)=∫a(y)a(y+x)dy. Then

\[
 F_n=n^{-1/2}C(\log n),\quad C(0)=E,
\quad [C(x_i-x_j)]_{i,j}\succeq0.                         \tag{14}
\]

This positivity is proved by writing the quadratic form as the squared
norm of a finite sum of translated copies of a. It imposes more than an
arbitrary choice of signed d_j. Nevertheless, positivity alone does not
force absolute summability: C(x)=exp(−a|x|), with 0<a<1/2, is a
positive-definite autocorrelation and yields F_n=n^(−1/2−a), not in ℓ¹.
This example does not satisfy the residue equations; it only invalidates
an inference from positivity alone. No contradiction from (4) together
with the canonical constraints has been proved here.

The follow-up [exact controls](autocorrelation_decay.md) strengthen this test:
the same function supplies both the primitive and its self-correlation, including
the target/norm normalization. Their failure of the arithmetic equations is a
zeta factor. The correlation identity is a consequence of canonical projection,
not an independent regularity theorem.

For finite residuals the exact corresponding identity has a defect:

\[
 F_N(n)-\langle r_N,D_nr_N\rangle
 =\langle r_N,D_nP_N\chi\rangle.                          \tag{15}
\]

It cannot be set to zero at finite N, since D_nV_N generally extends beyond
V_N. The probe retains and checks this term. For each fixed n it tends
to zero as N → ∞ by strong convergence of the projections.

## 5. Rigidity with an explicit summability hypothesis

**Theorem.** Suppose d ∈ ℓ¹ satisfies (4), and set
F_n=Σ_(j≥n)d_j. If additionally

\[
 \boxed{\sum_{n\ge1}|F_n|<\infty,}                         \tag{16}
\]

then F_n=0 and d_n=0 for every n. For the canonical remainder, (16)
therefore forces E=F₁=0 and h=0.

**Proof.** Absolute summability now permits taking the two limits in (6)
separately. With A=Σ_jF_j this gives

\[
 \sum_{k\ge1}F_{nk}=A/n\qquad(n\ge1).                     \tag{17}
\]

For y≥2, let P_y be the product of all primes ≤y. Finite
inclusion-exclusion, followed only by absolutely convergent sums, gives

\[
 \sum_{\substack{m\ge1\\(m,P_y)=1}}F_{nm}
 =\sum_{d\mid P_y}\mu(d)\sum_{k\ge1}F_{ndk}
 =\frac A n\prod_{p\le y}(1-1/p).                         \tag{18}
\]

On the left, dominated convergence gives F_n as y→∞: every fixed
m>1 is eventually excluded. On the right, the product tends to zero.
For an elementary justification, its reciprocal expands as a positive
sum over integers with prime factors ≤y and is at least
Σ_(m≤floor(y))1/m, which diverges. Hence F_n=0. ∎

The same calculation supplies the quantitative bound

\[
 |F_n|\le\frac{|A|}{n}\prod_{p\le y}(1-1/p)
                +\sum_{m>y}|F_{nm}|.                      \tag{19}
\]

This is a written conditional rigidity theorem with a full proof. The
finite sieve identity is tested with exact rational arithmetic. **Condition
(16) has not been derived for the hypothetical remainder.** Square-integrable
h gives (2), not (16); for example h(t)=t^(−1/4) on (0,1) is in L² but
its primitive F_n=(4/3)n^(−3/4) is not summable. That example is not an
annihilator, so it does not disprove a possible regularity theorem using
the full arithmetic constraints.

There is also a different sufficient regularity condition: if the canonical
h is bounded on (0,1), the earlier natural-Möbius L1 convergence argument
forces ⟨h,χ⟩=0. The tail is a constant multiple of 1/t by (8), so its
contribution tends to zero as well. A nonzero canonical remainder must
therefore be unbounded near zero and fail (16). Neither failure contradicts
L² membership.

## 6. Power modes expose the exact analytic threshold

Temporarily complexify the sequence space; testing against the real residue
generators uses the integral of the complex sequence against those generators.
For a complex s with Re(s)>0, s≠1, define

\[
 F_n^{(s)}=n^{-s}/s,\qquad
 d_j^{(s)}=[j^{-s}-(j+1)^{-s}]/s.
\]

These satisfy d ∈ ℓ¹. Abel's identity and the Euler-summation asymptotic
Σ_(j≤L)j^(−s)=L^(1−s)/(1−s)+ζ(s)+o(1) give

\[
 \boxed{\sum_{j\ge1}(j\bmod n)d_j^{(s)}
         =\frac{\zeta(s)}s(1-n^{1-s}).}                   \tag{20}
\]

The leading powers of L cancel between the two finite sums in (5).
Equation (20) also follows by grouping in residue classes and taking
differences of Hurwitz zeta functions. That second expression is checked
independently in the probe.

Since d_j^(s)=j^(−s−1)(1+O_s(1/j)),

\[
 \sum j(j+1)|d_j^{(s)}|^2<\infty
 \quad\Longleftrightarrow\quad \operatorname{Re}s>1/2.     \tag{21}
\]

Thus a hypothetical zero with 1/2<Re(s)<1 supplies an actual nonzero
vector of Q via q_j=j(j+1)d_j^(s). Formula (8) then supplies an
annihilator of the **full** dilation family with target pairing 1/s≠0.
This is conditional on such a zero; no such zero is exhibited or assumed
in the purported proof. It explicitly shows how an off-line zero would
survive all the residue and Hilbert-space constraints. It also fails (16).

At a critical-line zero, the residue equations still hold but the weighted
norm diverges logarithmically. Each dyadic band's squared norm tends to
log 2. Hence a mode on the critical line must not be passed off as an L²
counterexample. These are direct consequences of (20)–(21), not a
classification of every vector in Q into power modes.

The numerical check uses the first critical-line zero from mpmath and
the separate test point 0.75+i·Im(ρ₁). The latter is **not** a zeta zero:
|ζ|≈0.179513. It is a control that has finite weighted norm but fails the
residue equations. At the critical-line point the weighted energy on
[8192,16384) is approximately 0.693147087, close to log 2, as predicted.
The sampled zero and these values are floating-point checks, not zero
certificates.

## 7. What has and has not been achieved

The surviving-error hypothesis now has an exact arithmetic description:
the weighted cell sequence, all residue constraints, a finite-boundary
identity, adjoint decimation, and the additional autocorrelation constraint
for the canonical remainder. Finite-prime Möbius inversion kills the
remainder under (16). The attempt to remove (16) fails at a specific
operation: splitting the convergent difference in (6) into potentially
divergent sums and applying infinite inversion.

Related literature already studies annihilators under extra regularity:
[Calderaro–Manzur–Noor–Santos, Theorem 2](https://arxiv.org/pdf/2203.05030)
records trivial intersection with a local Dirichlet space in a different
Hardy-space realization. Its hypothesis must be established before use.
[Xue's operator formulation](https://arxiv.org/pdf/1911.04029) likewise
distinguishes dilation invariance from additional adjoint invariance; the
latter is an RH-equivalent condition in that model. Neither result closes
our gap, and no unproved equivalence between their regularity conditions
and (16) is asserted.

The remaining research question is whether the **canonical** conditions
force enough regularity for a valid inversion, or provide a different
rigidity argument at the weighted-L² threshold. Simply declaring infinite
Möbius inversion valid, or treating the positive kernel in (14) as strictly
positive on every relevant direction, would assume the missing conclusion.

## 8. Verification and reproduction

`Annihilator.lean` checks the generalized residue identity, its dilation
version, and the finite summation-by-parts boundary. The new module is
included in the default build and project-wide axiom audit. The projection
description, analytic dilation application, infinite summability theorem, and
zeta-mode argument above are written proofs, not Lean formalizations. The
subsequent decay investigation formalizes the general projection identity
behind the autocorrelation deduction and the nonzero-witness implication.

In this stage, seven new tests covered exact fractional-part arithmetic, independent cell
quadrature, the retained Abel boundary, rational prime inclusion-exclusion,
adjoint decimation, independent dilation-covariance quadrature, and the
power-mode formula/critical norm threshold. The NB suite then had 22 tests.
All passed. The pinned Lean build succeeded, and the audit passed for 201
declarations across nine modules, permitting only `propext`,
`Classical.choice`, and `Quot.sound`. See the follow-up for current counts.

The report checks five finite optimal residuals through N=4096 and 30
residue constraints, keeping their truncation tails. It is not a numerical
construction of h. Matrix operations and tail diagnostics are floating
point, not interval-certified.

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s nyman-beurling -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python nyman-beurling/annihilator_probe.py \
  --gram Riemann_Continuation_Verification/residual_capture/gram_8192.npy \
  --output Riemann_Continuation_Verification/residual_capture/annihilator_checks.json
```

In `lean/`, run `lake build` and `lake env lean check_axioms.lean` with the
pinned toolchain and manifest. Generated reports remain ignored and record
the source and matrix hashes.
