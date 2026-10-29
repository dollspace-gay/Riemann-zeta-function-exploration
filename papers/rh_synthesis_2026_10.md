# From cluster resolution to the signature of Weil's form

October 28, 2026. Research synthesis and reproducible computation.
Repository baseline: `90e0944c6d693bd57d7fa44967f18eddded7a6fe`.

**Result of this session:** the project's cluster-resolution analysis can
be transferred to a finite compression of the signed Weil form. A
resolved on-line pair has a small positive derivative channel; a
reflected off-line pair has a negative one, with the same quadratic
resolution scale. We derive this exactly, exercise it numerically, and
implement a separate arithmetic-side reconstruction of a zeta Weil
compression. A zero-free-strip input also improves an explicit coarse
tail envelope. These are a usable bridge to the recent RH work, not a
proof of RH. No novelty claim is made for the elementary identities.

## 1. The inputs and their status

| Input | What is available | How this session uses it |
|---|---|---|
| This repository's zero Gram matrices | Cluster localization and divided-difference resolution; symmetry and pair bounds formalized | Transfer the resolution mechanism, not a presumed spectral equivalence |
| Nyman–Beurling branch | RH-equivalent approximation distance, chain bounds, trial-function spectroscopy | Preserve as the approximation-side program |
| `kfloor.md`, Session 27 | Primorial construction refutes the proposed polylogarithmic GCD floor | Replay the counterexample; do not revive Lemma (A) or the old K-floor |
| Alpöge–Furman, arXiv:2608.13637 | The 0.67250… simple-on-line proportion via a compression of Weil's form | Use the signed block structure and its explicit-formula normalization |
| OpenAI's quasi-RH manuscripts | Published zero-free-half-plane results at 7/8 and, in a separate human-assisted manuscript, 11/12 | Treat these as named external inputs in a conditional tail lemma; no local proof rebuild is claimed |

The proportion theorem and the zero-free strip control different things.
A positive asymptotic proportion does not exclude a sparse set of
off-line zeros, and a strip bound does not move the remaining zeros to
the line. Even asymptotic proportion one could leave finitely many
exceptions. Neither input supplies an inverse-derivative moment bound
for the `1/|zeta'(rho)|` weights in `amplitude_theory.md`.

## 2. Keep the three operators distinct

The original prime-zero Gram matrix is of the form `V* V` (or its real
cosine/sine realization). It is positive semidefinite for any input
locations. Inserting hypothetical off-line locations into a Hermitian
outer product preserves positivity. A condition-number change alone
cannot turn this into a positivity criterion for RH.

The Nyman–Beurling matrix is another positive Gram matrix, `A_N`, with
`b_k = <chi,e_k>`. The target is

    d_N^2 = ||chi||^2 - b* A_N^{-1} b -> 0.

Its smallest eigenvalue is only part of that expression: the projections
of `b` onto the eigenvectors also matter. A lower bound on the spectral
floor is not an upper bound tending to zero for the approximation
distance.

The new object is Weil's form. Use the Fourier convention

    F(z) = integral phi(u) exp(-i u z) du,
    z_rho = (rho - 1/2)/i = gamma - i(beta-1/2).

For probe transforms `F_j`, its compression has entries

    W_ij = sum_rho m_rho F_i(z_rho) conjugate(F_j(conjugate(z_rho))).

This form can be indefinite. Its positivity on the entire admissible
test space is an RH criterion. Positivity on one finite probe family
does not settle that criterion.

## 3. Exact pair/cluster bridge — derived here

Take `F` real entire and even, and `F_j(z)=F(z-t_j)` for real probe
centers. For an off-line zero `rho=1/2+delta+i gamma`, its reflected partner
`1-conjugate(rho)` has spectral coordinate conjugate to `z_rho`.
The two off-line zeros have the same multiplicity `m >= 1`.
Writing the probe vector as `v=a+i b`, their contribution is

    Q = m [vv^T + conjugate(vv^T)] = 2m aa^T - 2m bb^T.

Its restriction has one positive and one negative direction when `a,b`
are independent. Under projection they need not remain independent.
Replacing the transpose by a conjugate transpose instead gives

    H = m [vv* + conjugate(v)v^T] = 2m aa^T + 2m bb^T,

which erases the signature. The distinction is structural, not numerical.

For two symmetric probes `t_1=gamma-h`, `t_2=gamma+h`, the off-line pair
has eigenvalues exactly

    4m Re(F(h+i delta))^2,    -4m Im(F(h+i delta))^2.

For two **simple on-line zeros** at `gamma-epsilon`, `gamma+epsilon`, the two
eigenvalues are exactly

    [F(h+epsilon)+F(h-epsilon)]^2,
    [F(h+epsilon)-F(h-epsilon)]^2.

If both on-line zeros have a common multiplicity `m`, multiply both
eigenvalues by `m`. For unequal multiplicities `m_-`, `m_+`, use the
weighted matrix

    C = m_- u_- u_-^T + m_+ u_+ u_+^T,
    u_- = (F(h-epsilon), F(h+epsilon))^T,
    u_+ = (F(h+epsilon), F(h-epsilon))^T.

Its sum and difference vectors need not be eigenvectors.

For the simple on-line cluster and the off-line pair of fixed multiplicity
`m`, the channels that tend to zero satisfy

    lambda_cluster = +4 epsilon^2 F'(h)^2 + O(epsilon^4),
    lambda_offline = -4m delta^2 F'(h)^2 + O(delta^4).

Since `m > 0`, multiplicity preserves the off-line signature and the
quadratic resolution scale.

These are labeled channels; near zero they are the small eigenvalues
if `F(h)` and `F'(h)` are nonzero. This is the direct connection to the
project's divided-difference picture: resolution controls both the
positive cluster degeneracy and visibility of negative pair directions.

An additional exact two-probe identity, without symmetric centers, is

    det Q = -4m^2(a_1 b_2 - a_2 b_1)^2.

The planted pair, cluster, and background-masking experiments below use
unit multiplicity. The weighted formulas above also cover multiple zeros.

**A necessary negative control:** on-line background can mask the
negative direction in a finite compression. The experiment exhibits
this explicitly. Cluster-aware probe design is therefore an instrument
to investigate visibility; no universal detection theorem follows.

## 4. The zero-free strip becomes a tail-envelope improvement

**Conditional lemma, derived here.** Suppose a supplied zero-free
result plus the functional equation confines the nontrivial zeros to
`|beta-1/2| <= a`. Let `phi` be supported in `[-B,B]`, with enough
regularity to integrate by parts `r >= 1` times without boundary terms,
and let `C_r = ||phi^(r)||_1`. Let there be `d` modulated probes with
`|t_j| <= R`. For `H > R`, the operator contribution `E_H` from zeros
with `|gamma| > H` obeys

    ||E_H||_op <= d C_r^2 exp(2 B a)
                   * sum_{|gamma|>H} m_rho (|gamma|-R)^(-2r).

*Proof.* Integration by parts gives
`|F_j(z_rho)| <= C_r exp(Ba)/|z_rho-t_j|^r`, and
`|z_rho-t_j| >= |gamma|-R`. Each individual rank-one term `vv^T`
has operator norm `||v||_2^2`. Sum these norms with multiplicities.
The reflected tail is Hermitian even though individual terms need not
be. Standard zero counting makes the displayed sum convergent. QED.

For **identical windows, centers, and counting envelopes**, the bounds
compare as follows:

| Supplied zero-free half-plane | Allowed `a` | Envelope relative to the classical `a=1/2` strip |
|---|---:|---:|
| `Re s > 11/12` | `5/12` | `exp(-B/6)` |
| `Re s > 7/8` | `3/8` | `exp(-B/4)` |

These factors improve this coarse *upper-bound expression*. They do
not assert that the actual tail norm decreases by that factor, or
that larger bandwidth becomes uniformly cheaper: derivative norms,
prime support, and all other errors still have to be paid for.
The lemma is conditional on the chosen published input; its proof
does not independently establish either quasi-RH manuscript.

For a future certified matrix approximation `W_hat` with an independently
proved error `||W-W_hat||_op <= eta`, Weyl's inequality gives the useful
acceptance gate `lambda_min(W_hat) < -eta`. The scripts do not yet
provide that full error enclosure.

## 5. Arithmetic-side reconstruction — implemented and run

`compute/weil_prime_side.py` assembles a two-probe compression from
the explicit formula, rather than prescribing a zero configuration.
For the chosen transforms put `g_ij=f_i*f_j`, with
`f_j(u)=phi(u) exp(i t_j u)`. Its even part is used in the explicit
formula. In the stated Fourier convention,

    W_ij = 2 Re(F_i(i/2) F_j(i/2))
         + integral_R F_i(t) F_j(t) mu(t) dt
         - 2 sum_{n<=exp(2B)} Lambda(n)/sqrt(n) Re(g_ij(log n)),

    mu(t) = [Re digamma(1/4+i t/2) - log pi]/(2 pi).

The script includes prime powers. At `B=1` the arithmetic sum is over
`2,3,4,5,7`; dropping `4` would change the form. Pole and gamma terms
are indispensable too. This is not the original positive prime-only
Gram matrix.

The normalized window is

    phi(u) = (4/(3B)) cos^4(pi u/(2B)), |u|<=B; zero elsewhere.

It vanishes to fourth order at the endpoints, so its extension is
`C^3` and is an admissible `C_c^2` window. Its transform is entire:

    q = Bz/pi,
    F(z) = sinc(q) + (2/3)[sinc(q-1)+sinc(q+1)]
                     + (1/6)[sinc(q-2)+sinc(q+2)],
    sinc(x) = sin(pi x)/(pi x), sinc(0)=1.

The arithmetic assembly uses no zero sum. A known first-zero ordinate
chooses the two probe centers, and seven rounded ordinates already in
the repository provide a separate validation sum. **This is a
reconstruction check at a selected known height, not blind discovery.**

## 6. Execution record and mathematical limits

Commands, from the repository root:

```bash
python compute/weil_signature.py --output compute/results/weil_signature.json
python compute/weil_prime_side.py --output compute/results/weil_prime_side.json
```

Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. CPU only.
The saved JSON files contain parameters, controls, and raw matrices.
The planted spectra in `weil_signature.json` use unit-multiplicity atoms.
The arithmetic-side reconstruction retains the full Weil form; its separate
seven-pair validation sum assigns unit multiplicity to the recorded zeros.

| Exercise | Observation |
|---|---|
| Window transform vs independent adaptive integration, 36 cases | Maximum difference `3.49e-16` |
| Pair determinant and on-line positivity, seed 20261028 | 160 varied probe geometries pass |
| Small on-line cluster channel | Fitted exponent `1.99999745` |
| Small off-line signed channel magnitude | Fitted exponent `2.00000255` |
| Background-masking control | Minimum changes from `-3.00182697e-5` to `+0.00816691606` |
| Primorial cube, maximum support 510510 | Rayleigh/eigenvalue `0.01627466184` |
| Deeper divisor box, maximum support 900900 | `0.01054705714`, beating that cube |
| Arithmetic-side matrix, integration cutoffs 64 vs 256 | Operator difference `3.39e-12` |
| Same, 128 vs 256 | Operator difference `2.45e-15` |
| Arithmetic matrix vs seven rounded zero pairs | Operator discrepancy `3.59e-10` |
| Arithmetic-side eigenvalues | `0.02103983339`, `1.81908603503` |

The declared numerical acceptance thresholds are in the scripts, before
the calculations. They are floating-point diagnostics, not rigorous
interval bounds. The integral-cutoff comparison is a convergence check,
not an enclosure of the infinite tail. The low-zero sum has both
rounding error and omitted terms. No negative eigenvalue of the actual
zeta form was found or is claimed.

An independent agent reran the programs and checked the pair algebra
and window transform by fixed Gauss quadrature, rather than relying on
agreement with the adaptive quadrature used by the implementation.
It also independently enumerated the divisor supports. The tail lemma
was reviewed separately. On the arithmetic side it varied `(B, probes)`
over `(0.8,[11,16])`, `(1.6,[20,22])`, and
`(1,[10.3,14.7,20.1])`: low-zero discrepancies were respectively
`1.88e-9`, `8.73e-11`, and `4.39e-9`. Direct improper integration of
the gamma term agreed with the panel implementation to `1.18e-11`
or better across those cases. Invalid bandwidths and an insufficient
cutoff were rejected. These are additional numerical checks, not
rigorous interval enclosures. Those computational checks did not formalize
the new lemmas. The original computational session did not rebuild the
existing Lean files; its runtime had no `lake` executable.

**Lean verification follow-up:**
[`WeilVerification.lean`](../lean/RHCrystal/WeilVerification.lean) now proves
25 theorems covering the finite pair/cluster identities, multiplicity
scaling, the integer moment obstruction, exponential envelope comparisons,
and the rank-one operator-norm and tail-summation steps. The tail results
take componentwise Fourier decay and scalar summability as explicit
hypotheses. From `lean/`, `lake build` and
`lake env lean check_pr1_axioms.lean` pass; the fail-closed audit checks
157 declarations across five project modules and permits only `propext`,
`Classical.choice`, and `Quot.sound`. The Fourier-decay derivation, zero
counting, Taylor remainder, explicit formula, and external strip results
remain outside this local formalization.

## 7. What would make this advance RH further

The recent proportion argument uses a sharp rank–trace certificate;
its paper explicitly discusses the limits of its first-two-moment
information and bandwidth-one regime. A concrete elementary obstruction:
the positive spectrum `(1,15,29)` and indefinite spectrum `(-1,21,25)`
have the same trace `45` and squared Hilbert–Schmidt norm `1067`.
Their inertia differs. Reusing only those two moments cannot settle
positivity in general.

The next target is therefore **additional, quantitatively controlled
spectral information**: a cluster-adapted finite probe family, built
on the arithmetic side, with a rigorous error enclosure and a bound
that survives increasing height and bandwidth. Merely enlarging a
matrix or observing positivity at known zeros does not supply it.

On the approximation side, continue the surviving chain work and the
corrected arithmetic floor question. The primorial obstruction is
independent of zero locations; a zero-free strip cannot restore the
refuted `1/log^2 N` GCD-floor conjecture. `amplitude_theory.md` still
assumes RH and simple zeros and still lacks its stated convergence
justifications. The new simple-zero proportion does not discharge
those hypotheses.

See [Plan 9](../plans/09-weil-signature.md) for the next exact gates.

## Sources and provenance

1. [Alpöge–Furman, *More than two thirds of the zeta zeros are simple and on the critical line*, arXiv:2608.13637](https://arxiv.org/abs/2608.13637).
   Sections 1–3 give the compressed form, normalization, and rank–trace argument;
   Section 7 discusses sharpness. AI-discovery provenance is explicit in the paper.
2. [Companion Lean project](https://github.com/anthropics/formal-math/tree/16a5ce41304fcdaea4e0be7c132e1c361dc26da5/zeta23),
   pinned to the inspected October 28 snapshot. We read its statement/verification
   documentation, not rebuilt its complete analytic development.
3. [OpenAI, *The Quasi-Riemann Hypothesis: A Zero-Free Half-Plane Re(s)>7/8*](https://github.com/openai/math/blob/7ef5c0cc5ca4b0e072b7a050fca09ab24bc6fa3a/preprints/The-Quasi-Riemann-Hypothesis-September-30-2026/paper.pdf).
4. [OpenAI, human-assisted 11/12 manuscript and source](https://github.com/openai/math/tree/7ef5c0cc5ca4b0e072b7a050fca09ab24bc6fa3a/preprints/The-Quasi-Riemann-Hypothesis-October-5-2026).
   Its introduction/outline reduces zero exclusion to power-saving smoothed
   twisted Möbius sums. That is a different analytic target from the GCD
   kernel floor; no transfer theorem is asserted here.
5. In-repository foundations: [v4](rh_crystal_v4.md),
   [scaling theory](scaling_theory.md),
   [Nyman–Beurling](../nyman-beurling/README.md),
   [K-floor Session 27](../nyman-beurling/kfloor.md#session-27-lemma-a-is-false--the-primorial-refutation-inter-agent-verified-and-what-falls-with-it),
   and [amplitude theory](../nyman-beurling/amplitude_theory.md).

This session's derivations and code were produced with Codex. Existing
results retain their original attribution. No outreach was sent.
