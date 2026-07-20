# Results Log

## Session 1 — June 9, 2026: first computation, N ≤ 500

Code: `compute_dN.py`. Raw output: `~/rh_output/nb_results.txt`; Gram matrix
saved to `~/rh_output/nb_gram.npz`. RTX 3090, float64; the full coprime Gram
build to N = 500 (76,116 pairs) takes 1.1 s.

### Validation (all pass)

| check | scheme | reference | diff |
|-------|--------|-----------|------|
| A(1,1) | 1.260661401498 | log 2π − γ = 1.260661401508 | 1e−11 |
| A(1,2) | 0.772209256 | brute-force integration | 4e−10 |
| A(2,3) | 0.441103509 | brute-force integration | 3e−10 |
| A(2,4) | 0.386104628 | brute force (tests gcd reduction) | 6e−10 |
| ψ₁ at 1, 1.5, 2 | — | π²/6 etc. | ~5e−11 |

Spectral-cutoff sensitivity of d_N²: zero at every N ≤ 500 (κ ≤ 10⁶ —
float64 has headroom to N of several thousand).

### Finding 1: the Burnol/BDBLS rate law, visible by N = 50

d_N² · log N against the conjectured constant C = 2 + γ − log 4π = 0.0461914:

| N | d_N² (0,∞) | d²logN (0,∞) | d²logN (0,1) |
|------|------------|--------------|--------------|
| 10 | 0.0237719 | 0.05474 | 0.05253 |
| 50 | 0.0118691 | 0.04643 | 0.04558 |
| 100 | 0.0101888 | 0.04692 | 0.04618 |
| 200 | 0.0089207 | 0.04727 | 0.04661 |
| 300 | 0.0079949 | 0.04560 | 0.04504 |
| 400 | 0.0077078 | 0.04618 | 0.04563 |
| 500 | 0.0074199 | 0.04611 | 0.04558 |

Both normalizations sit within ~2% of C from N = 50 onward and oscillate
around it. **The fluctuation around C (e.g. the dip at N = 300) is itself
interesting** — the known second-order asymptotics of d_N² carry oscillatory
terms from the zeta zeros; fitting the residual d²logN − C against
zero-frequency oscillations is an obvious next experiment.

### Finding 2: the null directions are (k, 2k) dilation chains — with an exact identity

The λ_min eigenvector of G_500 has PR ≈ 10 and is supported on pairs
(k, 2k) with k near N/2: components at k = 246, 247, 248, 249 paired
against 492, 494, 496, 498, with v_k ≈ −v_{2k}/2.

Mechanism, verified exactly: for t > 1/k both e_k and e_{2k} are pure
power laws (1/(kt), 1/(2kt)), so

    e_{2k} − e_k/2  is supported entirely on (0, 1/k],

and by gcd-scaling its norm is exactly

    ‖e_{2k} − e_k/2‖² = c₀/k,    c₀ = (3/4)(log 2π − γ) − A(1,2) = 0.1732868…

Numerically confirmed to 7 digits at k = 50…249. Pairs with k ∈ (N/2, N]
admissible give single-pair Rayleigh quotients ~ c₀·2/N; the observed
λ_min ≈ 1.3/N² (λ_min·N² = 1.25, 1.31, 1.33, 1.37 at N = 200…500) is a
factor ~N below that — the eigenvector gains the extra power of N by
combining ~10 chain pairs so that the leftover (0, 1/k]-supported
corrections also cancel against each other. This is the direct analog of
the compound-cluster discovery in the zeta-zero Gram project: single-pair
bounds are loose; chains win.

**Conjecture (numerical): λ_min(G_N) ≍ N^{−2}** (possibly with a slowly
varying factor — λ_min·N² drifts 1.25 → 1.37 over N = 200…500), hence
κ(G_N) ≍ N². A proof via the chain structure looks accessible and would
be a clean lemma about the NB basis. Note the contrast with the zeta-zero
Gram matrix, where λ_min was governed by random extreme events; here the
degeneracy is *deterministic and arithmetic* (dilation chains), which is
why the law is clean.

### Finding 3: Möbius structure of the optimal coefficients, sign corrected

The optimal c_k correlate at |r| = 0.90 with −μ(k)(1 − log k / log N) —
the sign is *negative* Möbius, as the identity Σ_{k≤x} μ(k)⌊x/k⌋ = 1
dictates (our first run had the sign backwards in the comparison; the data
corrected us, and the derivation confirms the data). Sample at N = 500:
c₁ = −0.978, c₂ = +0.964, c₃ = +0.958, c₄ = +0.063 (μ(4) = 0), c₆ = −0.779.
Squarefree-with-odd/even prime factors alternate as −μ dictates; c_k at
non-squarefree k are near zero but not exactly zero (c₄ = 0.063) — the
deviation of c from the −μ profile is exactly where the second-order
structure of d_N lives.

### Next steps

1. Extend to N = 2000–5000 (cost ~N³ in the Gram build, ~70 s at N = 2000;
   float64 conditioning still fine). Sharper d²logN − C residual.
2. Fit the residual d_N²logN − C against oscillations cos(γ_j log N) from
   the first zeta zeros — if the zeros are visible in the residual, we are
   literally watching RH data in an RH-equivalent quantity.
3. Prove λ_min ≍ N^{−2} via the (k, 2k)-chain structure (candidate Lean
   target after paper proof: the support identity and the c₀/k norm are
   exact algebra over our existing formalized machinery's style).
4. Residual function profile r_N(t): where the unapproximable mass lives.
5. Deviation field δ_k = c_k + μ(k)(1 − log k/log N): structure?

---

## Session 2 — June 9, 2026: full curve to N = 2500, zero hunt, chain structure at scale

Code: `oscillations.py` (+ shared `nb_gram.py`). Raw output:
`~/rh_output/nb_oscillations.txt`. Gram build to N = 2500: ~3 min GPU;
the **single-Cholesky trick** then gives the entire d_N curve at once
(leading-submatrix property: y = L⁻¹b, d_N² = 1 − Σ_{k≤N} y_k²; the
Gram–Schmidt energies y_k² = d_{k−1}² − d_k² are themselves forensic data).

### d_N²·log N at larger N — the rate law holds

| N | 500 | 1000 | 1500 | 2000 | 2500 |
|---|------|------|------|------|------|
| d²logN | 0.04611 | 0.04529 | 0.04567 | 0.04647 | 0.04651 |

Slow oscillation around C = 0.0461914 continues; at N = 2500 we sit 0.7%
above it.

### Zero hunt: honest negative

Two statistics, two failures to detect the zeta zeros:

- *Cumulative*: periodogram of d_N²logN − C after 1/log-power drift
  removal. Peaks at freq 3.0, 5.1, 10.3, 8.1 (drift-removal artifacts at
  low frequency); nothing at γ₁ = 14.13 / γ₂ = 21.02 / γ₃ = 25.01 above
  amplitude ~5×10⁻⁵.
- *Increments*: s_N = N log²N · y_N² has mean 0.0412 (≈ C, consistent
  at finite-N level) but rms fluctuation 0.058 — the sequence is dominated
  by **arithmetic noise** (divisor structure of each N), and its
  periodogram peaks are not at zero frequencies either.

Conclusion: at N ≤ 2500 the fine structure of d_N is arithmetic
fluctuation, not visible zero oscillation. Detecting the zeros here needs
larger N, smarter averaging of the increments, or a theoretical prediction
of the expected amplitude (which may be genuinely tiny). Parked with data
saved (`nb_periodogram.npz`).

### λ_min = Θ(N^{−2+o(1)}), and the chain structure strengthens

| N | 500 | 1000 | 1500 | 2000 | 2500 |
|---|------|------|------|------|------|
| λ_min·N² | 1.365 | 1.432 | 1.504 | 1.526 | 1.548 |

The slow upward drift suggests λ_min ~ (a + b·log N)/N². At N = 2500 the
null direction (PR = 13) is supported on **even** k near N — 2478…2496 —
paired with their halves 1242…1246 near N/2, exactly the (k, 2k) doubling
chains of Session 1, now cleanly visible at scale.

### Two new structural observations

1. **Gram–Schmidt energies are arithmetically flat.** For k ≥ 100, mean
   y_k² is 1.82×10⁻⁶ (squarefree) vs 1.72×10⁻⁶ (non-squarefree) —
   essentially no difference, despite the coefficients c_k tracking −μ(k)
   at r = 0.88. The Möbius structure lives in the *correlations between*
   dilations, not in the marginal contribution of each new one.
2. **The deviation field δ_k = c_k + μ(k)(1 − log k/log N) concentrates
   on smooth squarefree numbers.** Largest |δ_k| at k = 2500 (boundary),
   2370 = 2·3·5·79, 2262 = 2·3·13·29, 1155 = 3·5·7·11, and notably
   2310 = 2·3·5·7·11 (the 5th primorial). rms(δ) = 0.17 on squarefree vs
   0.086 on non-squarefree: the refinement of the Möbius profile happens
   where numbers have many small prime factors.

### Next steps (revised)

1. Prove λ_min ≍ N⁻²(log N?) via the doubling-chain structure — most
   tractable, now well-supported empirically across 5× in N.
2. Zero hunt v2: window-averaged increments (sum y_k² over dyadic blocks
   kills arithmetic noise ∝ 1/√width), or push N to 10⁴ (Gram build
   ~N³ → ~3 h; feasible overnight).
3. Theory for the smooth-number deviation field (connects to how 1/ζ
   partial sums get corrected — Möbius inversion truncation errors live
   on smooth numbers).
4. Residual profile r_N(t) (unchanged from Session 1 list).

---

## Session 3 — June 9, 2026: the doubling-chain mechanism, dissected

Code: `chains.py`; theory write-up: `chains_theory.md`. Raw output:
`~/rh_output/nb_chains.txt`.

### Proved exactly (elementary)

- **Square-wave identity:** e_{mk} = e_k/m + (⌊1/(kt)⌋ mod m)/m, so chain
  differences are square waves supported on (0, 1/k]; verified pointwise
  to machine precision and against the Gram data at 1e−12.
- **C₂ = (log 2)/4**, and the closed form
  A(1,2) = ∫ρ(1/t)ρ(1/2t)dt = (3/4)(log 2π − γ) − (log 2)/4, agreeing with
  the computed entry to all digits.
- **Zero-sum reduction:** on Σα = 0, chain combinations reduce to the
  oscillatory functional (1/16)∫[Σα_k(−1)^{⌊u/k⌋}]²u⁻²du.

### Measured

- **The chain subspace attains λ_min within a stable factor 1.19–1.23**
  across N = 500…2500, saturating at ~20 chains; 3-chains add nothing.
  The optimal α: alternating signs, smooth envelope, Σα = 0.0000.
- **Kernel law:** M_{jk} = C₂/max(j,k) − φ(|j−k|)/K² with
  φ(d) = d(1.073 − 0.143·log d), fit to ~3 decimals over d = 1…159.
- Block-averaged zero hunt (queued item): still negative; one minor peak
  at 32.87 near γ₅ = 32.935 but isolated among larger non-zero peaks —
  not a detection.

### Status

λ_min(G_N) ≍ N⁻²·(slowly varying) now has a complete mechanistic account
with a concrete proof program (chains_theory.md §5); the nearest rigorous
target is the upper bound λ_min = O(N⁻² log N) via the explicit zero-sum
test family, pending only error-term bookkeeping in the kernel
asymptotics. If completed: κ(G_N) ≍ N², a deterministic conditioning law
for the NB basis (literature check pending — the mechanism may be new).

---

## Session 4 — June 9, 2026: THEOREM 1 (the upper bound), kernel constants derived

### Theorem 1 (proved; chains_theory.md §0)

    λ_min(G_N) ≤ (H_{K−1} + 1)/(10·K(K−1)),   K = ⌊N/2⌋,

so λ_min(G_N) = O(N⁻²·log N) and κ(G_N) ≥ c·N²/log N. The witness is a
four-term integer combination of dilations (the difference of two adjacent
doubling chains at K−1, K); the proof is the square-wave identity, an
exact disagreement-interval lemma (the parity-mismatch set of adjacent
counters is ⋃_{ℓ≤K−1}[(K−1)ℓ, Kℓ), a disjoint union), a harmonic sum, and
a one-line tail bound. Every step numerically validated: exact
enumeration vs Gram data agree to 0.12%; the tail term sits at 54% of its
proven bound; the theorem bound evaluates to 5.58e−7 at N = 2500 against
the true λ_min = 2.477e−7 (valid, factor 2.3).

First theorem of the NB phase. Candidate for Lean formalization (finite
sums + interval measure arithmetic; heavier than the interlacing bound
but feasible).

### Kernel constants derived (Session-3 fit superseded)

Exact decomposition M_{jk} = (log2/8)(1/j + 1/k) − P_{jk}/16 with P the
parity-disagreement integral; enumeration at K = 1250 gives
P·K²/(2d) = log(K/d) + γ + c(d), c(d): 0.545 → 0.661 over d = 1…32,
drifting toward log 2. **The kernel's log-slope is exactly 1/8**; Session
3's fitted 0.143 was contamination from normalizing mixed-scale pairs by
a single K² — a useful cautionary note on fitted constants.

### Remaining for λ_min ≍ N⁻²(slowly varying)

The matching lower bound (conditionally-positive-kernel program,
chains_theory.md §5) and closing the factor-1.2 chain-subspace gap.

### Overnight job launched

`overnight_10k.py`: Gram to N = 10⁴ (~2 h), full d_N curve, Theorem-1
check at scale, zero hunt v3 with 800 geometric blocks. Results next
session.

### Overnight N = 10⁴ results (completed same day, 166 min)

1. **Rate law to 10⁴:** d²logN = 0.04492, 0.04553, 0.04556, 0.04513 at
   N = 4000…10000 — holding within 2–3% of C, now with a small persistent
   *negative* deviation, consistent with a second-order term −c₂/log²N
   (c₂ ≈ 0.01). The approach to the Burnol constant is from below in this
   range.
2. **Theorem 1 at scale:** bound/actual = 2.25, 2.29, 2.33 at
   N = 2500, 5000, 10000 — the theorem captures the exact order. The
   measured law is λ_min·N² ≈ 0.50 + 0.134·log N (the three points are
   linear in log N to 1%), against the theorem's 0.4·log(N/2) + O(1):
   asymptotically sharp within a factor ≈ 3.
3. **Zero hunt v3: definitively negative at this scale.** With 4× the
   range and 800 blocks, top periodogram peaks sit at low frequencies
   (7.3, 4.8, 9.8, 11.5); nothing at any zeta zero above ~5×10⁻⁵. Three
   statistics across two ranges now agree: the fine structure of d_N at
   N ≤ 10⁴ is arithmetic noise, not zero oscillation. Parked — resuming
   requires a theoretical amplitude estimate (if the zero terms decay
   like N^{−1/2}, they are an order of magnitude below our noise floor,
   and no feasible N reaches them without better averaging).

---

## Session 5 — June 9, 2026: Theorem 1's analytic core machine-verified

`lean/RHCrystal/NymanBeurling.lean` (new module). Formalized in Lean 4
with zero sorries and only the three standard axioms (`#print axioms`
confirmed):

- `interval_of_odd` — the disagreement-interval lemma: if the adjacent
  counters ⌊u/J⌋, ⌊u/(J+1)⌋ disagree in parity at u ∈ (0, J(J+1)), then
  u ∈ [Jℓ, (J+1)ℓ) for some 1 ≤ ℓ ≤ J.
- `disagreement_integral_le` — the full analytic core of Theorem 1:
  ∫_D u⁻² ≤ (H_J + 1)/(J(J+1)) for the parity-disagreement set D, via
  exact FTC evaluation on each interval, the harmonic sum, measurability
  of D (floor functions), and the rpow tail integral.

What remains on paper (textbook steps, not formalized): the L²
bilinearity bookkeeping ‖f_{K−1} − f_K‖² = P/8 and the Rayleigh
principle — the latter is itself already formalized for matrices in
RHCrystal.lean (`exists_eigenvalue_mul_le_rayleigh`); connecting it
requires formalizing the e_k as L² elements, a larger project.

The repo now has four machine-verified results: the weight symmetry, the
interlacing pair bound, and the NB disagreement bound (plus cosh ≥ 1).

---

## Session 7 — July 19, 2026: the kernel constant exact — c* = 1 + log(2/π)

Code: `kernel_cd.py` (Plan 1, `plans/01`); raw output
`~/rh_output/kernel_cd.txt`. Environment note: clean machine, no GPU, no
cached June data — every number below regenerated from scratch, which
made the cached-value anchors (from the committed logs) genuine
cross-machine validations.

### Derived and confirmed

1. **Exact P evaluation (new method).** The parity pattern of
   ⌊u/j⌋ − ⌊u/k⌋ is periodic (period jk, doubled for odd gap after gcd
   reduction), so the disagreement integral telescopes to digamma
   differences — machine-precision P for every pair, O(K) cost.
   Validation: brute-force brackets (8 small pairs, all inside);
   Session 4's Gram-derived P(1249,1250) reproduced to rel. 9×10⁻⁷;
   all four cached λ_chain(K, w=20) values of `chains.py` reproduced to
   ≤ 7×10⁻⁵ from the decomposition alone. Exact special value:
   **P(1,2) = π/2** (Leibniz series — the disagreement set is
   ⋃[4r+1, 4r+3)).
2. **The kernel constant in closed form.** The local odd-parity
   fraction converges to the triangle wave dist(λ, 2ℤ) in the mean
   parameter λ = ud/(jk); the head is the harmonic sum, the tail is
   ∫₁^∞ dist(λ,2ℤ)·λ⁻²dλ = **1 + log(2/π)** (Wallis product). So

       P·jk/(2d) = H_{⌊j/d⌋} + c*,   c* = 1 + log(2/π) = 0.548417…

   to 5–6 decimals across K = 156…5000, d = 1…64. The Session-4
   "c(d) drifts toward log 2" reading is **superseded**: the drift was
   normalization (K² vs jk) plus discreteness (log vs harmonic number);
   the June table is reproduced exactly by the exact P. Third instance
   of the fitted-vs-derived lesson, and the second time a measured
   "constant" dissolved under normalization forensics.
3. **§5 lemma pre-check passed.** The zero-sum floor K²·min αᵀMα/‖α‖²
   is flat in window width (0.51 → 0.48 over w = 8…200 at K = 250;
   0.61 → 0.59 at K = 1250) — the w-independent-constant premise of the
   lower-bound program survives. The log-drift of λ_min·N² is located
   in the kernel's K-dependence, not the window. Unconstrained and
   zero-sum floors coincide to 4 decimals.

### Status after this session

The upper bound is Theorem 1; the kernel is now exact (formula + c*);
the lower-bound program's remaining mathematical content is the
conditional positivity αᵀΨα ≳ ‖α‖² on zero-sum α (the −|d| part has a
two-line proof via partial sums; the d·log d part is the open step) plus
the chain-to-full-G extension. Details in `chains_theory.md` §4.1–4.2.

---

## Session 8 — July 20, 2026: THE ZEROS ARE DETECTED (Plans 2, 3, 5)

Code: `amplitude.py`, `amplitude_verify.py`, `residual.py`,
`vasyunin_check.py`, `plan5_sr.py`; Gram rebuilt from scratch on a clean
machine (`nb_gram_np.py`, numpy port of the June builder, validated to
1e−10–1e−12 against closed forms and June values before use). Figures:
`figures/zero_detection.png`, `figures/residual_profile.png`.

### Finding 1 (headline): the zeta zeros are visible in the trial-function distance

The three June zero hunts failed on the *optimal* d_N² — dominated by
arithmetic noise from the minimizer chasing each N's divisor structure.
Plan 2's bet: the **Möbius trial function** c̃_k = −μ(k)(1 − log k/log N)
gives a distance curve D̃_N² = 1 − 2c̃ᵀb + c̃ᵀAc̃ that is smooth in the
coefficients (new dilations enter with weight zero), killing the
arithmetic noise. Result, N ∈ [50, 2000], 360-point log grid:

| statistic | γ₁ = 14.13 | γ₂ = 21.02 | γ₃ = 25.01 | noise floor |
|-----------|-----------|-----------|-----------|-------------|
| trial curve D̃²logN | **3.21e−3 (top peak, at 14.140)** | 9.9e−4 | 7.1e−4 | 3.3e−4 |
| optimal curve d²logN | 3.2e−5 (absent) | 3.9e−5 | 4.7e−5 | 7.0e−5 |

The optimal-curve row reproduces the June negative exactly; the trial
row lights up at all three zeros. **Verification battery (all passed):**
amplitude stable under drift bases 0..2/0..3/0..4 (SNR 9.0–9.3);
present in every N-subrange (SNR 5.6–7.4); γ₁, γ₂ survive Hann
tapering; **phase at γ₁ coherent across disjoint half-ranges to 0.035
rad**. The 11.74/16.55 companions sit symmetrically at γ₁ ± 2.40 and
behave as AM sidebands of the decaying γ₁ envelope, not independent
lines. Measured amplitude decay across ranges: 3.9e−3 (mean N ≈ 190) →
1.2e−3 (mean N ≈ 1180); measured ratios A₁:A₂:A₃ = 1 : 0.31 : 0.22
versus the naive 1/|ρ|² law's 1 : 0.45 : 0.32 — same order, faster
falloff; deriving the correct amplitude law A_j(N) (|ζ′(ρ)| weights)
is Plan 2 Step 2, now with measured targets.

*The Session-1 phrasing — "we are literally watching RH data in an
RH-equivalent quantity" — is now a measurement, not an aspiration.*

### Finding 2: the residual mass is spread, not concentrated (Plan 3A)

Exact scheme (no quadrature): in u = 1/t coordinates the residual is
piecewise linear with integer breakpoints — r̃(u) = 1 + T(⌊u⌋) − uS,
T by divisor sieve — so the profile integrates in closed form. Energy
closure (head + S² + tail vs 1 − bᵀc): 2–7×10⁻⁴ relative at
N = 100…2000, all PASS. Findings: the t > 1 mass S² is negligible
(≈0.1% of d²); the bulk sits at intermediate scales u ~ 10–10³ with
median-mass point u_½ = 66, 122, 218, 243 at N = 100…2000 — growth
≈ N^0.44, NOT u ~ N: the unapproximable mass does not live at the
missing dilations; it lives well inside the available range, spreading
slowly outward (mass beyond u = 4000: 0.9% → 11%).

### Finding 3: Vasyunin closed, three ways (Plan 3B)

The Vasyunin formula was obtained verbatim (Darses–Hillion
arXiv:2004.10086, quoting [Vas95], [BDBLS00 p.141]) and implemented:
agreement with our exact-scheme A entries to 1e−12–1e−15 on 20 coprime
pairs + gcd-scaled cases; **A(1,2) = (3/4)(log 2π − γ) − (log 2)/4 is
verified to be a Vasyunin special case** (hand derivation: n=1, m=2,
cot(π/2) = 0). The parity-disagreement integral P now has three
independent computations agreeing to 1e−11: interval enumeration,
digamma-period (Session 7), and Vasyunin combinations. LITERATURE.md
updated; the arXiv:2405.06349 overlap flag is resolved (Ehm's paper is
the *Mellin-weighted* kernel, no spectral content — and its Theorem 2.1
is a useful published anchor for our Mellin identity).

### Finding 4: the uncertainty lemma is tight against the adversary (Plan 5)

`mellin_lower_bound.md` states and proves the elementary coefficient
uncertainty lemma: for unit x, Σ_{r≤N} τ(r)s_r²/r ≥ 1/(N(log N + 1)),
s_r = Σ_{d|r}x_d. Measured saturation Q·N(log N+1) against the TRUE
λ_min eigenvectors of A_N: 29, 31, 52, 64 at N = 200…2000 — the
elementary floor loses only a slowly growing factor (~N^0.35) against
the real adversary. The eigenvector's s_r mass concentrates at
r ≈ N/2 and r ≈ N (the chain-cancellation remnant; top 12 r's = 70% of
Q). Consequence for the assembly: the dominant loss in the sketched
λ_min ≫ N^{−3−ε} bound is the T ≍ N² window, not the lemma — M3's
target is the window, as diagnosed.

### Also this session (Plan 4)

`lean/RHCrystal/NymanBeurlingL2.lean` (new): the dilation family
formalized in L² — `eDil_memLp_two`, zero sorries, standard axioms,
full project build green (3499 jobs) against current Mathlib. Milestone
1 of the end-to-end Theorem 1 formalization; scouting notes for M2–M4
in `plans/04`.

---

## Session 19 — July 20, 2026: THEOREM 1 MACHINE-VERIFIED END TO END

`lean/RHCrystal/NymanBeurlingMain.lean` (new): `nb_gram_eigenvalue_le`
— for 2 ≤ J and 2(J+1) ≤ N, some eigenvalue of the N×N Nyman–Beurling
Gram matrix is at most (H_J + 1)/(10·J·(J+1)) [= Theorem 1's
(H_{K−1}+1)/(10·K(K−1)) with K = J+1]. **Zero sorries; axioms
propext/Classical.choice/Quot.sound only; full build green (3500
jobs).** The complete chain is formal: e_k ∈ L² (M1) → the square-wave
identity (M2) → the witness integral as a set MEASURE (M3 — the key
simplification: in t-coordinates the u⁻²du weight is plain Lebesgue
measure, so the June disagreement bound transports by pure interval
arithmetic, no change of variables) → the L² Gram matrix, the
four-dilation witness (coefficient norm 5/2 computed formally), and
the already-formalized Rayleigh principle (M4).

To our knowledge this is the first genuinely new theorem produced by
this collaboration whose entire proof — analytic core AND spectral
assembly — is machine-verified. The June formalization covered the
analytic core; today the remaining "textbook steps" (which the June
log honestly listed as unformalized) were completed. Plan 4 is CLOSED.

Also this session: γ₆ CLEAN attempt at 10⁷ — amplitude ratio 1.09
(exactly the common offset) but SNR 2.6: still an honest no. The BCHB
kernel floor measured: λ_min(K)·log T = 6.69→6.82 ≈ CONSTANT over
N = 250…2000 — if the sourced BCHB statement cooperates, Tier 2 lands
at N^{−2−δ}/log N, nearly optimal (mellin_lower_bound.md, second
revision). Stale lean/README.md finally rewritten to match reality.

---

## Session 18 — July 20, 2026: the five-thread run (goal session)

One session across all five open threads. Logs: `thread2_killtests.txt`,
`thread3_phase.txt`, `thread3_10e7.txt`, `thread1_harvest.txt`,
`redheffer_B.txt` (+ background 16384 pending) in `~/rh_output`.

**Thread 2 — both kill-tests passed.** (a) Tier-3 Route B survives:
chain-subspace overlap of the true null vector is 92.2% (N = 2000) →
91.9% (N = 10⁴) — localization persists at 5× scale; s_r top-12 share
70% → 66%. (b) First real-data envelope scan (the pair-creation
detector on the actual curve): every resolvable line has negative
envelope exponent — γ₁ at exactly −0.50 (theory: −½), γ₂ −0.58,
γ₃ −0.50 — no brightening line. The vacuum is clean at N ≤ 10⁶.

**Thread 3 — the amplitude law's last checks.** (a) PHASES: the
predicted combination φ_j + arg ζ′(ρ_j) + 2 arg(ρ_j − 1) is constant
across all five measurable lines (0.006, 0.032, 0.126, 0.095, −0.057
mod 2π; spread 0.12 rad) — and the common constant is ≈ 0. Frequency,
envelope, intensity, normalization, first correction, and now phase:
every prediction of the law is verified. (b) j-dependence: the theory
predicts NONE at order 1/L (the j-piece is O(1/γ²) in amplitude);
measured per-zero scatter has χ² = 8.6 on 6 dof vs pipeline noise —
consistent. (c) N = 10⁷: blocked-sieve run (26 points, 20 min; seam vs
the 10⁶ leg 2×10⁻⁷). γ₁ SNR 21.6 over a 2×10⁵ range. The 1/log N
discriminator now shows structure: upper-thirds offsets decline
1.139 → 1.089 vs predicted 1.104 → 1.069 (slope matches within noise);
the lowest third (L̄ ≈ 5) sits off the first-order line — second-order
territory. γ₆ remains honestly undetected (SNR 0.9).

**Thread 1 — Tier 2 sharpened, one route dead.** The harvest at the
special indices m = qr measured with the true adversarial eigenvector:
the prime window pinches at T = N² (0.01× lemma floor) and reopens for
δ ≥ ¼ (0.67/1.72/2.59× at δ = ¼/½/¾) — T = N^{2+δ} is forced by data.
The truncated-sum variant of M2 is DEAD (MV error ~N² above main;
arithmetic recorded). Live: the √T-AFE route (χ-sum cross term = the
classical hard part) and the new preferred **BCHB reduction** — at
T = N^{2+δ} the polynomial is inside the twisted-second-moment range,
reducing Tier 2 to lower-bounding the explicit BCHB gcd-kernel form.
Details in `mellin_lower_bound.md` M2-revision. No theorem today; a
better problem.

**Thread 4 — Lean M2 complete.** `fract_half` (the halving identity
for fractional parts, fully general — no positivity hypotheses
needed), `chainDiff_eq_squareWave`, `chainDiff_values`,
`chainDiff_eq_zero` (support), `chainDiff_memLp_two` — zero sorries,
standard axioms, full build green (3499 jobs). The square-wave
identity that powers Theorems 1 and 2 is now machine-verified.

**Thread 5 — Redheffer Part B.** (a) The multiplicity mystery is
closed by source + our measurement: algebraic mult of eigenvalue 1 is
n − ⌊log₂n⌋ − 1, geometric is ⌊n/2⌋ − 1 — exactly our measured
31/63/127 — and the dominant-pair asymptotic ±√n + log√n + γ − 3/2
matches our +35.4/−28.3 at n = 1024. (b) The Mertens cancellation
anatomy, measured: Σ log σ_i = log|M(n)| verified to 10⁻⁸ at
n ≤ 8192, with the two bulks S₊ = +2604.0, S₋ = −2600.9 (n = 8192)
cancelling to 3.09 — Mertens is a one-part-in-a-thousand bulk-vs-bulk
cancellation across the whole σ-spectrum, not a few-mode effect.
(c) σ_min stays flat (0.0028–0.0049) to n = 8192. (n = 16384 SVD
running; appended when done.)

---

## Session 17 — July 20, 2026: the form-factor ladder — five new satellite rungs, all exact

Context: an external agent (via doll) proposed testing the explicit
formula's Λ(n)/√n peak-height law on the prime-power satellites as "the
positive fingerprint of the phase." That test was in fact run in June
(`prime_spectroscopy.py`, commit 0bc9a0e) for m ≤ 3. Today: zeros
re-downloaded on this machine (2,001,052, cross-machine revalidation)
and the ladder EXTENDED to the never-tested deep rungs:

| n | measured/predicted | | n | measured/predicted |
|---|---|---|---|---|
| 16 = 2⁴ | 0.354 / 0.354 | | 27 = 3³ | 0.431 / 0.431 |
| 25 = 5² | 0.657 / 0.657 | | 32 = 2⁵ | 0.250 / 0.250 |
| 49 = 7² | 0.567 / 0.567 | | | |

All June rungs reproduced identically. The agent's named ratios:
4-vs-2 = 0.7071 (1/√2), 8-vs-2 = 0.5000 (1/2), 9-vs-3 = 0.5774 (1/√3)
— four-decimal, zero-free-parameter agreement. The diffraction pattern
of the zeros carries the von Mangoldt form factors exactly.

Also this session, from the same exchange: the agent's Cooper-pairing
correction to Session-level rhetoric is ACCEPTED (Pauli forbids
condensation of *free* fermions; the wall is the measured free-fermion
kernel — no attractive channel — not fermionic statistics as such);
its Painlevé V/JMMS claim and class-II hyperuniformity classification
both verify. Inter-agent adversarial review has entered the workflow
organically — process-paper material.

---

## Session 16 — July 20, 2026: the +9% offset — pipeline exonerated, next-order term matches

`offset_check.py`. (1) Decisive control: known amplitudes injected
through the identical pipeline return 0.997 ± 0.009 (γ₁) — no
measurement bias; the offset is physics. (2) The next-order s = 1
residue term gives A_j × [1 + c₁/log N], c₁ = (ζ″/ζ′)(0) − log 2π +
1 − γ = +0.768 (computed, not recalled; ζ′(0) cross-checked to 9
digits): predicted full-range offset 1.102 vs measured 1.110 on γ₁ —
parameter-free to <1%. (3) The 1/log N scaling discriminator is
within noise on half-ranges (predicted 1.12→1.07, measured
1.109→1.104, ±5–6% per half) — size confirmed, scaling open;
j-dependent same-order terms (ζ″(ρ_j)/ζ′(ρ_j)) queued as the likely
source of the per-zero scatter. Details: `amplitude_theory.md`
addendum.

---

## Session 15 — July 20, 2026: the Tail Variation Lemma PROVED — Theorem 2 debt-free

Writeup: `chains_theory.md` §7. Validation: `tvl_check.py`,
`~/rh_output/tvl_check.txt`.

**The lemma:** for coprime j < k, d = k − j: tail·jk/(2d) = c* + τ with
0 ≤ τ ≤ A·d/j — the Session-7 c* law with explicit error, which was the
single deferred piece of Theorem 2's sharp form. **Now proved.** The
chain-subspace floor Θ(K⁻² log K) carries no outstanding debts.

**The proof's engine** (all elementary): (1) two boundary miracles —
k·m₀ ≤ U₀ exactly and I_{m₀+1} starts exactly at U₀, so the tail sees
only whole intervals; (2) (−1)^Δ = Π(1 − 2·𝟙_{I_ℓ}) expanded by
inclusion–exclusion, with the subset sum over fixed (min, max)
collapsing binomially to 4(−1)^{t−1}; (3) the **tent identity**
2G(T) − 1 = −dist(T, 2ℤ)/T — the triangle wave emerges as an exact
identity, not an average (verified 3×10⁻¹⁵); (4) alternating
Euler–Maclaurin for the ℓ/(ℓ+t) correction; (5) 1-Lipschitz
Riemann-sum comparison feeding the §4.1 Wallis integral
∫₁^∞ dist(λ,2ℤ)λ⁻²dλ = 1 + log(2/π).

**Two bonuses from the validation table:**
- The deviations match **τ = (j mod d)/j** exactly, row by row
  (d = 13, j = 1000: dev/(d/j) = 0.923 = 12/13; d = 3 family: 2/3 to
  three digits; d = 7, j = 2003: 1/7). The second-order law observed
  at (d/j)² accuracy — recorded as observed-not-derived, A = 1
  effectively, deviation one-signed.
- **Correction (repo discipline):** §6.7 had stated the constant as
  c* − γ; the correct constant is c* (the Session-7 anchor already
  said so: tail·jk/2 = 0.5485 at the Theorem-1 pair). Caught during
  the proof; Theorem 2's §6.5 assembly always used the correct
  bracket and is unaffected.

The Plan 6 Tier 1 arc is complete: Theorem 1 (upper) + Theorem 2
(lower, sharp, debt-free) ⟹ the doubling-chain family's spectral
floor is Θ(K⁻² log K) with explicit constants both sides.

---

## Session 14 — July 20, 2026: THEOREM 2 written (Plan 6 Tier 1) — the chain floor is Θ(K⁻² log K)

Writeup: `chains_theory.md` §6 (new). Verification appended to
`~/rh_output/tier1_verify.txt`.

**Theorem 2′ (fully proved, nothing deferred):** for fixed w₀ ≥ 4 and
K ≥ exp(4w₀²), every chain combination on the window (K−w₀, K]
satisfies ‖Σα_j f_j‖² ≥ (log K)·‖α‖²/(64K²). Elementary throughout:
exact rank-two vanishing on zero-sum vectors, the disjoint-interval
head of P (Theorem 1's lemma at general gap), the partial-sums
identity, box-overlap positivity, and AM–GM bookkeeping for the
non-zero-sum component. Ineffective threshold, unconditional content.

**Theorem 2 (sharp, effective at real K):** same statement with
constant (log K + γ + c* − log w₀ − 1)/2 + 1 over 8K², proved modulo
ONE named lemma — the Tail Variation Lemma (§6.7): tail·jk/(2d) =
c* − γ + O(d′/j′), i.e. the Session-7 law with explicit error. Lemma
verified numerically across K = 156…5000 (A ≤ 0.7); proof route
(parity-periodicity + Wallis average + Beatty boundary layer) is one
dedicated session.

**Corollary (with Theorem 1):** the doubling-chain family's floor is
Θ(K⁻² log K) — the four-dilation witness is order-optimal among
chains, and the measured factor ≈ 1.2 to λ_min(G_N) is the entire
remaining gap (which belongs to the Mellin program, not to chains).

**Validation ledger:** (T2.1) identity 5×10⁻¹⁵; Step-4 identity
(with its +‖α‖² bonus term, found during writeup) to 1e−8 quadrature;
sharp bound 0.237/K² vs measured floor 0.593/K² (valid, slack 2.5);
Step-6 quantities: Q(𝟙) = 4.34e−3 vs predicted 4.44e−3, row-sum
variation 8.5e−5 ≪ envelope 4.7e−3, unrestricted floor = zero-sum
floor to 4 decimals.

Next on this thread: prove the Tail Variation Lemma (retires Theorem
2's only debt); then Lean targets §6.1/6.3 (finite algebra).

---

## Session 13 — July 20, 2026: Redheffer first pass (Plan 8) — the left null space is refined Möbius

Origin: doll's question ("is there a dusty-corner statement RH
determines, that points back at RH?") → Plan 8. Gate (literature check)
passed with one flag: σ_min/near-null analysis of A_n not found;
Hilberdink's two singular-value papers unread, flagged as the
possible-overlap source. Log: `~/rh_output/redheffer_A.txt`.

**Anchors.** Exact integer determinants match M(n) at n = 5…60; the
identity **μᵀA_n = (M(n), 0, …, 0)** verified exactly (it *is* Möbius
inversion in matrix form) — giving σ_min ≤ |M(n)|/‖μ‖ for free.

**Corrected recall, measured structure.** The recalled multiplicity law
n − ⌊log₂n⌋ − 1 for eigenvalue 1 is evidently the *algebraic* count
(float eigensolvers scatter it — the matrix is heavily defective);
the **geometric multiplicity is exactly n/2 − 1** (rank(A−I) in exact
arithmetic mod p, n = 64/128/256). Dominant eigenvalue pair
≈ ±√n + (log n)/2 (n = 1024: +35.4, −28.3 vs √n = 32), consistent with
Barrett–Jarvis/Vaughan. Two instrument bugs caught and fixed in-log
(defective-spectrum tolerance, a sort_complex slip).

**The new measurements (n = 256…4096).**
- σ_max/√n drifts 1.413 → 1.377; κ ≈ 10⁴ at n = 4096 — the Redheffer
  matrix is only mildly ill-conditioned (contrast NB Gram κ ~ N²).
- **σ_min ≈ 0.004–0.008, roughly flat in n** (fit n^−0.08,
  non-monotone) — no prior σ_min analysis found (flag above).
- **The left near-null vector is refined Möbius**: corr(u_min, μ) =
  +0.999 and PR(u_min) = 2490.2 = exactly the squarefree count below
  4096. But σ_min sits a growing factor BELOW the raw Möbius bound
  |M(n)|/‖μ‖ (ratio 0.053 → 0.013 over n = 256 → 4096): the deviations
  from μ do real, increasing work — the same phenomenon as the NB
  deviation field (Session 2), now on the Redheffer side.
- v_min (right) is a different object: mass on small k (6, 14, 37, …),
  no μ correlation — left/right asymmetry to characterize.

**Open (next session):** verify the algebraic-multiplicity law from
sources; larger n via sparse/iterative σ_min; the u_min − μ̂ deviation
field's arithmetic (smooth numbers again?); whether σ_min's flatness
survives and what, if anything, it says about the det-vs-spectrum
cancellation (Plan 8 Part B).

---

## Session 12 — July 20, 2026: OUT-OF-SAMPLE PASS — the law holds to N = 10⁶, eight zeros visible

Code: `bigN_trial.py` (predictions committed BEFORE the run),
`bigN_analysis.py`. Data: `bigN_curve.npz`. Log: `bigN_analysis.txt`.
Figure: `figures/out_of_sample_10e6.png`.

**Method leap.** The trial curve needs no Gram matrix: the Plan-3A
divisor-sieve scheme evaluates D̃² exactly in near-linear time — the
full N = 50…10⁶ curve (576 points, range factor 20,000) took 10 min on
CPU, vs ~73 days for a Gram build at 10⁵. Validated against the
Gram-based curve to 2e−7 and U-converged before use. Matched filter:
rectifying by √N/(4 log 2π) makes every line constant-amplitude
1/(|ρ_j|²|ζ′(ρ_j)|) — the law read off as a flat spectrum.

**Pre-registered test.** γ₄, γ₅ amplitudes stated in advance
(`predicted_lines.npy`, commit before run). Result, after CLEANing the
known lines and predicted beats (astronomy-style):

| zero | predicted | measured | ratio | SNR | status |
|------|-----------|----------|-------|-----|--------|
| γ₄ = 30.4249 | 8.28e−4 | 9.16e−4 | 1.11 | **6.3** | **DETECTED (pre-registered)** |
| γ₅ = 32.9351 | 6.67e−4 | 7.10e−4 | 1.06 | **4.9** | **DETECTED (pre-registered)** |
| γ₆ = 37.5862 | 3.66e−4 | 3.90e−4 | 1.07 | 2.7 | peak at 37.590, sub-threshold |
| γ₇ = 40.9187 | 4.01e−4 | 4.33e−4 | 1.08 | — | unprompted peak at 40.950 |
| γ₈ = 43.3271 | 2.91e−4 | 3.35e−4 | 1.15 | — | unprompted peak at 43.230 |

In-sample lines: γ₁ 22.2σ (ratio 1.11), γ₂ 7.2σ (1.14), γ₃ 3.9σ
(1.07). **All eight ratios in 1.05–1.15** — a systematic ~+9% offset
worth chasing (second-order term of the derivation?), but the law's
structure (½ exponent, |ρ|²|ζ′| weights, absolute scale) is confirmed
out of sample across 4.5 octaves of amplitude. Beat lines confirmed as
population: peaks at γ₂−γ₁, γ₃−γ₁, γ₄−γ₃, γ₆−γ₃ frequencies; the
uncleaned floor (3.2e−4) drops to 1.5e−4 after removing them.

Honest boundaries: γ₆–γ₈ are consistent candidates, not formal
detections (sub-3σ or not pre-registered); the ~9% common offset is
unexplained; the seam between Gram-based (N < 10⁴) and sieve-based
(N ≥ 10⁴) data contributes a small systematic noted in the analysis.

---

## Session 11 — July 20, 2026: the amplitude law DERIVED, parameter-free (Plan 2 Step 2)

Writeup: `amplitude_theory.md`. Check: `~/rh_output/amplitude_theory_check.txt`.

    A_j(N) = 4·log(2π) · N^{−1/2} / (|ρ_j|²·|ζ′(ρ_j)|)

derived via Mellin + tapered-Möbius contour: the zeros' Fejér bumps
give Burnol's C/log N as the leading term (self-anchor ✓); the
oscillation comes from a contour shift whose dominant capture is the
s = 1 pole — at distance ½ from the critical line, hence N^{−1/2};
its residue carries (ζ′/ζ)(0) = log 2π (verified 2e−16) and the
1/(|ρ_j|²ζ′(ρ_j)) weights. All three measured targets hit with no
fitted constants: envelope ½ (measured 0.524); intensity law
1/(|ρ|²|ζ′|) (screen winner, log-rms 0.089); absolute γ₁ amplitude
matched at 3–12% across four N-octaves (ratios 1.06/1.12/1.05/0.97).
Bonus: predicted second-order beat lines γ_j − γ_k; the stray 10⁴-run
peak at ≈ 10.9–11.2 sits on γ₃ − γ₁ = 10.876 at 2.2× floor.

Honesty: explicit-formula rigor (RH + simple zeros, formal ρ-sum
manipulations), not theorem grade; the ×4 bookkeeping was confirmed by
data before being fully trusted (flagged); phase check open. Sharp
falsifiable prediction: γ₄, γ₅ become detectable at N ~ 10⁵–10⁶ with
amplitudes given in advance by the formula.

---

## Session 10 — July 20, 2026: N = 10⁴ on GPU — detection sharpens, envelope ~ N^−1/2

Code: GPU build (`nb_gram.py`, RTX 3090, 30.4M coprime pairs, 105 min)
+ `gpu10k_analysis.txt`. Data: `~/rh_output/nb_10k_gpu.npz`,
`gpu10k_curves.npz`. Figure: `figures/zero_detection_10k.png`.

**Theorem 1 order-sharp, confirmed at 10⁴.** λ_min(A_10⁴) = 1.7315×10⁻⁸,
so λ_min·N² = 1.7315 — the June law 0.50 + 0.134 log N predicts 1.7342
(0.15%). First exact λ_min at this scale; the N⁻²·(slowly varying) law
holds across 20× in N. d²logN = 0.04512 (Burnol C = 0.04619), the
small negative second-order deviation of Session 4 persisting.

**Detection sharpens at 5× range.** Trial-curve periodogram, N ≤ 10⁴:

| zero | amp | SNR (was, N≤2000) |
|------|-----|-------------------|
| γ₁ = 14.13 | 2.45e−3 | **12.8** (9.1) |
| γ₂ = 21.02 | 7.27e−4 | 3.8 |
| γ₃ = 25.01 | 5.05e−4 | 2.6 |

γ₁ is the top peak (at 14.165); γ₂, γ₃ are the 2nd/3rd peaks (at 20.970,
24.895). γ₄, γ₅ remain below the floor (SNR 1.5, 1.3) — the 5× range
did not yet raise them, consistent with the fast envelope decay below.
Frequency accuracy on γ₁: 14.165 vs 14.1347 (0.2%).

**Envelope decays as N^−1/2.** γ₁ amplitude across four N-octaves
(mean N = 106 → 5657): 4.77e−3 → 2.60e−3 → 1.22e−3 → 5.99e−4, fitting
**A₁ ~ N^−0.524**. A clean near-½ exponent — the target the Plan 2
Step 2 derivation must reproduce, and the reason γ₄, γ₅ stay hidden
(their base intensity is already low and the envelope crushes it).

**Intensity law, sharper.** Extended-range ratios 1 : 0.297 : 0.206
against the four laws: **1/(|ρ|²|ζ′(ρ)|) fits at log-rms 0.089** (was
0.127 at N ≤ 2000), versus 0.430 for 1/|ρ|² and 0.459 for 1/(|ρ||ζ′|).
The screen strengthens with more data — the amplitude law carries an
inverse |ζ′(ρ)| weight, now the concrete analytic target.

---

## Session 9 — July 20, 2026: spectroscopy instrument tests (Plan 7)

Code: `spectroscopy.py`; raw output `~/rh_output/spectroscopy.txt`.
Doll's question — "does the spectroscopy view give insights toward
solving RH?" — has its proof-relevant answer in Plans 5/6 (positivity =
no perfect absorption); this session captured the two instrument-grade
insights as experiments.

**A. Envelope classification (what RH-failure would look like).** A
zero at ½ + δ + iγ would show as a line whose envelope decays *slower*
by N^δ. Synthetic control: five RH-consistent lines (envelope −0.64,
matched to the measured γ₁ decay and noise floor) plus one planted
off-line pair at γ = 18, δ = 0.10. A global alternating profile fit
(per-line envelope exponents, full-range resolution — windowed fits
provably cannot separate lines 2–4 apart on this range) recovers every
RH exponent within ±0.06 and the planted exponent exactly (−0.540),
classifying it at 3.2σ. Resolution sweep: δ = 0.10 is the 3σ limit at
this range/noise; δ ≤ 0.05 invisible. This quantifies "the
spectrometer's β-resolution grows only logarithmically with effort" —
why computation diagnoses but cannot prove RH. Synthetic-only; tests
the instrument, not ζ.

**B. Inverse line intensities (the curve knows ζ′).** |ζ′(ρ_j)|
computed directly (mpmath, dps 30): 0.7932, 1.1368, 1.3717. Screening
four intensity laws against the measured A₁:A₂:A₃ = 1 : 0.309 : 0.221:

| law | predicted ratios | log-rms |
|-----|------------------|---------|
| 1/\|ρ\|² | 1 : 0.452 : 0.320 | 0.375 |
| 1/(\|ρ\|\|ζ′\|) | 1 : 0.469 : 0.327 | 0.405 |
| 1/(\|ρ\|\|ζ′\|)² | 1 : 0.220 : 0.107 | 0.567 |
| **1/(\|ρ\|²\|ζ′\|)** | **1 : 0.316 : 0.185** | **0.127** |

Best screen: A_j ∝ 1/(|ρ_j|²|ζ′(ρ_j)|) — three points against four
laws is screening, not confirmation; it is now the concrete target the
Plan 2 Step 2 derivation must hit. Inversion demo: |ζ′(ρ₁)| estimated
from the measured A₂/A₁ under that law = 0.94 vs true 0.79 (19%) — a
constant of deep arithmetic read off an approximation-theory curve.
Both parts rerun at N = 10⁴ when the GPU data lands (tighter δ-limit,
possibly γ₄, γ₅ as new data points).

---

## Session 6 — June 9, 2026: the Mellin identity verified; the spectral story

`mellin_check.py`; ζ-grid (55k points to T = 2000) cached in
`~/rh_output/zeta_grid.npz`. The identity

    xᵀA x = (1/π)∫₀^∞ |ζ(½+it)|²/(¼+t²)·|X(t)|² dt,  X(t) = Σ x_k k^(−½−it)

verified against the Gram data: random vectors agree to 0.06–0.15%, with
the residual matching the estimated T-cutoff tail almost exactly (6.2e−3
observed vs 6.7e−3 estimated on the worst case). Chain differences agree
to 1.4% at K = 20, degrading to 8.9% at K = 200 — not identity failure
but cutoff: the chain's weighted mass provably sits at t ≳ K (measured at
K = 100: only 1% of mass below t = 10, 25% below t = 100, 73% below
t = 300), so larger K pushes mass past T_MAX.

**The spectral story of Theorem 1, now verified end to end:** differencing
adjacent dilations suppresses |X(t)|² by ~t²/K² below t ~ K; the weight
contributes |ζ|²/t²; the product accumulates ζ's second moment between
t ~ 1 and t ~ K, which grows like log K — the harmonic number of the
combinatorial proof, seen from the spectral side. Every quantity measured
this week is a statement about |ζ|² on the critical line.

The lower-bound program (Montgomery–Vaughan mean values + BCHB twisted
second moments, see Session 5 discussion) now rests on a numerically
verified foundation.
