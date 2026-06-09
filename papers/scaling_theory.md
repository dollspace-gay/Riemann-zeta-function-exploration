# Toward Deriving the Scaling Law: Divided-Difference Stencils and GUE Extreme-Value Theory

*Working notes, June 2026. Companion code: `compute/rh_theory.py`; raw output
`~/rh_output/theory_results.txt`. Status: theory + numerical validation;
not a proof.*

**Summary of where this lands.** The stencil identity (§1) is rigorous and
its channel structure is confirmed numerically with exact exponents
(Test A: δ^2/δ^4/δ^6 measured as 1.95/4.01/6.05). The extreme-value step
(§2) gives channel exponents N^{−2/(k+2)} whose k = 1,2 values bracket every
exponent we have measured. The honest complication (Test B): at currently
accessible N the actual minimizers live in the *resonant* regime where the
coupling oscillates, so measured exponents are crossover values; the clean
stencil asymptotics — λ_min ~ N^{−2/3} via the pair channel — is a
prediction about larger N. The conditional-theorem target is stated in §4.*

## 1. The stencil identity (rigorous)

Work with the trace-normalized Gram matrix, diag(G) = 1, entries
G_ij = c(γ_i − γ_j) with

    c(δ) = Σ_p ŵ(p) cos(δ log p),     ŵ(p) = w(p,1/2) / Σ_q w(q,1/2).

For any real test vector x supported on zeros at positions t_1, …, t_m,

    xᵀGx = Σ_p ŵ(p) | Σ_i x_i e^{i t_i log p} |².      (∗)

(Equation (∗) is exact — it is the Gram factorization restricted to x.)

Now suppose x **annihilates polynomials of degree < k**: Σ_i x_i t_i^m = 0
for m = 0, …, k−1 (a k-th divided-difference stencil). Taylor-expanding the
exponential in (∗) with u = log p, every term below order k vanishes:

    Σ_i x_i e^{i t_i u} = (iu)^k/k! · Σ_i x_i t_i^k + O(u^{k+1} s^{k+1}),

where s is the span of the cluster. Hence, in the well-resolved regime
s · log(p_max) ≪ 1:

    xᵀGx ≈ (M_{2k} / (k!)²) · (Σ_i x_i t_i^k)²,     M_{2k} = Σ_p ŵ(p)(log p)^{2k}.

**The invisible directions of the prime-zero coupling are k-th
divided differences over zero clusters: the prime sum cannot resolve the
k-th derivative of the phase at small scales.** This subsumes and explains
the v4 findings: the pair stencil (1,−1) is k = 1 (Lehmer-pair direction);
the cluster localization of §5.5 is the k ≥ 2 channels.

Specializing to equally spaced clusters of gap δ (unit-norm stencils):

| cluster | stencil | prediction |
|---------|---------|-----------|
| pair | (1,−1)/√2 | λ ≈ (M₂/2)·δ² |
| triple | (1,−2,1)/√6 | λ ≈ (M₄/6)·δ⁴ |
| quadruple | (1,−3,3,−1)/√20 | λ ≈ (M₆/20)·δ⁶ |

These have **no free parameters**: the moments M_{2k} are computable from
the weights. Test A below checks them directly.

## 2. The extreme-value step (heuristic, GUE input)

By interlacing (machine-verified for k = 1; numerically tight for k+1-windows
in the well-resolved regime, cf. the fixed-P sweep), λ_min(N) is governed by
the best stencil over the best cluster among ~N sites:

    λ_min(N) ≈ min over k of  (M_{2k}/(k!)²) · c_k · s_k(N)^{2k},

where s_k(N) is the smallest span of k+1 consecutive zeros among N. Under
GUE statistics (β = 2), the probability that k+1 consecutive zeros fall
within a span s (in units of mean spacing) is ~ s^{(k+1)²−1} per site, so

    s_k(N) ~ N^{−1/((k+1)²−1)} = N^{−1/(k²+2k)},

giving the **channel exponents**

    λ_min^{(k)}(N) ~ N^{−2k/(k²+2k)} = N^{−2/(k+2)}:

| channel | exponent | numerical span at N = 8000 |
|---------|----------|---------------------------|
| k = 1 (pairs) | N^{−2/3} ≈ N^{−0.667} | s₁ ≈ 8000^{−1/3} ≈ 0.05 |
| k = 2 (triples) | N^{−1/2} | s₂ ≈ 8000^{−1/8} ≈ 0.33 |
| k = 3 (quadruples) | N^{−2/5} | s₃ ≈ 8000^{−1/15} ≈ 0.55 |

Two checks against the v4 data, before any new experiment:

- The smallest gap among the first 8000 zeros is the Lehmer pair,
  δ = 0.0377 — the k = 1 prediction s₁ ≈ 0.05. ✓
- The compound cluster that actually carries the λ_min eigenvector has
  gaps ≈ 0.27–0.55 — the k = 2 prediction s₂ ≈ 0.33. ✓

The measured exponents (λ_min ~ N^{−0.47} at fixed P, N^{−0.53} at P = 20N)
sit at the triple channel's −1/2, and the participation ratio ≈ 3 of the
λ_min eigenvector says the same: **at currently accessible N, the k = 2
channel dominates.**

**Falsifiable prediction.** Since −2/3 < −1/2, the pair channel must win
asymptotically: as N → ∞ (in the well-resolved regime) the exponent should
drift from −1/2 toward −2/3 and the participation ratio from ~3 toward ~2.
The crossover N is set by the moment ratio M₄/M₂ (and by resonant-regime
saturation: spans s ≳ 1/log p_max leave the quadratic regime, which caps
the k ≥ 2 coefficients).

A secondary prediction: the tail of the 3-window eigenvalue distribution.
P(pair-channel value < t) ~ P(δ < √(t/a₁)) ~ t^{3/2} (using the GUE gap
density P(s) ~ s²); the triple channel gives P(W < t) ~ t^{8/4} = t².

## 3. Numerical validation

*(Results from `compute/rh_theory.py`; tables below filled from
`theory_results.txt`.)*

### Test A: planted clusters — mechanism confirmed

Clusters planted in a rigid unit-spaced background (N = 500, P = 10,000;
moments M₂ = 96.3, M₄ = 10,185, M₆ = 1.13×10⁶; quadratic regime
δ ≪ 1/log p_max = 0.087):

| δ | pair λ_min | (M₂/2)δ² | triple λ_min | (M₄/6)δ⁴ | quad λ_min | (M₆/20)δ⁶ |
|------|---------|---------|---------|---------|---------|---------|
| 0.005 | 7.48e−4 | 1.20e−3 | 6.22e−8 | 1.06e−6 | 1.72e−11 | 8.8e−10 |
| 0.010 | 2.95e−3 | 4.81e−3 | 1.00e−6 | 1.70e−5 | 1.12e−9 | 5.7e−8 |
| 0.020 | 1.12e−2 | 1.93e−2 | 1.62e−5 | 2.72e−4 | 7.52e−8 | 3.6e−6 |
| 0.040 | (saturated) | — | 2.67e−4 | 4.35e−3 | 5.58e−6 | 2.3e−4 |
| 0.080 | — | — | 4.46e−3 | — | 5.15e−4 | — |

**Fitted small-δ slopes: pair 1.95 (theory 2), triple 4.01 (theory 4),
quadruple 6.05 (theory 6).** The exponents are exact. Three honest notes:

1. The coefficient ratios λ_measured/λ_predicted are constant in the
   quadratic regime (≈ 0.61 / 0.059 / 0.020 for k = 1/2/3) but below 1:
   the binomial stencil is an **upper-bound witness**, not the optimizer.
   The true minimizer additionally (a) trades small non-zero low moments
   against the leading one via cross terms like −(M₄/3)m₁m₃ in the full
   expansion xᵀGx = m₀² + M₂(m₁² − m₀m₂) + M₄(m₂²/4 − m₁m₃/3 + m₀m₄/12) + …,
   and (b) leaks onto background zeros ("dressing"). Both effects are
   N-independent constants; the channel exponents are untouched.
2. Each curve saturates at the background's own λ_min (≈ 0.0129) once the
   planted eigenvalue exceeds the rigid-lattice floor — as it must.
3. The saturation onset matches the resonant-regime boundary
   δ ~ 1/log p_max.

### Test B: the real eigenvector is a *resonant-regime* object (important refinement)

The λ_min eigenvector of the real zeros at N = 8000 (site 6896, γ ≈ 7173):

| idx | γ | v_i |
|------|--------|--------|
| 6895 | 7172.81 | −0.236 |
| 6896 | 7173.36 | **+0.695** |
| 6897 | 7173.62 | **+0.610** |
| 6898 | 7174.91 | +0.089 |

The two dominant components have the **same sign**, and Σv ≈ +1.0 over the
window — this is *not* a polynomial-annihilation stencil. It is the
signature of the resonant regime: at gap 0.268 the coupling c(δ) is
negative, so the 2×2 block [[1,c],[c,1]] has its small eigenvalue 1+c on
the *symmetric* vector. The current-N minimizer cancels phase against the
oscillating kernel, not against polynomials.

This cleanly partitions the problem into two regimes:

- **Resonant regime** (cluster span ≳ 1/log p_max): current-N minimizers.
  The GUE clustering statistics still control rarity (the observed compound
  cluster spans 0.27–0.55 match s₂ ≈ N^{−1/8} ≈ 0.33), but the eigenvalue
  functional is c(δ)-shaped rather than s^{2k}, and the eigenvector signs
  follow the sign structure of c rather than alternating.
- **Quadratic regime** (span ≪ 1/log p_max): the stencil channels of §1,
  directly confirmed by Test A. At N = 8000 only the very tightest pairs
  (e.g. the Lehmer pair, δ = 0.0377 < 0.069) are in this regime; the
  quadratic estimate for the Lehmer pair (λ ≈ 0.07–0.11, cf. the measured
  interlacing bound 0.113) loses to the resonant cluster's 0.030 —
  consistent with a channel crossover at much larger N (the coefficient
  ratio enters at the 6th power: N* ~ (a_res/a₁)⁶).

The asymptotic prediction (pair channel, N^{−2/3}) is untouched, but the
exponents measured at accessible N (−0.47…−0.53) should be read as
*crossover* values, not asymptotic ones — their numerical agreement with
the k = 2 channel's −1/2 is suggestive but not conclusive.

### Test C: GUE ensemble (8 seeds × N = 1000…8000)

| N | ⟨λ_min⟩ | median λ_min | ⟨κ⟩ | ⟨PR⟩ |
|------|---------|---------|--------|------|
| 1000 | 0.0814 | 0.0819 | 32.0 | 4.18 |
| 2000 | 0.0522 | 0.0561 | 56.8 | 3.68 |
| 4000 | 0.0323 | 0.0348 | 117.5 | 3.11 |
| 8000 | 0.0250 | 0.0271 | 134.4 | 3.50 |

**Ensemble exponents: mean λ_min ~ N^−0.579, median ~ N^−0.548** —
bracketed by the triple channel (−0.50) and the pair channel (−0.667),
and steeper than the single-realization real-zero fits. Read through the
two-regime lens of Test B, this is the crossover in action, drifting in
the predicted direction (toward −2/3).

**PR drifts downward** (4.18 → 3.1–3.5), the direction predicted by the
asymptotic pair-channel takeover, though noisy at 8 seeds.

**The W₃ tail test is inconclusive by construction**: the fitted
α ≈ 5.1 at quantiles 5×10⁻⁴…2×10⁻² corresponds to window eigenvalues
t ≈ 0.07–0.16, whose minimizing clusters have spans ~0.3 — resonant
regime, where the functional is much flatter in span and the tail
correspondingly steeper. The asymptotic quadratic-regime tail laws
(t^{3/2} pair / t² triple) live at depths this sample (32k windows)
cannot reach. A dedicated large-ensemble tail experiment targeting only
sub-0.07-span clusters would be needed.

## 4. What a proof would require

1. **Stencil identity** (§1): already rigorous; the Taylor remainder is
   elementary to bound. Lean-formalizable with finite effort (it is
   polynomial algebra plus the exact identity (∗), which is the
   already-formalized Gram factorization idea).
2. **Locality**: λ_min is attained (up to a constant) by a window of
   bounded size. Empirically true in the well-resolved regime (fixed-P
   sweep: W₃₂/λ_min → 1.04 at 160 primes/zero); a proof would need a
   quantitative version of the delocalization decay.
3. **GUE clustering input**: P(span of k+1 zeros < s) ~ s^{(k+1)²−1}.
   For genuine zeta zeros this is conjectural (GUE Hypothesis); for GUE
   matrices it is a theorem (and the smallest-gap extreme value for k = 1
   is Ben Arous–Bourgade). A theorem of the form "GUE Hypothesis ⟹
   λ_min ~ N^{−2/3+o(1)}" looks achievable with current technology.
4. **Channel competition**: show the k = 1 channel dominates asymptotically
   (moment/coefficient bookkeeping plus the resonant-regime cap).

The realistic theorem target: **conditionally on the GUE Hypothesis for
zeta zeros, the trace-normalized prime-zero Gram matrix on N zeros (with
primes-per-zero → ∞) satisfies λ_min(N) = N^{−2/3+o(1)}, hence
κ(N) = N^{2/3+o(1)} · λ_max(N).** The λ_max factor (numerically ~ N^{0.11})
needs separate (Toeplitz-symbol) treatment.
