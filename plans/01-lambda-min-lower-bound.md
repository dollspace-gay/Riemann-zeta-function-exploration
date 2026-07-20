# Plan 1: The Matching Lower Bound — λ_min(G_N) ≳ N⁻²

*Target: upgrade `chains_theory.md` §5 from program to theorem. Combined
with Theorem 1 (λ_min ≤ (H_{K−1}+1)/(10K(K−1))), this gives
λ_min(G_N) ≍ N⁻²·(slowly varying) and κ(G_N) ≍ N² — a deterministic,
arithmetic conditioning law for the Nyman–Beurling dilation basis.*

## Current state

- **Proved:** the square-wave identity, ‖f²_k‖² = C₂/k with C₂ = (log 2)/4,
  the zero-sum reduction (2), Theorem 1 (upper bound), and the exact kernel
  decomposition M_{jk} = (log 2/8)(1/j + 1/k) − P_{jk}/16.
- **Measured:** the chain subspace attains λ_min within a stable factor
  1.19–1.23 (N = 500…2500), saturating at ~20 chains; the kernel
  P_{jk}·K²/(2d) = log(K/d) + γ + c(d) with c(d): 0.545 → 0.661 over
  d = 1…32, drifting toward log 2; λ_min·N² ≈ 0.50 + 0.134·log N.
- **Open:** everything below.

## Step 0 (warm-up, mechanical): derive c(d) exactly

The parity-disagreement set of ⌊u/j⌋, ⌊u/k⌋ for general d = k − j is a
union of intervals whose endpoints are multiples of j and k — Beatty /
three-distance combinatorics, heavier than the d = 1 telescoping but the
same kind of enumeration. Targets:

1. Closed form (or explicit asymptotic with error term) for
   P_{jk} = (2d/K²)(log(K/d) + γ + c(d)) + O(·), c(d) explicit.
2. Consistency anchors: c(1) must reproduce Theorem 1's exact harmonic
   sum (measured 0.545); c(d) → log 2 as d → ∞ must fall out of the
   alternating tail Σ_r[1/(2r+1) − 1/(2r+2)] = log 2.

**Validation gate:** derived c(d) vs the exact enumeration at K = 1250
(`chains.py` machinery) for d = 1…64, to ≥ 4 digits. Also cross-check
selected P_{jk} against Vasyunin's closed-form Gram entries if Plan 3's
cross-check lands first (the two plans feed each other).

## Step 1: the max-kernel part on zero-sum vectors

M's first term (log 2/8)(1/j + 1/k) is rank-two-like. On zero-sum α over
a window (K−w, K], expand 1/j = 1/K + (K−j)/K² + O(w²/K³): the constant
kills against Σα = 0, leaving a ±|j−k|-type variation kernel of the same
O(d/K²) size as the P-part. Do this bookkeeping exactly, with the O(w²/K³)
error carried explicitly. Output: αᵀMα = (1/K²)·αᵀΨα + error, with
Ψ_{jk} = ψ(|j−k|) fully explicit from Step 0.

**Validation gate:** evaluate αᵀΨα/K² against αᵀMα computed from the
exact Gram data for the *measured* optimal α (saved from `chains.py`),
at N = 500…2500. Agreement should be at the level of the carried error
term, not "roughly."

## Step 2 (the key lemma): quantitative conditional positivity of Ψ

Expected shape ψ(d) = a·d − b·d·log d (+ bounded corrections from c(d)).
Two ingredients:

- The −|j−k| part: on zero-sum α with partial sums S_m (S_w = 0), the
  exact identity αᵀ(−|i−j|)α = 2Σ_m S_m², and
  ‖α‖² = Σ(S_m − S_{m−1})² ≤ 4ΣS_m², give
  αᵀ(−|i−j|)α ≥ ‖α‖²/2 — *w-independent*, elementary, already checkable.
- The +d·log d part (the delicate one): d^s is conditionally negative
  definite for 0 < s ≤ 2 (recalled — verify against a source before use);
  d log d is the boundary derivative case and its sign contribution must
  be controlled, not assumed. Candidate routes: an integral representation
  d log d = ∫ (d^s family) ds, or a direct spectral-density computation of
  the kernel on zero-mean sequences.

**Numerical pre-check before proving anything:** build the exact Ψ from
Step 0, restrict to the zero-sum subspace, and compute its smallest
eigenvalue as a function of window width w = 4…256. The lemma predicts it
is bounded below by a positive constant independent of w. If this *fails*
numerically, the lemma is false as stated and the w-dependence must enter
the theorem — find that out for the cost of an afternoon, not a month.

**Output if it holds:** λ_chain ≥ c/K², hence with Theorem 1,
λ_chain ≍ K⁻²·(slowly varying) on the chain subspace. This alone is
**Theorem 2** — publishable-grade as a lemma about the NB basis even
before Step 3.

## Step 3 (hard, research-grade): extend from the chain subspace to G_N

The measured gap is a stable factor 1.19–1.23, but "the chain subspace
nearly attains λ_min" is an empirical fact, not a theorem. Candidate
approaches, in order of plausibility:

1. **Schur complement / projection:** split L²-span{e_k} into the chain
   subspace and a complement; show the complement's Rayleigh quotients are
   ≫ N⁻² (the complement should look like "generic" dilation directions,
   whose Gram floor is governed by shorter chains or no chains at all).
2. **Mellin route:** any lower bound on the full quadratic form
   xᵀAx = (1/2π)∫|ζ(½+it)|²/(¼+t²)·|X(t)|²dt over unit x subsumes this
   step entirely — see Plan 5. If Plan 5's Milestone 2 lands at N^{−3−ε},
   it does not close this step, but its machinery (coefficient
   uncertainty lemma) may adapt.
3. **Fallback:** state Theorem 2 (chain subspace) + the measured factor
   1.2 as a documented conjecture for the full matrix.

## Honest expectations

Steps 0–1 are mechanical and will land. Step 2 is a real lemma with a
cheap falsification test — do the numerical pre-check first. Step 3 may
not close this year; the fallback is still a clean result. Throughout:
every recalled fact about conditionally negative definite kernels gets
re-derived or source-checked — recalled constants have burned us twice.

## Deliverables

- `chains_theory.md` §5 rewritten with Theorem 2 (+ proof) and the Step 3
  status; RESULTS.md session entry with all validation tables.
- Lean candidate afterwards: the ΣS_m² identity and the −|d| positivity
  are finite algebra, well within existing formalization style.

## Effort estimate

Step 0: 1 session. Steps 1–2: 1–2 sessions including numerics. Step 3:
open-ended; timebox exploratory attempts to a session each.
