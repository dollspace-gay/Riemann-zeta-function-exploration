# Plan 4: Theorem 1 Machine-Verified End to End

*Current state: the analytic core (`disagreement_integral_le` in
`NymanBeurling.lean`) is done — zero sorries, standard axioms. What
remains on paper is the L² bookkeeping and the Rayleigh connection. This
plan closes that gap, making Theorem 1 the first result of this project
— and possibly of any layman+AI collaboration — where a genuinely new
theorem's ENTIRE proof is machine-checked. This is the heaviest Lean
project in the repo so far; it is staged so every milestone is a
committable artifact even if a later one stalls.*

## What must be formalized

Paper chain (chains_theory.md §0):

1. e_k(t) = ρ(1/(kt)) is in L²(0,∞).                        [M1]
2. Square-wave identity: e_{2k} − e_k/2 = (1/2)·𝟙{⌊1/(kt)⌋ odd},
   hence f²_k supported on (0, 1/k], ‖f²_k‖² = C₂/k.        [M2]
3. The witness w = f_{K−1} − f_K (a 4-term integer combination of e's)
   has ‖w‖² = P/8 where P = 2∫_D u⁻²du is the disagreement
   integral — the change of variables u = 1/t and the bilinear
   expansion.                                               [M3]
4. Rayleigh: λ_min(A_N) ≤ ‖Σx_k e_k‖²/‖x‖² with ‖x‖² = 5/2,
   assembled with `disagreement_integral_le` to give
   ∃ eigenvalue of A_N ≤ (H_{K−1}+1)/(10·K(K−1)).           [M4]

## Milestone 1: the dilation family lives in L²

- Define `e (k : ℕ+) : ℝ → ℝ := fun t => Int.fract (1/(k·t))` on the
  measure `volume.restrict (Ioi 0)`.
- Measurability: `Int.fract` composed with measurable — Mathlib has
  fract measurability (the floor measurability idiom is already used in
  `D_measurable`).
- MemLp 2: dominate |e_k| by min(1, 1/(k·t)): bounded by 1 on (0, 1],
  and ≤ 1/(k·t) on (1, ∞) whose square integrates via the existing
  `integrableOn_inv_sq_Ioi`. Both pieces are already in our toolbox.
- Package as `MeasureTheory.Lp ℝ 2` elements (needed for the inner
  product space structure `L2.innerProductSpace`).

Committable artifact: `e_k ∈ L²` + basic lemmas. Modest.

## Milestone 2: the square-wave identity in L²

- Pointwise identity first (pure `Int.fract`/`Int.floor` algebra — the
  paper proof is four lines; expect the Lean to be the usual 10× but
  routine).
- Norm computation ‖f²_k‖² = (log 2)/(4k): after u = 1/t this is the
  integral of u⁻²/4 over {⌊u/k⌋ odd} — the SAME shape as the already-
  formalized disagreement machinery (odd-floor sets, interval unions,
  u⁻² integrals). The alternating series Σ 1/(j(j+1)) over odd j
  telescoping to log 2 needs Mathlib's `Real.log` series or an
  integral-comparison derivation — scout Mathlib for
  alternating-harmonic = log 2 before hand-rolling.
- **Scope decision:** M2's norm formula is not strictly needed for M4
  (Theorem 1 uses the disagreement bound directly, not C₂/k). It IS the
  natural first exact constant and a self-contained lemma. Do the
  pointwise identity as part of M3's requirements; treat the C₂ norm as
  a detachable bonus.

Committable artifact: pointwise square-wave identity (+ optionally C₂).

## Milestone 3: the witness norm equals the disagreement integral

- Change of variables t ↦ 1/u on (0,∞): Mathlib's
  `integral_comp_inv`-style lemmas / `MeasurePreserving` machinery —
  scout what exists for x ↦ x⁻¹ pushforward with Jacobian; this is the
  most likely place to hit missing-Mathlib friction. Fallback: prove the
  specific identity ∫₀^∞ g(1/t)dt = ∫₀^∞ g(u)u⁻²du for our nonneg
  measurable g directly.
- Bilinearity bookkeeping: ‖f_{K−1} − f_K‖² expanded so that
  (ε_K − ε_{K−1})² = 4·𝟙{parity disagreement} — squares of ±1 functions;
  finite algebra a.e.
- Land exactly on `∫_D u⁻²` with D as ALREADY DEFINED in
  `NymanBeurling.lean` (definitions must be literally shared, not
  parallel — refactor D's definition into a common location if needed).

Committable artifact: `witness_norm_eq : ‖w‖² = (1/4)·∫_D u⁻²` (or the
P/8 form; pick one normalization and note it against the paper's).

## Milestone 4: assembly

- The finite Gram matrix A_N of {e_k} with entries ⟨e_m, e_n⟩ (real
  inner product on Lp). Its Hermitian-ness: inner product symmetry.
- The quadratic-form identity xᵀA_N x = ‖Σx_k e_k‖² (finite sum
  bilinearity — same structure as `gram_entry` handling in
  RHCrystal.lean).
- Apply the existing `exists_eigenvalue_mul_le_rayleigh` with the
  4-support witness vector x (entries −1/2, 1, 1/2, −1; ‖x‖² = 5/2;
  indices distinct needs K ≥ 3 arithmetic), then chain with
  `disagreement_integral_le` (J = K−1, needs J ≥ 2 ⟺ K ≥ 3 —
  hypotheses already match).
- Final statement, mirroring the paper:

      theorem nb_gram_eigenvalue_le (hK : 3 ≤ K) (hN : 2*K ≤ N) :
        ∃ i, eigenvalues (nbGram N) i ≤ (harmonic (K-1) + 1)/(10*K*(K-1))

Committable artifact: **Theorem 1, end to end, zero sorries.** Update
`#print axioms`, RESULTS.md, paper §0, and the stale `lean/README.md`
(which still lists a long-closed sorry and a nonexistent Basic.lean —
fix it in this milestone's commit at the latest).

## Risks and scouting order

Before writing anything, spend one scouting pass in Mathlib on the three
friction candidates: (1) x ↦ 1/x change of variables on (0,∞); (2)
alternating harmonic series = log 2; (3) Lp inner-product-to-integral
unfolding ergonomics. The rest is patterns this repo has already
executed. If (1) is genuinely absent from Mathlib, M3's fallback keeps
the project moving but expect it to dominate the effort.

## Optional extension (separate, smaller)

The k×k window interlacing bound (v4 open problem 6's leftover):
principal-submatrix interlacing in general. Independent of M1–M4;
worthwhile only if the main chain lands with energy to spare.

## Effort estimate

M1: 1 session. M2: 1 session (pointwise) + optional C₂. M3: 1–2 sessions
(the change-of-variables risk). M4: 1 session. Total realistic: 4–6
sessions of Lean work, each ending in a committable state.
