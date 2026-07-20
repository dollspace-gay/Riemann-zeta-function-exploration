# Plan 8: The Redheffer Matrix — RH as a Determinant Growth Law

*July 2026, from doll's question: "is there a dusty-corner statement
whose truth RH determines, and which points back at RH?" The Redheffer
matrix is the candidate best matched to this repo's instruments.
GATED: Part 0 (literature check) decides whether Parts A–B proceed and
in what form. Difficulty-conservation warning applies in full: the
equivalence carries RH's entire weight; the bet is only that our
particular toolkit (spectral-floor forensics, null-vector arithmetic,
exact identities, scaling laws) has not been pointed at it.*

## The object

A_n is the n×n 0-1 matrix with A[i,j] = 1 iff j = 1 or i | j
(Redheffer 1977). Classical facts (recalled — Part 0 verifies sources):

- det(A_n) = M(n) = Σ_{k≤n} μ(k), the Mertens function.
- **RH ⟺ M(n) = O(n^{1/2+ε}) ⟺ a growth law on these determinants.**
- Structure: A = T + (𝟙 − e₁)e₁ᵀ where T[i,j] = [i | j] is unipotent
  (det 1) and T⁻¹ is the Möbius matrix [i|j]·μ(j/i) — so A is a
  rank-one update of the divisibility lattice, and the determinant
  formula det(A) = 1 + e₁ᵀT⁻¹(𝟙 − e₁) = Σ_{k≤n}μ(k) drops out of the
  matrix determinant lemma. Mertens as a rank-one perturbation
  determinant: exactly our kind of object.
- Recalled spectral folklore (verify): ~n − ⌊log₂ n⌋ − 1 eigenvalues
  equal to 1; two dominant real eigenvalues ≈ ±√n; a small complex
  cloud carrying the arithmetic (Barrett–Jarvis; Vaughan I–II).

## Part 0 — the gate: literature check

Establish, with sources cited in LITERATURE.md style:
1. What is proven about the eigenvalues (dominant asymptotics, the
   1-eigenvalue count, the nontrivial cloud)?
2. Is there ANY published work on the **singular values** / smallest
   eigenvalue / conditioning of A_n?
3. Any published analysis of **near-null vector structure** (the
   arithmetic of the directions that make det small)?
4. Any spectral reformulation of the RH-relevant det growth?

Proceed only where the literature stops. If items 2–3 are virgin
territory (the expectation), Parts A–B go ahead. If not, read first.

## Part A — the instrument pass (exploratory, standards as always)

1. **Exact anchors before anything**: verify det(A_n) = M(n) via the
   rank-one identity (exact integer sieve vs the formula — NOT via
   float LU, which cannot see an O(√n) determinant under roundoff at
   large n); verify the 1-eigenvalue count law numerically.
2. **Full spectrum + singular values** to n ~ 4096 (dense, GPU for the
   top sizes): eigenvalue cloud geometry, σ_max and σ_min scaling laws
   with the repo's fitting discipline, condition number growth.
3. **Near-null forensics**: the arithmetic profile of the smallest
   singular vectors — support on divisor chains? smooth numbers?
   primes? (Directly comparable to the NB doubling-chain discovery —
   same lattice, different matrix.)
4. **Fluctuation coupling**: how do spectral observables co-move with
   M(n) as n increments (eigenvalues are continuous-ish in n; M(n)
   jumps by μ(n))? Where in the spectrum does the Mertens information
   live?

## Part B — the RH-facing question (only if A finds structure)

det(A_n) = M(n) is a product over the spectrum; the many unit
eigenvalues contribute nothing; the dominant ±√n pair contributes
−n-ish; so the small cloud must conspire to divide it back down to
O(n^{1/2+ε}) — under RH, an extraordinary cancellation *in the
spectrum itself*. Characterize that cancellation empirically; ask
whether any of our positivity/floor machinery constrains it. Honest
expectation: description, not theorems — but a genuinely new
description would already justify the plan.

## Risks

- Literature may already cover more than expected (the gate exists for
  this).
- The det-vs-spectrum gap: bounding |det| needs ALL the spectrum, not
  a floor — our lower-bound machinery may have no purchase (flagged
  now so a null result reads as an answer, not a failure).
- Difficulty conservation: nothing here evades RH's weight.

## Effort

Part 0: one search session. Part A: 1–2 sessions (matrix builds are
trivial next to what this repo already runs). Part B: open-ended,
timeboxed.
