# Plan 3: The Residual Profile r_N(t) + the Vasyunin Cross-Check

*Two bounded computations that have been on the session lists since
Session 1 and never got done. Both are cheap; both have structural
payoff; the second also retires the biggest flagged unknown in
LITERATURE.md before any outside contact.*

---

## Part A: where does the unapproximable mass live?

The residual r_N(t) = χ(t) − Σ_{k≤N} c_k e_k(t) has ‖r_N‖² = d_N².
Nobody (in this repo, and per the literature check, possibly anywhere)
has looked at its *profile*.

### Exact structure to exploit

- **Tail is exact:** for t > 1, χ = 0 and every e_k(t) = 1/(kt), so
  r_N(t) = −(Σ c_k/k)/t exactly, and the tail mass is (Σ c_k/k)².
  The scalar S_N = Σ c_k/k is itself a diagnostic: for Möbius-shaped c
  it is a weighted Σμ(k)/k truncation, which should → 0 — compare its
  measured N-decay against the known Σ_{k≤N} μ(k)/k ≪ decay (recalled —
  verify) as a free side-experiment.
- **On (0, 1]:** evaluate Σc_k e_k(t) on a dense t-grid. Breakpoints of
  e_k are at t = 1/(k·m), dense near 0 — use a log-spaced grid in t plus
  exact handling near breakpoints where needed. Cancellation between
  O(N) terms with O(1) coefficients: float64 first, mpmath spot checks at
  ~20 grid points to bound evaluation error.

### The validation gate (non-negotiable before interpreting anything)

**Energy closure:** grid-quadrature of ∫₀¹ r_N² dt plus the exact tail
must reproduce d_N² = 1 − bᵀG⁻¹b from the linear algebra to the accuracy
of the quadrature. This single test validates the coefficients, the
pointwise evaluation, and the grid at once. No closure → no analysis.

### Questions the profile answers

1. **Concentration:** define m_N(τ) = ∫₀^τ r_N²/d_N² and track the
   median-mass point τ_½(N). Does it scale like 1/N (the missing-high-
   dilation picture: e_k with k > N can't be used, and those live at
   t ~ 1/k), like a power of 1/log N, or not at all (spread)?
2. **Shape:** is r_N locally sawtooth-like (pointing at specific missing
   dilations) or smooth (pointing at a density obstruction)? Compare
   against the naive prediction r ≈ the Möbius-truncation error of
   Σ_{k≤x}μ(k)⌊x/k⌋ = 1 restated in t = 1/x coordinates.
3. **Connection to the deviation field:** Session 2 found the deviation
   δ_k = c_k + μ(k)(1 − log k/log N) concentrates on smooth squarefree k.
   Does the residual profile correlate with where those smooth-number
   dilations live in t? (Overlay test.)
4. **N-evolution:** profiles at N = 100, 500, 2500, 10⁴ (all Gram data
   cached) — does the profile converge in shape after rescaling by
   τ_½(N)? A stable rescaled shape is a structural finding.

Deliverable: `nyman-beurling/residual.py`, profile figures, RESULTS.md
entry answering 1–4 with numbers.

---

## Part B: the Vasyunin cross-check

LITERATURE.md flags Vasyunin (1996) as the most likely prior home of our
closed forms: he gives explicit formulas for ⟨e_m, e_n⟩ via cotangent
sums. We have never compared. This is both an independent validation of
the entire Gram scheme and the honest way to resolve the "A(1,2)
rederived?" question.

### Protocol (recalled-knowledge discipline applies in full)

1. **Do not implement a remembered formula.** Obtain the actual statement
   (paper if accessible; secondary sources with the formula quoted
   otherwise; worst case, formulas reproduced in the BDBLS/Balazard
   survey literature). If no source can be obtained verbatim, STOP and
   record that — do not fill the gap from memory.
2. Implement whatever formula the source states, exactly, with its
   normalization conventions made explicit in comments (their inner
   product may differ from ours by the L²(0,∞) vs L²(0,1) split or a
   variable change — derive the dictionary, don't guess it).
3. **Cross-check table:** their formula vs our exact A_{mn} for ~50 pairs
   spanning coprime/non-coprime, small/large, near-diagonal/far
   (we hold A to ~1e−11 against brute force, so any real discrepancy is
   theirs-vs-ours normalization, findable and fixable).
4. Specialize their formula to (m,n) = (1,2): does
   A(1,2) = (3/4)(log 2π − γ) − (log 2)/4 drop out? Record the outcome
   in LITERATURE.md either way ("special case of Vasyunin, as suspected"
   or "not directly recoverable — genuinely independent derivation").

### Payoffs beyond validation

- If the cotangent form is confirmed, it may give **exact evaluations of
  the P_{jk} disagreement integrals** — feeding directly into Plan 1
  Step 0 (c(d) exact) from a second, independent direction.
- Updates the calibrated claim level for any eventual outreach (upgrade
  "presumably recoverable from Vasyunin" to a checked statement).

Deliverable: `nyman-beurling/vasyunin_check.py`, the cross-check table in
RESULTS.md, LITERATURE.md updated with the resolved flag.

---

## Effort estimate

Part A: one session including validation. Part B: one session, dominated
by getting the formula's exact statement and normalization dictionary
right. No dependencies on other plans; both feed Plans 1 and (eventually)
any outreach.
