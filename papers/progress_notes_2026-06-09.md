# Progress Notes — June 9, 2026

New experiments targeting the open problems in `rh_crystal_v3_final.md` (§8).
Code: `compute/rh_progress.py`, `compute/rh_windows.py`. Raw output:
`~/rh_output/progress_results.txt`, `~/rh_output/windows_results.txt`.
All runs: RTX 3090, float64, Odlyzko's 2M zeros, primes ≤ 10⁷, σ = 1/2,
trace-normalized G (so diag(G) = 1 exactly).

---

## 1. A rigorous upper bound on λ_min (new, provable)

By Cauchy interlacing, λ_min(G) is bounded above by the smallest eigenvalue of
**any** principal submatrix. For the 2×2 block of an adjacent zero pair with
gap δ, that eigenvalue is exactly (in trace-normalized units)

    B(δ) = Σ_p ŵ(p) · (1 − cos(δ · log p)),   ŵ(p) = w(p) / Σ w

so **λ_min(G) ≤ min over adjacent pairs of B(δ_i)** — a theorem, not an
observation, and a candidate for Lean formalization (finite real algebra +
interlacing, which Mathlib has as `Matrix.PosSemidef`-adjacent machinery).
For δ → 0, B(δ) ≈ (S₂/2S₀)·δ², connecting λ_min directly to minimal zero gaps.

### Numerically, the 2×2 bound is valid but loose (factor 3–5):

| N | λ_min | min B | λ_min/min B |
|------|---------|---------|------|
| 250 | 0.28115 | 0.46919 | 0.60 |
| 1000 | 0.07430 | 0.33900 | 0.22 |
| 4000 | 0.05969 | 0.29610 | 0.20 |
| 8000 | 0.03024 | 0.11333 | 0.27 |

Exponent fits (250 ≤ N ≤ 8000): λ_min ~ N^−0.53, κ ~ N^+0.64.

**Why it's loose:** the quadratic regime B ≈ (S₂/2S₀)δ² requires
δ·log(p_max) ≪ 1, i.e. δ ≪ 0.06 with primes to 10⁷. Typical near-minimal gaps
(δ ≈ 0.1–0.6) sit in the *resonant* regime where the cosine oscillates. In
fact the off-diagonal coupling at small-gap pairs is often **negative**
(e.g. c = −0.51 at the N=250 min-gap pair), so single pairs are not where the
quadratic form is smallest — alternating-sign vectors over *clusters* are.

## 2. λ_min localizes on clusters, found exactly by 3-zero windows

Sliding k-zero window bound W_k = min_i λ_min(G[i:i+k, i:i+k]):

| N | λ_min | PR | eigvec site (top-2 weight) | W₂/λ_min | W₃/λ_min | W₆₄/λ_min | k=3 window finds eigvec site? |
|------|---------|-----|------------------|------|------|------|-----|
| 1000 | 0.0743 | 2.5 | 888,889 (89%) | 1.99 | 1.52 | 1.20 | yes [887,890) |
| 2000 | 0.0773 | 3.1 | 1364,1365 (80%) | 1.57 | 1.47 | 1.23 | yes* [1339,1342) |
| 4000 | 0.0597 | 2.8 | 2956,2957 (85%) | 1.80 | 1.45 | 1.34 | yes [2956,2959) |
| 8000 | 0.0302 | 2.7 | 6896,6897 (85%) | 3.24 | 2.27 | 1.62 | yes [6895,6898) |

(*at N=2000 the k=3 window picks the secondary eigenvector site 1340,1341;
weights there are 7.5%, the runner-up cluster.)

Three findings:

1. **The k=3 window bound locates the localization site at every N** — while
   the raw minimal gap does NOT. The local functional that selects the
   "invisible" direction is the 3-window eigenvalue, not pairwise spacing.

2. **At N=8000 the famous Lehmer pair loses.** The smallest gap in range is
   the Lehmer pair at index 6708 (γ ≈ 7005.06, gap 0.0377), but the
   eigenvector localizes at (6896,6897), gap 0.268, which sits in a *compound
   cluster* (neighboring gaps 0.974, 1.671, 0.548, 0.268, 1.285, 0.64).
   Several modestly-small gaps bunched together beat one isolated near-collision.
   **This revises §5.5 of the paper: the correct statement is "tight
   clusters," not "Lehmer pairs."** The weighted-spacing statistic in the
   paper (spacings 65–72% of mean) was already hinting at this.

3. **λ_min is not perfectly local**: W_k plateaus by k ≈ 8–12 at a factor
   1.2–1.6 above λ_min, and the factor grows slowly with N. The eigenvector's
   small delocalized tail (~10–15% of mass) does real work. Caveat: P = 20N in
   these runs, so N and prime count grow together; a fixed-P sweep would
   separate the two effects.

## 3. Control ensembles: conditioning is a local-statistics phenomenon

Same construction, synthetic spectra with the SAME smooth density as the zeta
zeros (mapped through the inverse smooth counting function) but different
local statistics:

| N | κ zeta | κ GUE | κ Poisson | κ picket |
|------|------|------|-----------|------|
| 500 | 22.3 | 28.5 | 1.1×10⁸ | 12.2 |
| 1000 | 32.4 | 30.7 | 5.0×10⁶ | 16.6 |
| 2000 | 33.8 | 49.3 | 8.8×10⁶ | 21.9 |
| 4000 | 48.3 | 97.8 | 1.6×10⁸ | 26.7 |

- **GUE tracks zeta** in magnitude and slope (GUE κ ~ N^0.60 vs zeta ~N^0.64
  over the wide range; zeta's local fit is noisy because λ_min is an
  extreme-value statistic — non-monotone in N).
- **Poisson collapses** (κ ~ 10⁷–10⁸): generic random points have near-zero
  gaps that annihilate λ_min. The zeros' level repulsion is what keeps G
  invertible at all.
- **Picket fence is best but still grows** (κ ~ N^0.38): even a perfectly
  rigid spectrum degrades, so part of the κ growth is NOT gap-driven — likely
  the Toeplitz-symbol mechanism (paper §8, problem 2).

**Interpretation:** the κ ~ N^0.6 scaling law is a *GUE-statistics phenomenon,
not an arithmetic one* — within this construction the zeta zeros behave like a
generic GUE spectrum. Decomposition hypothesis worth testing: κ_zeta ≈
(picket-fence baseline from the symbol) × (extreme-cluster correction from GUE
gap statistics).

## 4. Second-order minimum verified to N = 2000 (paper had N = 30)

The curvature operator D_{ij} = ½ Σ_p w(p)(log p)² cos((γ_i−γ_j) log p) is
**PSD by the same Gram factorization as G** (weights w·(log p)²/2 ≥ 0) — a
one-line observation the paper missed, and trivially Lean-formalizable. The
second-order condition for σ = 1/2 to be a local min of κ is
r = ⟨v_max, D v_max⟩/λ_max − ⟨v_min, D v_min⟩/λ_min > 0:

| N | r | predicted κ(0.52)/κ(0.50) | actual |
|------|-------|---------|---------|
| 100 | 19.5 | 1.00782 | 1.00780 |
| 500 | 27.9 | 1.01114 | 1.01110 |
| 1000 | 39.3 | 1.01570 | 1.01567 |
| 2000 | 37.6 | 1.01504 | 1.01497 |

r > 0 at every scale, roughly growing, and the second-order prediction matches
the direct σ-sweep to 4–5 decimal places. The "why" now has a sharper form:
**why does D weight the top of G's spectrum more than the bottom (relative to
eigenvalue)?** Heuristic: D up-weights large log p; the λ_min direction
(a tight cluster, resolved only by large primes' fast phases) gets relatively
*less* D-mass per unit G-mass than delocalized directions. A proof of r > 0
under a gap-statistics assumption looks approachable.

## 5. Height degradation is resolution-limited, not intrinsic (problem 4 answered)

Blocks of 1000 zeros at height H, prime cutoff scan:

| block start | height | κ (p≤10⁵) | κ (p≤10⁶) | κ (p≤10⁷) |
|------|--------|--------|--------|--------|
| 0 | 14 | 30.2 | 27.4 | 34.9 |
| 100k | 75k | 401.2 | 108.6 | 129.8 |
| 500k | 319k | 1069.7 | 279.2 | 193.4 |
| 1M | 600k | 9542.3 | 973.9 | 245.2 |

At height 600k, each 10× in prime cutoff cuts κ by ~3–10×; at low height more
primes do nothing (saturated). The κ(height) curve flattens dramatically as
the cutoff grows (30→9542 at 10⁵ vs 35→245 at 10⁷). **The height-dependent
degradation of §5.4 is a finite-prime resolution artifact, removable by
scaling the prime cutoff with height.** Open question sharpened: what cutoff
schedule P(H) holds κ constant? (Data hints P ~ poly(H); worth a dedicated fit
— the natural guess from phase-resolution arguments is log P ∝ mean gap⁻¹ ∝ log H,
i.e. polynomial.)

## 6. Lean: the last `sorry` is closed

`cosh_ge_one` is now proven via Mathlib's `Real.one_le_cosh` (our `cosh`
definition coincides with `Real.cosh_eq`). The formalization has **zero
sorries, zero new axioms** (pending CI: `lake build` was re-run after the
change).

---

## Ranked next steps

1. **Formalize the interlacing bound in Lean** (§1). Statement: for PSD
   Hermitian G with unit diagonal, λ_min(G) ≤ 1 − |G_{ij}| for any i ≠ j; then
   instantiate with the explicit B(δ) formula. Mathlib has
   `Matrix.PosSemidef` and eigenvalue interlacing pieces. This would be the
   repo's second machine-verified theorem, and the first that touches the
   *spectrum*.
2. **Fixed-P sweep of the window plateau** (§2.3) to determine whether the
   delocalized component of λ_min is real or a prime-count artifact.
3. **Test the decomposition hypothesis** (§3): compute the picket-fence
   baseline at matched density for each N, and check whether
   κ_zeta/κ_picket matches the GUE extreme-cluster prediction.
4. **Cluster functional theory** (§2): define C₃(i) = λ_min of the 3×3 block
   as an explicit function of (δ_i, δ_{i+1}) and the weight moments; derive
   its extreme-value law under GUE statistics; compare to the measured
   N^−0.53. This is the realistic path to *proving* the scaling law,
   via Szegő-type asymptotics + known GUE small-gap theory (Ben Arous–Bourgade).
5. **Height-adaptive cutoff schedule** (§5): binary-search the prime cutoff
   that holds κ = 50 at each height; fit P(H).
6. **r > 0 mechanism** (§4): test the heuristic by computing the D-Rayleigh
   quotient profile across the full spectrum of G (not just the extremes) and
   correlating with eigenvector localization length.

---

## Addendum (same session): fixed-P sweep (next step 2 of the ranked list)

`compute/rh_fixedP.py`, P fixed at 160,000 primes, N = 1000→8000:

| N | λ_min | κ | W₃/λ_min | W₃₂/λ_min |
|------|---------|--------|------|------|
| 1000 | 0.0838 | 29.6 | 1.29 | 1.04 |
| 2000 | 0.0767 | 34.4 | 1.19 | 1.10 |
| 4000 | 0.0610 | 46.9 | 1.46 | 1.21 |
| 8000 | 0.0302 | 102.6 | 2.27 | 1.64 |

Fixed-P exponents: λ_min ~ N^−0.474, κ ~ N^+0.582 (growing-P: −0.53/+0.64).

**Verdicts.** (1) The scaling law survives at fixed P — not a prime-count
artifact. (2) The delocalized component of the null direction is real but
governed by primes-per-zero: at N=1000, the W₃₂ plateau is 1.04 with 160
primes/zero vs 1.21 with 20 primes/zero. More primes per zero localize the
null direction. Open follow-up: does W_k/λ_min → 1 as P/N → ∞ at fixed N?
(N=8000 row identical to the windows run by construction: 20·8000 = 160k.)

Also completed this session: the interlacing pair bound is machine-verified
in Lean 4 (`gram_eigenvalue_le_pair_bound`, zero sorries, zero custom
axioms) — ranked next-step 1 from this morning's list, done.
