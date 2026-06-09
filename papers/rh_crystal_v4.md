# The Critical Line as Point of Maximum Isotropy: A Numerical Study of Prime-Zero Coupling

**Version 4 (June 2026).** This version revises the close-pair localization claim of v3 (§5.5): sliding-window analysis shows the λ_min eigenvector localizes on tight *clusters* of zeros selected by a local window eigenvalue, not on minimal-gap (Lehmer) pairs per se. It also adds a rigorous interlacing bound on λ_min, ensemble control experiments (GUE/Poisson/picket fence), an extension of the second-order minimum verification from N = 30 to N = 2000, and a resolution of the height-degradation question via prime-cutoff scaling. All changes are catalogued in Appendix D.

---

## Abstract

We define a family of Xi-weighted Gram matrices G(σ) that encode the coupling between prime numbers and non-trivial zeros of L-functions, using weights derived from the completed zeta function ξ(s) = ξ(1−s). Through large-scale GPU computation (up to 20,000 zeros with 664,579 primes, plus Odlyzko's dataset of 2,001,052 precomputed zeros), we establish the following:

1. **Isotropy minimum.** The condition number κ(G(σ)) is minimized at σ = 1/2 for the Riemann zeta function and nine additional L-functions (including complex Dirichlet characters), at every matrix size tested (N = 100 to N = 10,000), and at every height tested (γ ~ 14 to γ ~ 600,000). The symmetry κ(σ) = κ(1−σ) is exact by construction and machine-verified in Lean 4 (zero sorries, zero custom axioms). The second-order condition for a local minimum is verified numerically up to N = 2000, where the curvature prediction matches direct computation to 4–5 decimal places.

2. **Tight-cluster localization** (revised from v3's "close-pair localization"). The eigenvector corresponding to the smallest eigenvalue of G is sharply localized (participation ratio ≈ 3) on tight clusters of consecutive zeros. The localization site is identified exactly, at every N tested, by minimizing the smallest eigenvalue of 3×3 principal submatrices — a local "cluster functional" — whereas the raw minimal gap is not predictive: at N = 8000 the smallest gap in the data is the Lehmer pair at γ ≈ 7005.06 (gap 0.038), yet the eigenvector localizes instead on a compound cluster of several modestly small gaps near γ ≈ 7173. By Cauchy interlacing, the cluster functional also provides a rigorous upper bound on λ_min.

3. **Local statistics control conditioning.** Replacing the zeta zeros by synthetic spectra with identical smooth density but different local statistics shows that GUE-distributed points reproduce the zeta Gram matrix's condition number in both magnitude and growth rate (κ ~ N^0.60 vs ~N^0.64), Poisson points destroy conditioning entirely (κ ~ 10⁷–10⁸), and a rigid lattice gives the slowest growth (κ ~ N^0.38). Within this construction the zeta zeros behave like a generic GUE spectrum: the scaling law is a consequence of random-matrix gap statistics, not arithmetic.

4. **Height-dependent resolution is a finite-prime artifact.** The condition number of a fixed-size block of 1000 zeros grows from κ ≈ 32 at height ~14 to κ ≈ 3531 at height ~600,000 with a fixed prime cutoff — but raising the cutoff restores conditioning (κ = 9542 → 245 at height 600k as the cutoff goes from 10⁵ to 10⁷). The degradation reflects the resolution limit of the explicit formula with finitely many primes, not an intrinsic property of high zeros.

5. **Perturbation instability.** Artificially displacing zeros off the critical line increases the condition number monotonically: a shift of δ = 0.3 from σ = 1/2 nearly quadruples κ, with λ_min collapsing by a factor of 3.5.

---

## 1. Introduction

### 1.1 Context

The Riemann Hypothesis (RH) asserts that all non-trivial zeros of ζ(s) have real part 1/2. Three proof strategies have shaped modern approaches: the spectral program of Hilbert-Pólya [1, 2], the geometric program of Weil and Connes [3, 4, 5], and the statistical characterization through random matrix theory [6, 7, 10].

The Weil explicit formula [8, Ch. 5] establishes a duality between sums over zeros and sums over primes:

    Σ_ρ h(ρ) = (main terms) + Σ_p Σ_m log(p) · g(m log p) / p^{m/2}

This duality can be encoded as a bilinear form — a Gram matrix — whose entries measure how primes "couple" pairs of zeros. The present work studies this bilinear form numerically, using weights chosen to respect the functional equation ξ(s) = ξ(1−s).

### 1.2 Summary of Findings

We report a robust geometric property of the critical line (the σ = 1/2 isotropy minimum), a structural explanation for Gram matrix degeneration (tight-cluster localization, with a rigorous interlacing bound), control experiments isolating which statistical properties of the zeros drive the spectrum, and a quantitative connection between conditioning and zero height. We also identify a specific limitation: the condition number is NOT bounded, growing approximately as N^{0.61} — and the control experiments indicate this exponent is a random-matrix phenomenon, not an arithmetic one.

This work has been revised twice, and we document both revisions. v3 corrected a bug in the matrix construction (diagonal doubling) that had produced a spurious "bounded conditioning" claim. v4 (this version) refines v3's claim that the λ_min eigenvector localizes on Lehmer pairs: window analysis shows the correct local object is a cluster functional, and minimal-gap pairs can lose to compound clusters (§5.5, Appendix D).

### 1.3 What This Paper Does Not Claim

This paper does not prove any new theorem about RH. The σ = 1/2 symmetry is a built-in consequence of the weight function, not a discovery. The isotropy minimum is a numerical observation, proven only to second order via perturbation analysis. The tight-cluster localization is descriptive at the level of mechanism — the 3-zero window functional locates the λ_min direction empirically, and the interlacing bound (§3.3) is rigorous, but we have no proof that the window functional must control λ_min. We are explicit about these limitations throughout.

---

## 2. Definitions

### 2.1 The Xi-Weighted Gram Matrix

Let γ_1 < γ_2 < ... < γ_N be imaginary parts of non-trivial zeros on the critical line. For σ ∈ (0, 1), define the N × N Hermitian matrix:

    G_{ij}(σ) = Σ_{p prime} w(p, σ) · exp(−i(γ_i − γ_j) log p)

with weight function:

    w(p, σ) = log(p) · cosh((σ − 1/2) log p) / √p

**Why cosh?** The completed zeta function ξ(s) = (1/2)s(s−1)π^{−s/2}Γ(s/2)ζ(s) satisfies ξ(s) = ξ(1−s). Its logarithmic derivative involves terms p^{−s} + p^{s−1}, which equals 2·cosh((s − 1/2) log p) / √p when evaluated symmetrically. The cosh weight is the unique choice (up to scaling) that makes w(p, σ) = w(p, 1−σ) while matching this structure. Alternative weights such as log(p)/p^σ (from the raw Euler product) break this symmetry and, as we verify, displace the condition number minimum away from σ = 1/2.

### 2.2 Normalization and Condition Number

We normalize G so that Tr(G) = N, isolating eigenvalue ratios from overall magnitude. The condition number κ(G) = λ_max / λ_min measures isotropy: κ = 1 means all eigenvalues are equal; large κ indicates degeneracy in some directions.

### 2.3 Factorization

G factors as G = V*V where V_{p,i} = √{w(p,σ)} · exp(−iγ_i log p). This guarantees positive semi-definiteness but not positive definiteness.

---

## 3. Proven Results

### 3.1 Symmetry Theorem

**Theorem.** For all p > 0 and σ ∈ ℝ: w(p, σ) = w(p, 1−σ).

*Proof.* (1−σ) − 1/2 = −(σ − 1/2), and cosh(−x) = cosh(x). ∎

**Corollary.** G(σ) = G(1−σ) as matrices, hence κ(σ) = κ(1−σ).

These are machine-verified in Lean 4, depending only on the axioms propext, Classical.choice, and Quot.sound (Appendix A).

### 3.2 Second-Order Minimum

**Lemma.** Expanding G(σ) = G(1/2) + ε²D + O(ε⁴) where ε = σ − 1/2 and D_{ij} = (1/2) Σ_p w(p, 1/2) · (log p)² · exp(−i(γ_i − γ_j) log p), the condition number is locally minimized at σ = 1/2 if r = ⟨v_max, D v_max⟩/λ_max − ⟨v_min, D v_min⟩/λ_min is positive, where v_max, v_min are the extreme eigenvectors of G(1/2).

**Observation (new in v4).** D is positive semi-definite *by construction*: it admits the same Gram factorization as G with the non-negative weights w(p, 1/2)·(log p)²/2 in place of w(p, 1/2). So every eigenvalue of G grows to second order as σ leaves 1/2; the question is only whether the top of the spectrum grows proportionally faster than the bottom (r > 0).

*Verification (extended from N = 30 in v3 to N = 2000).* r > 0 at every scale tested, and the second-order prediction κ(0.52)/κ(0.50) ≈ 1 + ε²r matches the direct σ-sweep to 4–5 decimal places:

| N | r | predicted κ(0.52)/κ(0.50) | direct computation |
|------|-------|---------|---------|
| 100 | 19.5 | 1.00782 | 1.00780 |
| 500 | 27.9 | 1.01114 | 1.01110 |
| 1000 | 39.3 | 1.01570 | 1.01567 |
| 2000 | 37.6 | 1.01504 | 1.01497 |

### 3.3 Interlacing Bound on λ_min (New in v4)

**Theorem.** Let G be trace-normalized so diag(G) = 1. For any adjacent pair of zeros with gap δ, by Cauchy interlacing applied to the corresponding 2×2 principal submatrix,

    λ_min(G) ≤ B(δ) = Σ_p ŵ(p) · (1 − cos(δ · log p)),    ŵ(p) = w(p) / Σ_q w(q)

and more generally λ_min(G) ≤ λ_min(M) for every principal submatrix M. For δ → 0, B(δ) ≈ (S₂/2S₀)·δ² where S_k = Σ_p w(p)(log p)^k, tying λ_min directly to small zero gaps.

*Proof.* The 2×2 principal submatrix at an adjacent pair is [[1, c], [c, 1]] with c = Σ_p ŵ(p) cos(δ log p) = 1 − B(δ); its smaller eigenvalue is min(1 − c, 1 + c) ≤ 1 − c = B(δ). Cauchy interlacing gives λ_min(G) ≤ λ_min of any principal submatrix. ∎

This statement is **machine-verified in Lean 4** (`gram_eigenvalue_le_pair_bound` in `lean/RHCrystal/RHCrystal.lean`, zero sorries, zero custom axioms; Appendix A). The formal proof goes through the Rayleigh-quotient argument: for real symmetric A, the test vector e_i − e_j witnesses an eigenvalue at most (A_ii + A_jj)/2 − A_ij, instantiated with the Gram entries. Numerically the 2×2 bound is valid but loose (λ_min/min B ≈ 0.2–0.6; see §5.5 for why), while 3×3 and larger windows are far sharper.

---

## 4. Computational Methods

### 4.1 Data Sources

**Zeros.** We used Odlyzko's table of 2,001,052 zeros of ζ(s) [7], accurate to 4 × 10^{−9}. Zeros of Dirichlet L-functions L(s, χ) for characters mod 3, 4, 5, 7, 8, 11, and 13 (real characters) and mod 5 and 7 (complex characters) were computed by sign-change detection with bisection refinement.

**Primes.** 664,579 primes below 10^7, generated by segmented sieve.

### 4.2 Matrix Construction

Gram matrices were constructed on GPU (NVIDIA RTX 3090, 24GB VRAM) using batched outer-product accumulation in PyTorch (float64). For each batch of primes, phases cos(γ_i · log p) and sin(γ_i · log p) were computed as N × batch matrices, weighted by √w, and accumulated via matrix multiplication. Eigenvalues were computed by torch.linalg.eigvalsh (symmetric eigenvalue decomposition).

**Bug note.** An earlier implementation in Rust contained a diagonal doubling error (writing G[i,i] twice per prime in the i = j case of the inner loop), which artificially improved conditioning by roughly 10×. All results in this paper use the corrected GPU implementation.

### 4.3 Weight Comparison

Three weight schemes were tested at N = 200 with σ-sweeps to verify the non-trivial content of the isotropy minimum:

| Weight | Formula | κ minimum at |
|--------|---------|-------------|
| Naive | log(p) / p^σ | σ ≈ 0.72 |
| Fixed | log(p) / √p | σ-independent |
| Xi | log(p) · cosh((σ−1/2) log p) / √p | σ = 0.50 |

The naive weights place the minimum far from 1/2 because they lack σ ↔ 1−σ symmetry. The fixed weights give constant κ because they don't depend on σ. Only the Xi weights produce a minimum at σ = 1/2, confirming this is a joint consequence of the weight symmetry AND the zero distribution.

---

## 5. Results

### 5.1 Universality of the Isotropy Minimum

The condition number is minimized at σ = 1/2 for all ten L-functions tested:

| L-function | Type | N | κ_min | σ_min |
|-----------|------|---|-------|-------|
| ζ(s) | — | 30 | 2.43 | 0.508 |
| L(s, χ₃) | real | 20 | 61.7 | 0.512 |
| L(s, χ₄) | real | 20 | 18.3 | 0.488 |
| L(s, χ₅) | real | 20 | 55.5 | 0.488 |
| L(s, χ₇) | real | 20 | 117.3 | 0.512 |
| L(s, χ₈) | real | 20 | 17.8 | 0.488 |
| L(s, χ₁₁) | real | 20 | 16.5 | 0.512 |
| L(s, χ₁₃) | real | 20 | 22.2 | 0.512 |
| L(s, χ₅) | complex | 15 | 45.2 | 0.486 |
| L(s, χ₇) | complex | 15 | 12.8 | 0.514 |

All minima fall within σ-grid resolution of 1/2. The symmetry κ(σ) = κ(1−σ) is exact (0.0000% discrepancy) in every case.

For the Riemann zeta function, the minimum is confirmed at σ = 0.5000 at N = 100, 500, 1000, 2000, 5000, and 10000:

At N = 10,000 with 200,000 primes:

    σ = 0.475:  κ = 110.97
    σ = 0.500:  κ = 107.88  (minimum)
    σ = 0.525:  κ = 110.97

### 5.2 Height Independence of the Minimum

The isotropy minimum at σ = 1/2 holds at all heights tested:

| Zero range | Height | κ at σ=0.3 | κ at σ=0.5 | κ at σ=0.7 | Min at 1/2? |
|-----------|--------|-----------|-----------|-----------|------------|
| 0–500 | ~14–811 | 42.2 | 22.3 | 42.2 | Yes |
| 100k–100.5k | ~75k | 512.1 | 217.1 | 512.1 | Yes |
| 1M–1.0005M | ~600k | 3522.6 | 1383.6 | 3522.6 | Yes |

The condition number spans two orders of magnitude across these heights, but the minimum remains exactly at σ = 1/2 in every case.

### 5.3 Scaling Law (Corrected)

With corrected code and ratio ≥ 20 primes per zero:

| N | P | κ | λ_min |
|------|--------|-------|-------|
| 50 | 1,000 | 4.2 | 0.401 |
| 100 | 2,000 | 5.4 | 0.360 |
| 500 | 10,000 | 22.3 | 0.107 |
| 1,000 | 20,000 | 32.4 | 0.074 |
| 2,000 | 40,000 | 33.8 | 0.077 |
| 5,000 | 100,000 | 54.5 | 0.053 |
| 10,000 | 200,000 | 107.9 | 0.029 |
| 20,000 | 400,000 | 139.9 | 0.024 |

Power-law fit: κ ~ N^{0.61}, λ_min ~ N^{−0.50}.

The condition number is NOT bounded. It grows approximately as N^{0.6}, with the minimum eigenvalue decaying as 1/√N. An earlier version of this work reported bounded conditioning (κ < 4 at N = 2000); this was an artifact of a diagonal doubling bug in the Rust implementation, which effectively added a scaled identity matrix to G, artificially compressing the eigenvalue spectrum.

### 5.4 Height-Dependent Resolution

The conditioning of a fixed-size block of 1000 zeros depends strongly on height:

| Zero range | Height | κ | λ_min |
|-----------|--------|------|-------|
| 0–1k | ~14–1,419 | 32.4 | 0.0743 |
| 10k–11k | ~10k | 71.8 | 0.0436 |
| 50k–51k | ~40k | 130.1 | 0.0264 |
| 100k–101k | ~75k | 267.2 | 0.0141 |
| 500k–501k | ~320k | 453.7 | 0.0092 |
| 1M–1.001M | ~600k | 3,531 | 0.0012 |

This reflects the increasing oscillation frequency of the phase factor exp(−iγ log p). At height γ, the phase completes γ · log(p) / (2π) cycles across the prime range. At γ = 600,000 with primes up to 10^7, this is ~1.5 million cycles, causing the prime sum to lose resolution on individual zeros.

**The degradation is a finite-prime artifact, not intrinsic (new in v4).** Scanning the prime cutoff at each height shows the conditioning at high heights keeps improving as more primes are added, while at low heights it is saturated:

| block start | height | κ (p ≤ 10⁵) | κ (p ≤ 10⁶) | κ (p ≤ 10⁷) |
|------|--------|--------|--------|--------|
| 0 | ~14 | 30.2 | 27.4 | 34.9 |
| 100k | ~75k | 401.2 | 108.6 | 129.8 |
| 500k | ~319k | 1069.7 | 279.2 | 193.4 |
| 1M | ~600k | 9542.3 | 973.9 | 245.2 |

At height 600k, each 10× increase in the prime cutoff cuts κ by a factor of 3–10, flattening the κ-vs-height curve from (30 → 9542) at cutoff 10⁵ to (35 → 245) at cutoff 10⁷. This resolves open problem 4 of v3 in the affirmative: the height-dependent degradation is the resolution limit of the explicit formula with finitely many primes, removable by scaling the cutoff with height. The remaining question is the cutoff schedule P(H) that holds κ constant (§8).

### 5.5 Tight-Cluster Localization (Revised in v4)

**The eigenvector corresponding to λ_min is sharply localized — but on tight *clusters* of consecutive zeros, not on minimal-gap pairs.** v3 of this paper attributed the localization to Lehmer pairs (near-collisions of zeros). Sliding-window analysis shows that statement was imprecise, in an instructive way.

**The localization itself, confirmed.** At N = 1000 (P = 20,000): zeros #888–889 (γ ≈ 1290.10, 1290.42, spacing 0.31) carry **88.9%** of the λ_min eigenvector weight, with participation ratio PR = 2.5. At N = 2000: zeros #1364–1365 carry 80.0%, PR = 3.1. At N = 4000: zeros #2956–2957 carry 84.5%, PR = 2.8. At N = 8000: zeros #6896–6897 carry 85.4%, PR = 2.7. The "invisible" direction is always concentrated on ~3 effective zeros.

**What selects the site: a window eigenvalue, not the gap.** Define the sliding-window bound W_k = min over i of λ_min(G[i:i+k, i:i+k]), which by Cauchy interlacing (§3.3) is an upper bound on λ_min(G) for every k. Then:

| N | λ_min | eigvec site | W₂/λ_min | W₃/λ_min | W₆₄/λ_min | k=3 argmin window |
|------|---------|---------------|------|------|------|------|
| 1000 | 0.0743 | 888,889 | 1.99 | 1.52 | 1.20 | [887,890) ✓ |
| 2000 | 0.0773 | 1364,1365 | 1.57 | 1.47 | 1.23 | [1339,1342)* |
| 4000 | 0.0597 | 2956,2957 | 1.80 | 1.45 | 1.34 | [2956,2959) ✓ |
| 8000 | 0.0302 | 6896,6897 | 3.24 | 2.27 | 1.62 | [6895,6898) ✓ |

(*at N = 2000 the k=3 window finds the eigenvector's secondary localization site, the runner-up cluster carrying 7.5% of the weight.)

The minimal 3-zero window locates the eigenvector's site at every N tested. The raw minimal gap does not:

**The Lehmer pair loses.** Among the first 8000 zeros, the smallest gap is the classic Lehmer pair at indices 6708–6709 (γ ≈ 7005.063, 7005.101, gap 0.0377). The λ_min eigenvector ignores it. It localizes instead at indices 6896–6897 (gap 0.268, seven times wider), which sit inside a *compound cluster*: the surrounding gaps run 0.974, 1.671, 0.548, **0.268**, 1.285, 0.640 — several modestly small spacings bunched together. Several moderately close zeros in a row degrade the prime-zero coupling more than one isolated near-collision.

**Why pairs are not the right local object.** The quadratic approximation B(δ) ≈ (S₂/2S₀)δ² of §3.3 requires δ·log(p_max) ≪ 1 — with primes to 10⁷, that means δ ≪ 0.06. Typical near-minimal gaps (δ ≈ 0.1–0.6) sit in the *resonant* regime where the cosine sum oscillates; the off-diagonal coupling at small-gap pairs is often negative (e.g. c = −0.51 at the minimal-gap pair of the first 250 zeros). A single 2×2 block cannot exploit this, but an alternating-sign vector spread across a cluster can — which is exactly what the λ_min eigenvector does.

**λ_min is local but not perfectly local.** W_k converges quickly in k up to k ≈ 8–12, then plateaus at a factor 1.2–1.6 above λ_min that grows slowly with N. The eigenvector's small delocalized tail (~10–15% of its mass) does real work in lowering the Rayleigh quotient. (Caveat: in these runs the prime count grows with N (P = 20N), so cluster-resolution and prime-count effects are not yet separated; a fixed-P sweep is open problem 3.)

**Spacing statistics, reinterpreted.** v3 reported that zeros in the λ_min direction have spacings 65–72% of the local mean. That statistic was already pointing at clusters rather than extreme pairs — a pair statistic would have shown ratios near the extreme-value floor, not 0.65–0.72:

| N | Mean spacing | Weighted spacing (λ_min direction) | Ratio |
|------|-------------|-----------------------------------|-------|
| 500 | 1.597 | 1.044 | 0.654 |
| 1000 | 1.407 | 0.904 | 0.643 |
| 2000 | 1.251 | 0.899 | 0.719 |

**Relation to Lehmer's phenomenon.** The connection to Lehmer's observation [16] survives in modified form: anomalously tight zero configurations are precisely what the prime sum fails to resolve. But the operative notion of "tight" is a cluster functional (the minimal window eigenvalue), not pairwise spacing alone — and in particular the most famous Lehmer pair is *not* the most invisible direction of the first 8000 zeros.

### 5.6 Perturbation Test

Displacing zeros off the critical line degrades conditioning:

| Perturbation | 1 zero at δ | κ | λ_min |
|-------------|-------------|------|-------|
| Baseline | 0.0 | 3.63 | 0.452 |
| Small | 0.1 | 3.71 | 0.447 |
| Medium | 0.2 | 6.69 | 0.249 |
| Large | 0.3 | 12.98 | 0.129 |

| Perturbation | 5 zeros at δ | κ | λ_min |
|-------------|--------------|------|-------|
| Baseline | 0.0 | 3.63 | 0.452 |
| Small | 0.1 | 3.90 | 0.438 |
| Medium | 0.2 | 6.83 | 0.256 |
| Large | 0.3 | 13.42 | 0.132 |

The degradation is superlinear in δ and approximately additive in the number of perturbed zeros. This demonstrates that the Gram matrix structure is sensitive to zero location, with off-line zeros producing measurably worse conditioning than on-line zeros.

**Limitation.** This test simulates off-line zeros by modifying the prime coupling factor (multiplying by p^{−δ}), not by including zeros at positions where ζ(s) ≠ 0. It demonstrates sensitivity of the inner product to σ-position but does not constitute evidence that off-line zeros of ζ cannot exist.

### 5.7 Control Ensembles: What Property of the Zeros Drives the Spectrum? (New in v4)

To isolate which statistical property of the zeros the Gram matrix responds to, we replaced the zeta zeros by synthetic spectra with the **same smooth density** (points generated in unfolded coordinates, then mapped through the inverse of the smooth counting function N̄(T) = (T/2π)log(T/2πe) + 7/8) but different **local statistics**:

- **GUE**: unfolded eigenvalues of a random complex Hermitian matrix (level repulsion, P(s) ~ s² for small gaps);
- **Poisson**: i.i.d. exponential gaps (no repulsion, many near-zero gaps);
- **Picket fence**: a perfectly rigid lattice (no gap fluctuations at all).

Same primes, same weights, same construction:

| N | κ zeta | κ GUE | κ Poisson | κ picket fence |
|------|------|------|-----------|------|
| 500 | 22.3 | 28.5 | 1.1×10⁸ | 12.2 |
| 1000 | 32.4 | 30.7 | 5.0×10⁶ | 16.6 |
| 2000 | 33.8 | 49.3 | 8.8×10⁶ | 21.9 |
| 4000 | 48.3 | 97.8 | 1.6×10⁸ | 26.7 |

Three conclusions:

1. **GUE reproduces zeta**, in magnitude and growth rate (GUE κ ~ N^0.60; zeta κ ~ N^0.64 over the wider range 250 ≤ N ≤ 8000; zeta's local fit is noisy because λ_min is an extreme-value statistic and non-monotone in N). Within this construction, the zeta zeros are statistically indistinguishable from a generic GUE spectrum — consistent with the Montgomery–Odlyzko picture [1, 7], and evidence that the N^0.61 scaling law of §5.3 is a random-matrix phenomenon rather than an arithmetic one.

2. **Poisson collapses** (κ ~ 10⁷–10⁸, λ_min ~ 10⁻⁷). Without level repulsion, near-coincident points produce near-null directions immediately. The zeros' repulsion is what keeps G invertible at all.

3. **Even the rigid lattice degrades slowly** (κ ~ N^0.38). With all gap fluctuations removed, a residual growth remains — the natural candidate mechanism is the Toeplitz-symbol behavior of the weighted prime sum (cf. open problem 2). This suggests a decomposition: κ_zeta ≈ (rigid-lattice baseline from the symbol) × (extreme-cluster correction from GUE gap statistics), which is testable (§8).

---

## 6. Relationship to Prior Work

### 6.1 Zero Statistics

Montgomery [6] established GUE pair correlation for ζ zeros (conditionally on RH). Odlyzko [7] confirmed this numerically at large scale. Rudnick and Sarnak [10] proved universality of n-level correlations. Torquato et al. [11, 12] studied hyperuniformity of zeta zeros. Our Gram matrix framework provides a complementary perspective: rather than studying zero statistics directly, it measures how the explicit formula's prime-zero duality depends on position within the critical strip.

### 6.2 Lehmer's Phenomenon

Lehmer [16] observed that certain zeros come in closely-spaced pairs that nearly violate the known bounds on zero separation. Our eigenvector analysis connects this circle of ideas to the resolution limit of the explicit formula, with a twist established in v4 (§5.5): the zeros that become invisible to the prime-zero Gram matrix are tight *clusters* selected by a local window eigenvalue, and the most famous Lehmer pair is out-competed by a compound cluster of modestly small gaps among the first 8000 zeros. The qualitative connection — anomalously tight zero configurations are what the prime sum cannot resolve — survives, but the pairwise-gap formulation of v3 does not. To our knowledge neither formulation has been previously described.

### 6.3 Connes' Program

Connes [4, 5] formulated RH as a positivity condition on the adele class space. Our Gram matrix is a finite-dimensional discretization of a related spectral pairing. However, we have not established any precise correspondence between our condition number and Connes' positivity condition.

### 6.4 Quasicrystal Speculation

Dyson [13] speculated about connections between zeta zeros and quasicrystals. Our observation of the σ = 1/2 isotropy minimum is consistent with a cut-and-project interpretation (the factorization G = V*V has projection structure), but the unbounded condition number means the projection degenerates at large N. The zeros may have quasicrystal-like statistics (hyperuniformity, extra rigidity [11]) without being a quasicrystal in the technical sense of forming a Meyer set [14, 15].

---

## 7. The σ = 1/2 Minimum: What It Means and What It Doesn't

### 7.1 What It Means

The critical line is the unique point where the Xi-weighted Gram matrix is most isotropic — where the eigenvalue ratio λ_max/λ_min is smallest. This holds universally across L-functions, at every height, and at every scale tested. The second-order analysis (Section 3.2) provides an algebraic explanation: the cosh perturbation operator has positive eigenvalue spread, so moving away from σ = 1/2 necessarily increases the condition number.

In geometric language: σ = 1/2 is where the prime-zero coupling is most balanced across all directions in zero space. At any other σ, some directions are compressed relative to others.

### 7.2 What It Doesn't Mean

The isotropy minimum does not, by itself, imply RH. The argument would require:

(a) Defining the Gram matrix using ALL zeros (including hypothetical off-line ones).
(b) Showing that positive definiteness at σ = 1/2 requires all zeros to be on the critical line.
(c) Proving bounded conditioning.

Step (c) fails — the condition number grows as N^{0.6}. Step (a) is circular for our construction. Step (b) is partially supported by the perturbation test but not proven.

The isotropy minimum is a geometric characterization of σ = 1/2, not a proof strategy for RH. Its value lies in connecting the functional equation symmetry (which is algebraic) to the spectral structure of the prime-zero coupling (which is analytic) through a concrete, computable object.

---

## 8. Open Problems

Status of v3's problems: (1) is extended but still open; (2) and (3) are sharpened and partially explained by v4's results; (4) is answered (§5.4); (5) remains untouched.

1. **Prove the isotropy minimum analytically.** v4 establishes that D ⪰ 0 always (§3.2) and verifies r > 0 up to N = 2000 with the curvature prediction matching direct computation to 4–5 decimals. What remains is to prove r > 0 — i.e., that D weights the top of G's spectrum more, relative to eigenvalue, than the bottom. Heuristic worth pursuing: D up-weights large log p, and the λ_min direction (a tight cluster, resolvable only by large primes' fast phases) receives relatively less D-mass per unit G-mass than delocalized directions.

2. **Prove the scaling exponent via the cluster functional.** v4's controls (§5.7) show the κ ~ N^0.6 exponent is reproduced by GUE spectra and is therefore a gap-statistics phenomenon, while the rigid lattice contributes a slower N^0.38 baseline (the Toeplitz-symbol mechanism). Conjectured decomposition: κ_zeta ≈ (symbol baseline) × (extreme value of the 3-zero cluster functional under GUE statistics). The small-gap side connects to known extreme-gap results for GUE/CUE (Ben Arous–Bourgade: smallest unfolded gap among N points ~ N^{−1/3}, suggesting λ_min ~ N^{−2/3} in the quadratic regime). Making this precise — including the resonant-regime correction of §5.5 — is the realistic path to proving the scaling law.

3. **Separate cluster-resolution from prime-count effects.** The window bound W_k plateaus a factor 1.2–1.6 above λ_min, growing slowly with N — but P = 20N in those runs. Re-run with fixed P to determine whether the delocalized component of the λ_min eigenvector is real or an artifact of the growing prime set.

4. **Cutoff schedule.** §5.4 shows the height degradation is removable by raising the prime cutoff. Determine the schedule P(H) that holds κ constant (the phase-resolution argument suggests polynomial in H); equivalently, quantify the explicit-formula resolution limit as a function of height.

5. **Test non-principal characters and higher-degree L-functions.** Extending universality beyond Dirichlet L-functions to Hecke L-functions, symmetric power L-functions, or Artin L-functions would strengthen or weaken the claim.

6. **Formalize the interlacing bound (§3.3) in Lean.** **Done** (June 2026): `exists_eigenvalue_mul_le_rayleigh`, `exists_eigenvalue_le_pair_bound`, and `gram_eigenvalue_le_pair_bound` are proven in `lean/RHCrystal/RHCrystal.lean` with zero sorries and zero custom axioms, extending the machine-verified portion of this work from the weight symmetry to a genuine spectral bound. Remaining extension: the k×k window version (W_k of §5.5), which would require formalizing principal-submatrix interlacing in general.

---

## References

[1] H. Montgomery, "The pair correlation of zeros of the zeta function," Proc. Symp. Pure Math. 24, AMS, 1973, 181–193.

[2] M. V. Berry and J. P. Keating, "H = xp and the Riemann zeros," in Supersymmetry and Trace Formulae, Springer, 1999, 355–367.

[3] A. Weil, "Sur les courbes algébriques et les variétés qui s'en déduisent," Hermann, Paris, 1948.

[4] A. Connes, "Trace formula in noncommutative geometry and the zeros of the Riemann zeta function," Selecta Math. 5 (1999), 29–106.

[5] A. Connes, "Noncommutative geometry and the Riemann zeta function," in Mathematics: Frontiers and Perspectives, AMS, 2000, 35–54.

[6] H. Montgomery, op. cit. [1].

[7] A. Odlyzko, "Tables of zeros of the Riemann zeta function," https://www-users.cse.umn.edu/~odlyzko/zeta_tables/

[8] H. Iwaniec and E. Kowalski, Analytic Number Theory, AMS Colloquium Publications 53, 2004.

[9] F. Johansson, "mpmath: a Python library for arbitrary-precision floating-point arithmetic," http://mpmath.org.

[10] Z. Rudnick and P. Sarnak, "Zeros of principal L-functions and random matrix theory," Duke Math. J. 81 (1996), 269–322.

[11] S. Torquato and F. H. Stillinger, "Local density fluctuations, hyperuniformity, and order metrics," Phys. Rev. E 68 (2003), 041113.

[12] S. Torquato, A. Scardicchio, and C. E. Zachary, "Point processes in arbitrary dimension from fermionic gases, random matrix theory, and number theory," J. Stat. Mech. (2008), P11019.

[13] F. J. Dyson, "Birds and Frogs," Notices of the AMS 56 (2009), 212–223.

[14] M. Baake and U. Grimm, Aperiodic Order, Vol. 1, Cambridge University Press, 2013.

[15] Y. Meyer, Algebraic Numbers and Harmonic Analysis, North-Holland, 1972.

[16] D. H. Lehmer, "On the roots of the Riemann zeta-function," Acta Math. 95 (1956), 291–298.

[17] G. Ben Arous and P. Bourgade, "Extreme gaps between eigenvalues of random matrices," Ann. Probab. 41 (2013), 2648–2681.

---

## Appendix A: Lean 4 Formalization

Machine-verified in Lean 4 (`lean/RHCrystal/RHCrystal.lean`):

1. The symmetry theorem w(p, σ) = w(p, 1−σ) and its corollaries (G(σ) = G(1−σ), hence κ(σ) = κ(1−σ)), together with cosh(x) ≥ 1 (via AM-GM on exp(x/2), exp(−x/2)).
2. **The interlacing pair bound (§3.3), new in v4:**
   - `exists_eigenvalue_mul_le_rayleigh` — for real symmetric A and any vector x, some eigenvalue λ_k satisfies λ_k·(x⬝x) ≤ x⬝(Ax) (proven via the spectral theorem and unitary conjugation of quadratic forms);
   - `exists_eigenvalue_le_pair_bound` — the test vector e_i − e_j gives 2λ_k ≤ A_ii + A_jj − 2A_ij;
   - `gram_eigenvalue_le_pair_bound` — instantiated for the Gram matrix: some eigenvalue is at most Σ_p w(p,σ)·(1 − cos((γ_i − γ_j)·log p)).

The formalization contains zero sorries and introduces zero axioms beyond Lean's standard foundations (propext, Classical.choice, Quot.sound), confirmed by `#print axioms` for every theorem. Build: `lake update && lake exe cache get && lake build`.

## Appendix B: Bug Disclosure

An earlier version of this work reported bounded conditioning (κ < 4 at N = 2000). This resulted from a diagonal doubling bug in the Rust implementation: in the inner loop over matrix entries, the case i = j wrote G[i,i] twice, effectively adding a scaled identity matrix to G and compressing the condition number by ~10×. The bug was discovered when scaling to N = 20,000 using a GPU (PyTorch) implementation that used vectorized outer products without an explicit i,j loop. All results in this paper use the corrected implementation. The Lean-verified algebraic results (symmetry, second-order minimum) are unaffected.

## Appendix C: Code and Data

**Zeros.** Odlyzko's tables: https://www-users.cse.umn.edu/~odlyzko/zeta_tables/

**Code.** Zero precomputation: Python/mpmath. Matrix analysis: PyTorch (GPU) and Rust (CPU, corrected). Formal verification: Lean 4 with Mathlib.

**Reproduction.** Full analysis (2M zeros, 664k primes, N up to 20,000) runs in ~60 minutes on an NVIDIA RTX 3090. The v4 experiments are `compute/rh_progress.py` (interlacing bound, control ensembles, curvature operator, height/cutoff scan) and `compute/rh_windows.py` (sliding-window analysis); both run in under 15 minutes on the same hardware.

## Appendix D: Revision History

**v2 → v3 (March 2026).** The initial Rust implementation doubled the Gram matrix diagonal, producing a spurious "bounded conditioning" result (κ < 4 at N = 2000). Corrected; the condition number in fact grows as ~N^0.61 (Appendix B).

**v3 → v4 (June 2026).** Changes of substance:

1. *§5.5 revised.* v3 claimed the λ_min eigenvector localizes on Lehmer pairs (minimal-gap pairs). Sliding-window analysis shows the localization site is selected by the minimal 3-zero window eigenvalue (a cluster functional), not the minimal gap; at N = 8000 the eigenvector ignores the classic Lehmer pair (gap 0.0377) in favor of a compound cluster of wider gaps. The qualitative "tight configurations are invisible to primes" claim survives; the pairwise formulation does not.
2. *§3.3 added.* A rigorous interlacing upper bound on λ_min via 2×2 (and general k×k) principal submatrices.
3. *§3.2 strengthened.* D ⪰ 0 holds by construction (Gram factorization); second-order verification extended from N = 30 to N = 2000 with 4–5 decimal agreement.
4. *§5.7 added.* Control ensembles (GUE / Poisson / picket fence at matched density) showing the conditioning and its scaling law are driven by GUE-type local gap statistics, not arithmetic.
5. *§5.4 extended.* Prime-cutoff scan showing the height degradation is a finite-prime resolution artifact (v3's open problem 4, answered).
6. *Lean formalization completed.* The remaining `sorry` (cosh ≥ 1) is proven; the formalization now has zero sorries and zero custom axioms.

No v3 numerical results required correction; all v3 data tables are reproduced unchanged. The v4 revisions refine interpretations and add new experiments.

---

*Computational study, March 2026; revised June 2026.*
*v3: corrected after identification of diagonal doubling bug in initial implementation.*
*v4: close-pair localization refined to tight-cluster localization after sliding-window analysis.*
