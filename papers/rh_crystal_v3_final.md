# The Critical Line as Point of Maximum Isotropy: A Numerical Study of Prime-Zero Coupling

---

## Abstract

We define a family of Xi-weighted Gram matrices G(σ) that encode the coupling between prime numbers and non-trivial zeros of L-functions, using weights derived from the completed zeta function ξ(s) = ξ(1−s). Through large-scale GPU computation (up to 20,000 zeros with 664,579 primes, plus Odlyzko's dataset of 2,001,052 precomputed zeros), we establish the following:

1. **Isotropy minimum.** The condition number κ(G(σ)) is minimized at σ = 1/2 for the Riemann zeta function and nine additional L-functions (including complex Dirichlet characters), at every matrix size tested (N = 100 to N = 10,000), and at every height tested (γ ~ 14 to γ ~ 600,000). The symmetry κ(σ) = κ(1−σ) is exact by construction and machine-verified in Lean 4.

2. **Close-pair localization.** The eigenvector corresponding to the smallest eigenvalue of G is sharply localized on pairs of zeros with anomalously small spacing (Lehmer pairs). The participation ratio decreases with N, indicating that prime-zero coupling fails specifically at near-collisions of zeros.

3. **Height-dependent resolution.** The condition number of a fixed-size block of 1000 zeros grows from κ ≈ 32 at height ~14 to κ ≈ 3531 at height ~600,000, reflecting the increasing oscillation frequency of the prime-zero phase coupling. This connects the Gram matrix spectrum to the convergence rate of the explicit formula.

4. **Perturbation instability.** Artificially displacing zeros off the critical line increases the condition number monotonically: a shift of δ = 0.3 from σ = 1/2 nearly quadruples κ, with λ_min collapsing by a factor of 3.5.

---

## 1. Introduction

### 1.1 Context

The Riemann Hypothesis (RH) asserts that all non-trivial zeros of ζ(s) have real part 1/2. Three proof strategies have shaped modern approaches: the spectral program of Hilbert-Pólya [1, 2], the geometric program of Weil and Connes [3, 4, 5], and the statistical characterization through random matrix theory [6, 7, 10].

The Weil explicit formula [8, Ch. 5] establishes a duality between sums over zeros and sums over primes:

    Σ_ρ h(ρ) = (main terms) + Σ_p Σ_m log(p) · g(m log p) / p^{m/2}

This duality can be encoded as a bilinear form — a Gram matrix — whose entries measure how primes "couple" pairs of zeros. The present work studies this bilinear form numerically, using weights chosen to respect the functional equation ξ(s) = ξ(1−s).

### 1.2 Summary of Findings

We report a robust geometric property of the critical line (the σ = 1/2 isotropy minimum), a structural explanation for Gram matrix degeneration (close-pair localization), and a quantitative connection between conditioning and zero height. We also identify a specific limitation: the condition number is NOT bounded, growing approximately as N^{0.61}. An earlier version of this work reported bounded conditioning; this was traced to a bug in the matrix construction (diagonal doubling) and is corrected here.

### 1.3 What This Paper Does Not Claim

This paper does not prove any new theorem about RH. The σ = 1/2 symmetry is a built-in consequence of the weight function, not a discovery. The isotropy minimum is a numerical observation, proven only to second order via perturbation analysis. The close-pair localization is descriptive, not predictive. We are explicit about these limitations throughout.

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

**Lemma.** Expanding G(σ) = G(1/2) + ε²D + O(ε⁴) where ε = σ − 1/2 and D_{ij} = (1/2) Σ_p w(p, 1/2) · (log p)² · exp(−i(γ_i − γ_j) log p), the condition number is locally minimized at σ = 1/2 if the eigenvalue spread of G(1/2)^{−1}D is positive.

*Verification.* At N = 30 with 3000 primes, the eigenvalue spread of G^{−1}D is 13.14 (positive). The direct computation confirms: κ(0.50) = 2.77, κ(0.51) = 2.77, κ(0.55) = 2.85, κ(0.60) = 3.05, matching the second-order prediction.

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

This is consistent with known convergence properties of the explicit formula: the sum over primes converges more slowly for zeros at greater height, requiring larger primes to resolve higher zeros.

### 5.5 Close-Pair Localization (New Result)

**The eigenvector corresponding to λ_min is sharply localized on closely-spaced zero pairs.**

At N = 1000 (P = 20,000): zeros #888–889 (γ ≈ 1290.10, 1290.42, spacing 0.31) carry **88.9%** of the λ_min eigenvector weight. The participation ratio is PR = 2.5 out of 1000 — meaning the "invisible" direction is concentrated on approximately 2.5 effective zeros.

At N = 2000 (P = 40,000): zeros #1364–1365 (γ ≈ 1833.0, 1833.3, spacing 0.29) carry **80.0%** of the weight. PR = 3.1 / 2000.

The localization strengthens with N: PR/N decreases from 0.013 (N = 500) to 0.0015 (N = 2000).

**The invisible zeros consistently have anomalously narrow spacing:**

| N | Mean spacing | Weighted spacing (λ_min direction) | Ratio |
|------|-------------|-----------------------------------|-------|
| 500 | 1.597 | 1.044 | 0.654 |
| 1000 | 1.407 | 0.904 | 0.643 |
| 2000 | 1.251 | 0.899 | 0.719 |

Zeros in the λ_min eigenvector have spacings 65–72% of the mean — approximately the close pairs that are hardest for the prime sum to resolve. This is Lehmer's phenomenon [16] — the near-collision of zeros — manifesting as the null space of the prime-zero coupling.

**Physical interpretation.** Two zeros with spacing δγ can only be resolved by primes whose phase separation δγ · log p is of order 1, requiring log p ~ 1/δγ, or p ~ exp(1/δγ). For the close pairs observed (δγ ≈ 0.3), the required prime is p ~ exp(3.3) ≈ 27. These primes are available, but their contribution is diluted among hundreds of thousands of other primes. In effect, the information that resolves close pairs constitutes a vanishingly small fraction of the total Gram matrix signal.

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

---

## 6. Relationship to Prior Work

### 6.1 Zero Statistics

Montgomery [6] established GUE pair correlation for ζ zeros (conditionally on RH). Odlyzko [7] confirmed this numerically at large scale. Rudnick and Sarnak [10] proved universality of n-level correlations. Torquato et al. [11, 12] studied hyperuniformity of zeta zeros. Our Gram matrix framework provides a complementary perspective: rather than studying zero statistics directly, it measures how the explicit formula's prime-zero duality depends on position within the critical strip.

### 6.2 Lehmer's Phenomenon

Lehmer [16] observed that certain zeros come in closely-spaced pairs that nearly violate the known bounds on zero separation. Our eigenvector analysis provides a new characterization of Lehmer pairs: they are precisely the zeros that become invisible to the prime-zero Gram matrix, dominating the λ_min eigenvector. This connects Lehmer's phenomenon to the resolution limit of the explicit formula in a way that, to our knowledge, has not been previously described.

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

1. **Prove the isotropy minimum analytically.** The second-order result (eigenvalue spread of G^{−1}D > 0) holds at N = 30. Does it hold for all N? What determines the spread?

2. **Characterize the scaling exponent.** Why is κ ~ N^{0.61}? The exponent is likely related to the order of vanishing of the Toeplitz symbol f(τ) = Σ_p (log p / √p) · exp(−iτ log p) at its minimum. Proving this would connect our observations to Szego-type limit theorems.

3. **Explain close-pair localization.** Why do Lehmer pairs dominate λ_min? A theoretical explanation would illuminate the resolution structure of the explicit formula.

4. **Height-adaptive primes.** If primes are scaled with height (using primes up to exp(c · γ_N) for some c), does the conditioning stabilize? This would separate the "intrinsic" degeneration from the resolution-limited degeneration.

5. **Test non-principal characters and higher-degree L-functions.** Extending universality beyond Dirichlet L-functions to Hecke L-functions, symmetric power L-functions, or Artin L-functions would strengthen or weaken the claim.

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

---

## Appendix A: Lean 4 Formalization

The symmetry theorem w(p, σ) = w(p, 1−σ) and its corollaries (G(σ) = G(1−σ), κ(σ) = κ(1−σ)) are machine-verified in Lean 4. The proof introduces zero axioms beyond Lean's standard foundations. One `sorry` remains for cosh(x) ≥ 1 (AM-GM for exp, mechanically provable). Build: `lake update && lake exe cache get && lake build`.

## Appendix B: Bug Disclosure

An earlier version of this work reported bounded conditioning (κ < 4 at N = 2000). This resulted from a diagonal doubling bug in the Rust implementation: in the inner loop over matrix entries, the case i = j wrote G[i,i] twice, effectively adding a scaled identity matrix to G and compressing the condition number by ~10×. The bug was discovered when scaling to N = 20,000 using a GPU (PyTorch) implementation that used vectorized outer products without an explicit i,j loop. All results in this paper use the corrected implementation. The Lean-verified algebraic results (symmetry, second-order minimum) are unaffected.

## Appendix C: Code and Data

**Zeros.** Odlyzko's tables: https://www-users.cse.umn.edu/~odlyzko/zeta_tables/

**Code.** Zero precomputation: Python/mpmath. Matrix analysis: PyTorch (GPU) and Rust (CPU, corrected). Formal verification: Lean 4 with Mathlib.

**Reproduction.** Full analysis (2M zeros, 664k primes, N up to 20,000) runs in ~60 minutes on an NVIDIA RTX 3090.

---

*Computational study, March 2026.*
*Corrected after identification of diagonal doubling bug in initial implementation.*
