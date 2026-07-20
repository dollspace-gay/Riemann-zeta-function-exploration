# Lemma (A): brief for an external agent

*Self-contained problem statement from the Riemann-zeta-function-exploration
repository (July 2026). Calibration rules at the end — please keep the
proved/measured/conjectured labels intact in anything you produce.*

## The problem

Let G_N be the N×N matrix

    (G_N)_{mn} = gcd(m,n) / sqrt(mn),   1 <= m,n <= N.

It is symmetric positive definite (Gram matrix; see identity (*) below).

**Lemma (A) (OPEN, both directions):**

    lambda_min(G_N) ~ c / log^2 N,   measured c ≈ 0.83–0.84.

Measured values (exact eigensolves, float64, cross-checked):

    N:               250     500     1000      2000      4000
    lambda_min:       —       —    0.017460  0.014471  0.012047
    lambda*log^2 N: 0.813   0.826   0.8331    0.8360    0.8287

(The constant drifts non-monotonically at accessible N; treat "c ≈ 0.836"
as indicative, possibly with loglog corrections.) Also measured:
lambda_max(G_N)/log^2 N → ≈ 0.47.

The valuable half is the LOWER bound lambda_min >= c/log^2 N. Current best
rigorous lower bound (ours, elementary): exp(−C log N/log log N) — that is
N^{−o(1)} but WORSE than every fixed power of log. So **any** bound
c/(log N)^A for fixed A, however large, is new and useful. A proved upper
bound O(1/log^2 N) would also be new.

## Why it matters

G_N is the s = 1/2 member of the Lindqvist–Seip family
M_s = ((m,n)^{2s}/(mn)^s). In our program (lower bounds for the smallest
eigenvalue of the Nyman–Beurling Gram matrix A_N, an RH-equivalent object),
the Balasubramanian–Conrey–Heath-Brown/BCR twisted-second-moment kernel K
decomposes exactly as

    K = Ltilde∘G + 2(G∘log gcd) − (DG + GD)     (exact identities, verified)

and the needed K-floor lambda_min(K) >= c/log T reduces to two lemmas:
(B) a log-tilt bound — **proved** (von Mangoldt convolution + Volterra
model); (A) the gcd-floor above — **open**. With (A), our unconditional
bound lambda_min(A_N) >= N^{−4−ε} upgrades toward the conjectured-sharp
N^{−2}·polylog. (Caveat: that pipeline is "modulo the published BCR
statement" for their error term; (A) itself is a clean self-contained
problem independent of that caveat.)

## What is proved (elementary, in-repo, machine-checked numerically)

1. **The class-sum identity.** For any complex x (expand gcd = Σ_{d|·} φ(d)):

       x* G x = Σ_{d<=N} φ(d) · | Σ_{n<=N, d|n} x_n/√n |².        (*)

   Verified against the matrix to 2e-15. Consequences: G is PD; G = BᵀB
   with B_{d,n} = √φ(d)·[d|n]/√n; the form is O(N log N) to evaluate
   (we use this to reach N = 10^6 without a matrix).

2. **The symbol is zeta on the critical line.** Put x_n = W(n)·n^{−it}
   (W a smooth amplitude taper). Then the class-d sum in (*) is
   d^{−1/2−it} × (a W-smoothed partial sum of ζ(1/2+it) over k <= N/d).
   So all divisor classes resonate with ζ(1/2+it) simultaneously and all
   vanish together exactly at the Riemann zeros. This is proved
   (elementary); the finite-N quantitative consequences below are measured.

3. **Weak lower bound.** lambda_min >= exp(−C log N/log log N) via a
   per-divisor Möbius–Cauchy–Schwarz argument. The divisor-count loss
   τ(n) = N^{Θ(1/loglog N)} appears intrinsic to ANY per-divisor argument
   ("τ-obstruction") — a proof of (A) must treat classes collectively.

4. **Exactly solved model (the mechanism template).** For the ORTHONORMAL
   model system φ_n = e_n − e_{qn} (⟨e_m,e_n⟩ = δ_{mn}), the Gram matrix
   splits into q-adic chains = discrete Laplacians, and EXACTLY
   lambda_min = 4 sin²(π/(2(L+1))), L = 1 + floor(log_q N),
   ~ π² log²q / log²N. Its symbol is D(s) = 1 − q^{−s}; the floor is the
   finite-section resolution of the symbol's zeros (a lattice with spacing
   2π/log q). Verified to 1e-9. Our G is the non-orthogonal ζ-symbol
   analogue; the model deletes all off-chain gcd coupling.

## What is measured (no fits, no free parameters; scripts in repo)

5. **The ground state sings the zeros.** Periodogram (in log n, after
   √n-flattening and cubic detrend) of the bottom eigenvector peaks at
   f ≈ 14.04 for N = 1000/2000/4000 with SNR ≈ 21–22, i.e. at
   γ₁ = 14.1347 minus a displacement ≈ 0.10 = O(1/log N). It carries a
   COMB: line amplitudes over baseline at γ₁..γ₇ =
   16.6, 3.9, 2.9, 4.7, 4.0, 1.7, 3.2; at between-zero controls
   0.8, 0.4, 0.2, 2.3, 0.4, 0.5. The minimizer is an all-zeros object.
6. **Not Möbius-inheritance.** Raw μ(n) through the identical pipeline:
   SNR 3.3 only. (But the minimizer's SIGNS track μ(n) where μ ≠ 0, and
   its largest components sit on small smooth numbers 6, 30, 12, 2, 10…)
7. **The symbol curve.** S(t) = Rayleigh quotient of x_n = W(n)n^{−it}
   (Hann⁴ taper, N = 2000) has one local minimum per Riemann zero in
   band (7 of 7 up to t = 42), tracks A·|ζ(1/2+it)|² with
   corr(log,log) = 0.79 for t > 8, contrast up to ×9.4. Eigen-peak and
   S-dip agree to 3 decimals (both 14.035).
8. **Single-zero constructors FAIL (the measure obstruction).** Vectors
   x_n = Ω(log n/log N)·n^{−iγ₁} evaluated via (*) up to N = 10^6 give
   S·log²N growing (8.5 → 23.7), S plateauing ≈ 0.12: the right
   resonance (off-zero control is ×13–16 worse) but NOT the 1/log² law.
   Reason: in counting measure the log-uniform profile's ℓ²-mass sits in
   the last e-fold, where the taper vanishes. Consequences: (i) c ≈ 0.836
   is a COLLECTIVE quantity of all zeros — do not expect a
   |ζ′(ρ₁)|-numerology; (ii) this quantifies the same measure clash that
   makes s = 1/2 degenerate for Lindqvist–Seip.

## Literature status (searched independently twice)

- Lindqvist–Seip, Acta Arith. 84 (1998): sharp lambda bounds for M_s at
  s > 1 (ζ(2s)/ζ(s)² <= λ <= ζ(s)²/ζ(2s)); λ_min → 0 for 1/2 < s <= 1;
  s = 1/2 excluded (representation diverges — ζ(2s) pole). They remark an
  "arithmetic proof" of their sharp constants seems unlikely.
- Hedenmalm–Lindqvist–Seip, Duke 86 (1997): H² of Dirichlet series;
  dilated systems ↔ multiplier operators. The natural machinery, but no
  finite-section/truncation theorem with rates.
- GCD-sums literature (Gál, Aistleitner–Berkes–Seip, …): λ_max/spectral
  norm only. Hilberdink: multiplicative Toeplitz spectra, again no
  truncated λ_min law. Antezana–Carando–Scotti: dilation synthesis =
  Dirichlet multiplication (structural, no rates).
- Conclusion of both searches (ours and an independent agent's): the law
  λ_min(G_N) ≍ 1/log²N appears UNRECORDED. Also useful (Schoenberg
  bridge): M_{1/2+σ} = G ∘ exp(−σρ) with ρ(m,n) = log([m,n]/(m,n)),
  and ρ is (empirically) conditionally negative definite.

## Suggested attack surface (labeled by status)

A1. **HLS at the regularized exponent** (open route): run the
    Hedenmalm–Lindqvist–Seip multiplier analysis at s = 1/2 + c/log N on
    the N-truncated system, hoping the truncation regularizes the ζ(2s)
    divergence into the log²N law. No known obstruction, no known theorem.
A2. **Finite-section theorem for the ζ-symbol** (open route): prove the
    analogue of the model's exact Laplacian law for symbol ζ(1/2+it):
    "finite sections of length N resolve a boundary zero of the symbol at
    scale 1/log N, and the squared resolution is the eigenvalue floor."
    The model (item 4) is the worked template; the non-orthogonality of
    dilations is the whole difficulty.
A3. **Mollifier/Bessel dual** (PROPOSED BY US, UNTESTED — verify before
    trusting): by (*), (A)'s lower bound says: length-N vectors cannot
    make all φ-weighted ζ-classes simultaneously small — a quantitative
    non-resonance statement. Transfer to the integral world (Montgomery–
    Vaughan mean value theorem, T ≍ N^{1/θ}): it resembles
    ‖ζ(1/2+it)·A(1/2+it)‖_{L²[0,T]} >= (c/log T)·‖A‖ for Dirichlet
    polynomials A of length N. Strategy: extract each coefficient a_m by
    Cauchy–Schwarz against a shifted mollifier ψ_m ≈ (truncated 1/ζ)·m-shift:
    Σ_m |a_m|² <= ‖ζA‖² · BesselBound({ψ_m}); a Bessel/overlap bound
    O(log²N) for the mollifier family would GIVE (A). Upper-bound-type
    estimates are usually the tractable kind. CAUTION: the twisted second
    moment of ζ against |A|² is the BCR kernel K, and our program derives
    the K-floor FROM (A) — any attack through K's main-term matrix is
    circular. The non-circular version must use the analytic/integral side
    directly (contour/positivity/large-sieve tools), not the K-matrix.
A4. **Prove the upper bound first** (open, likely easier): construct
    vectors achieving O(1/log²N). Item 8 shows single-zero log-taper
    vectors fail; the true minimizer is μ-signed, small-smooth-supported,
    all-zeros-resonant. A μ-weighted MULTI-zero or fully mollified
    construction (à la c_k = μ(k)(1 − log k/log N), which is our
    Nyman–Beurling detection mollifier) evaluated through (*) is the
    natural candidate; (*) makes any candidate testable to N = 10^6 in
    seconds.

## Verification protocol (please follow)

- Any claimed bound or construction: evaluate through identity (*)
  numerically at N = 10^3..10^6 and report the measured constants next to
  the claimed ones. The repo instrument is `nyman-beurling/symbol_zero.py`
  (functions `phi_sieve`, `gcd_form`).
- Numbers to hit: the λ_min table above; the comb amplitudes; S-dip
  displacement γ₁ − 0.100 at N = 2000.
- Label every statement: proved / measured / proposed-untested. Recalled
  literature must be sourced before use (we have twice caught plausible
  misremembered constants in this project by insisting on this).
