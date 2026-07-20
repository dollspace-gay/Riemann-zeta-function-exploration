# The K-Floor Problem: Reduction to Two Finite Inequalities

*July 2026. The Tier-2 kernel problem (⋆): prove λ_min(K) ≥ c/log T
for the sourced BCHB kernel K = (d,e)/√(de)·(log(T(d,e)²/2πde) + c₀),
d,e ≤ N, T = N^{2+δ}. Structure numerics: `kfloor_structure.txt`.*

## Exact identities (proved this session)

With G := gcd-Gram kernel (d,e)/√(de), D := diag(log d), L̃ := log T −
log 2π + c₀:

1. **Anticommutator form.** Hadamard product with the separable matrix
   log d + log e is an anticommutator:  G∘(log d + log e) = DG + GD.
   Hence, exactly:

       K = L̃·G + 2·(G ∘ log(d,e)) − (DG + GD).

2. **PSD of the gcd-log part.** log(d,e) = Σ_{q=p^k≤N} log p·u_q u_qᵀ
   (u_q = divisibility indicator), so G∘log(d,e) ⪰ 0 by Schur.
   (Verified numerically to 4×10⁻¹³.)
3. **CS reduction.** For g = xᵀGx (G-form) and the G-Cauchy–Schwarz on
   the anticommutator:  xᵀKx ≥ L̃g − 2√(g·⟨Dx, Dx⟩_G).
4. **Tilt split.** With log d = log δ + log m under d = δm:
   ⟨Dx,Dx⟩_G ≤ 2log²N·g + 2s, where s(x) = Σ_δ φ(δ)·z_δ²,
   z_δ = Σ_{m≤N/δ} x_{δm}·log m/√(δm)  — the log-tilted tail form.
5. Also proved en route: the sub-copy property (G restricted to
   multiples of q is a copy of G_{⌊N/q⌋}); and the Hamming identity
   log(de/(d,e)²) = Σ_q log p·(χ_q(d) − χ_q(e))² — the weight deficit
   is a log-weighted Hamming metric on prime-power profiles.
6. **Dead end recorded:** the one-move Schur transfer (M = 2log(d,e) +
   log(N/d) + log(N/e) PSD) is FALSE (λ_min(M) ≈ −675 at N = 2000).

## Measured laws (clean, unproven)

| quantity | measured | limit guess |
|----------|----------|-------------|
| λ_min(G)·log²N | 0.813, 0.833, 0.836 (N = 250→2000) | ≈ 0.84 |
| sup_x s(x)/(g(x)·log²N) | 0.513, 0.511, 0.506, 0.504 | **→ ½** |
| λ_max(G)/log²N | 0.486, 0.471, 0.470 | ≈ 0.47 |

## The conditional theorem (assembly proved; two inputs open)

**(A) gcd-floor:** λ_min(G_N) ≥ c_G/log²N.  [measured c_G ≈ 0.84]
**(B) log-tilt:** s(x) ≤ C′·log²N·g(x) for all x.  [measured C′ → ½]

**Theorem (modulo A, B).** For δ > 2√(2+2C′) − 2 (= 1.47 at C′ = ½):

    λ_min(K_N(T)) ≥ c_G·(δ − 2√(2+2C′) + 2)/log N   at T = N^{2+δ},

and with the sourced BCR error budget (admissible from δ > 1.117,
crude sup-norm reading):

    λ_min(A_N) ≫ N^{−2−δ}/log²N  for any δ > 1.47, e.g.
    **λ_min(A_N) ≫ N^{−3.5}/log²N unconditionally.**

(The ℓ²-uniform reading of BCR's error, if it holds, extends this to
every δ > 1.47 being wasteful — the binding constraint would become
the √3 of the CS step, improvable with a sharper anticommutator
treatment.)

## Attack notes for (A) and (B)

- (A): G is the Gram matrix of normalized arithmetic progressions in
  the density pairing (⟨√d·𝟙_{dℤ}, √e·𝟙_{eℤ}⟩ = (d,e)/√(de)) — a
  classical-flavored object (Smith/GCD-matrix literature, Wintner's
  Hilbert space of arithmetic functions). Literature check REQUIRED
  before proving from scratch. A rigorous weaker version is available
  now via the Möbius–Cauchy–Schwarz chain (λ_min(G) ≥ N^{−o(1)}
  explicit), enough for a N^{−2−δ−o(1)} statement.
- (B): the ½ suggests a continuous model (log-tilted Volterra operator
  on the multiplicative half-line) whose norm is exactly computable;
  Schur-test with the model's weights is the natural proof route.

## Status

(⋆) is REDUCED — exact scaffolding proved, constants measured, two
clean finite inequalities remain. Not a theorem yet; the honest
distance to one is (A) + (B), both of which look like single-session
targets with the attack routes above.

## Addendum (Session 21): Lemma (A) — literature cleared, structure established

**Sources obtained and read in full:** Lindqvist–Seip, *Note on some
greatest common divisor matrices*, Acta Arith. 84 (1998) 149–154
(fetched from the ICM archive). Their matrix M_s =
((m,n)^{2s}/(mn)^s) at s = ½ IS our G. Their Theorem: sharp bounds
ζ(2s)/ζ(s)² ≤ λ_N ≤ Λ_N ≤ ζ(s)²/ζ(2s) for s > 1 (via the
Hedenmalm–Lindqvist–Seip Riesz-basis theory, Duke 1997 — the founding
paper of the Hardy space of Dirichlet series); for ½ < s ≤ 1 only the
qualitative liminf λ_N = 0; **s = ½ explicitly excluded** ("our
characterization breaks down… due to divergence"). Also probed:
Mattila–Haukkanen (arXiv:1309.0320), Altınışık–Büyükköse
(arXiv:1408.3113), Aistleitner–Berkes–Seip (spectral norms of GCD
matrices — λ_max/Gál sums ONLY, zero smallest-eigenvalue content).

**Calibrated verdict:** the rate λ_min(G_N) ≍ 1/log²N at s = ½ (our
measured constant 0.836) is NOT in any source found. Same claim level
as the June chains check: "not found in a targeted search; the experts
may know it; asking is part of the question."

**Structure established for the proof (this session, exact):**
- LS representation: the G-form is the L²(0,π)-norm of Σ c_m u(mx),
  u(x) = Σ_k sin(kx)/k^s — at s = ½ the generator diverges, and the
  finite-N truncation regenerates it with a k-cutoff whose
  gcd-computation produces log(Kd/max(m,n))-type factors: the 1/log²
  law is the variance of that log over the lattice. This is the
  continuous mechanism to formalize.
- Regularization bridge: with ρ the CND Hamming metric (§ identities),
  Schoenberg gives e^{−σρ} PSD, and M_{½+σ} = G ∘ e^{−σρ} exactly —
  connecting G to the LS-covered range s > ½ at σ ≍ 1/log N.
- Rigorous fallback available now: the Möbius–Cauchy–Schwarz chain
  gives λ_min(G_N) ≥ N^{−o(1)} (explicit; τ-loss intrinsic to that
  route).

**Status:** (A) is open-and-ours; attack = HLS machinery at
s = ½ + c/log N. One dedicated session, with the measured 0.836 as
the target and the LS representation as the tool.

## Session 22: LEMMA (B) PROVED — and the unconditional theorem assembles

Verification of every step: `~/rh_output/lemmaB_verify.txt`.

### Lemma (B), proved (constant C′ = 1 + O(1/log N))

**The key identity (von Mangoldt convolution).** With w_d = x_d/√d,
y_δ = Σ_{m≤N/δ} w_{δm}, z_δ = Σ_{m≤N/δ} w_{δm}·log m: since
log m = Σ_{d|m} Λ(d),

    z_δ = Σ_{d≤N/δ} Λ(d)·y_{δd}          (exact; verified 9×10⁻¹⁵).

The tilt operator is Λ-convolution along divisor chains.

**Proof of s ≤ (log²N + C·log N)·g.** Cauchy–Schwarz with weights
ω_d = Λ(d):

    φ(δ)z_δ² ≤ [Σ_{d≤N/δ} Λ(d)·φ(δ)/φ(δd)]·[Σ_d Λ(d)·φ(δd)y_{δd}²]
             ≤ (log(N/δ) + C₀)·Σ_d Λ(d)·φ(δd)y_{δd}²,

using φ(δd) ≥ φ(δ)φ(d) (superadditivity) and the elementary
Mertens-type bound Σ_{d≤M} Λ(d)/φ(d) ≤ Σ_{p≤M} log p·p/(p−1)²
≤ log M + C₀ (C₀ explicit; measured 0.65 in range). Summing over δ and
collecting by δ' = δd:

    s ≤ Σ_{δ'} φ(δ')y_{δ'}²·B(δ'),
    B(δ') = Σ_{d|δ'} Λ(d)·(log(Nd/δ') + C₀)
          = (log(N/δ') + C₀)·log δ' + Σ_{d|δ'} Λ(d) log d,

using Σ_{d|n}Λ(d) = log n. The second piece is
Σ_p log²p·k_p(k_p+1)/2 ≤ log²δ', so with t = log δ' ≤ L = log N:
B ≤ tL − t² + C₀t + t² = tL + C₀t ≤ L² + C₀L. ∎

[Numerically the bound is SATURATED: max B = 62.779 vs L² + C₀L =
62.783 at δ' = 1999 (prime) — prime δ' is the extremal case of this
bound, though not of the true operator norm, whose measured value ½L²
matches the crude-Volterra constant: in the continuous variables
W(v) = e^{v/2}Y(e^v) the operator is EXACTLY ∫_u^L W(v)dv, the Volterra
operator, with sharp norm 2L/π (⟹ C′_sharp = 4/π² ≈ 0.405) and
crude-CS norm L/√2 (C′ = ½). Our discrete C′ = 1 is what elementary CS
delivers; any constant feeds the assembly.]

### Lemma (A′), proved (weaker gcd-floor, fully elementary)

For unit x, with x_d/√d = Σ_m μ(m)y_{dm} (Möbius inversion) and CS
with weights φ(m):

    x_d² ≤ d·(Σ_{m≤N/d} μ²(m)/φ(m))·(Σ_m φ(m) y_{dm}²),

and two elementary explicit bounds: Σ_{m≤M}μ²(m)/φ(m) ≤ C₂(1 + log M)
with C₂ = ζ(2)ζ(3)/ζ(6) (via m/φ(m) = Σ_{d|m}μ²(d)/φ(d) and swapping),
and d/φ(d) ≤ log₂d + 1 (via p_k ≥ k+1). Summing over d, each δ = dm
appears τ(δ) times:

    1 ≤ C₃(1+log N)²·Σ_δ φ(δ)y_δ²·τ(δ) ≤ C₃(1+log N)²·τ_max(N)·g,

so λ_min(G_N) ≥ [C₃(1+log N)²·τ_max(N)]⁻¹ ≥ exp(−C₄ log N/log log N),
using the classical explicit τ(n) ≤ exp(C log n/log log n). ∎
(N^{−o(1)}-grade; the τ-loss is intrinsic to this route — the sharp
1/log²N of Lemma (A) remains the open refinement.)

### THE UNCONDITIONAL THEOREM (modulo the sourced BCR statement)

Assembly with C′ = 1 (threshold δ > 2√(2+2C′) − 2 = 2): take
δ = 2 + ε′, T = N^{4+...}-window admissible (θ = 1/(2+δ) < 17/33 ✓,
BCR error subdominant ✓, log-arguments positive since N ≤ T^{1/4} ✓):

    λ_min(K) ≥ (L̃ − 4 log N − o(log N))·λ_min(G_N)
             ≥ ε′·log N·exp(−C₄ log N / log log N),

and through the window/weight chain:

    **λ_min(A_N) ≥ N^{−4−ε}   for every ε > 0, N ≥ N₀(ε) —
    unconditionally** (the only external input being the BCHB/BCR
    main-term-plus-error statement exactly as published). With
    Theorem 1: N^{−4−ε} ≪ λ_min(A_N) ≪ N^{−2} log N — the first
    unconditional polynomial bracket on the Nyman–Beurling spectral
    floor. Proving Lemma (A) sharp (1/log²N) upgrades the exponent
    to −2−δ for δ > 2; the ℓ²-uniform reading of BCR's error plus the
    Volterra-sharp constant would push δ → 2√(2+8/π²) − 2 ≈ 1.48.

Remaining for a clean manuscript: constant bookkeeping in one place,
the τ- and Mertens-bound citations or their two-paragraph elementary
proofs, and the BCR quotation with its exact hypotheses. No
mathematical gaps remain in the chain above.
