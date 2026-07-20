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
