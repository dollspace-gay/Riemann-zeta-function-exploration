# Plan 6: The Lower-Bound Proof

*July 2026. Successor to plans/01 Steps 2–3 and plans/05, written after
the Session 7–8 results. Difference from every earlier lower-bound
discussion: Tier 1's key lemma chain is now COMPLETE mathematics (each
step an identity or a one-line inequality, written out below), not a
program. What remains there is error bookkeeping and writing. Tiers 2–3
remain research.*

## Scope: three different statements, in increasing strength

- **Tier 1 (chain-subspace floor).** ‖Σ α_k f_k‖² ≥ c·(log K)/K²·‖α‖²
  for windows of doubling-chain differences. Does NOT bound λ_min(G_N)
  (restriction bounds go the other way); it proves Theorem 1's witness
  family is order-optimal among chains, pinning the chain floor at
  Θ(K⁻² log K) and isolating the measured factor ≈1.2 as the entire
  remaining gap to λ_min.
- **Tier 2 (full matrix, unconditional).** λ_min(A_N) ≫ N^{−3−ε} via
  the Mellin route — the first polynomial floor of any exponent for the
  classical NB Gram matrix (none found in the literature check).
- **Tier 3 (the truth).** λ_min(A_N) ≫ N^{−2−ε}, matching Theorem 1.
  Research-grade; two routes with kill-tests.

---

## Tier 1: the proof (write this down first)

Setting: window W = (K−w, K], chain differences f_k = e_{2k} − e_k/2,
M_{jk} = ⟨f_j, f_k⟩ = (log2/8)(1/j + 1/k) − P_{jk}/16 (exact,
Session 3/7), P_{jk} = (2d/(jk))·(H_{⌊j/d⌋} + c* + ε(j,d)) with
c* = 1 + log(2/π) and ε the measured boundary layer (Session 7:
|ε| ≤ 0.003 for j/d ≥ 300, ε ~ A(θ)·d/j).

**Step 1 (identity — exact rank-two vanishing).** For Σα = 0:
Σ_{j,k} α_j α_k (g(j) + g(k)) = 2(Σαg)(Σα) = 0 for ANY g. Hence the
entire (log2/8)(1/j+1/k) part vanishes identically and

    Q(α) := αᵀMα = −(1/16)·αᵀPα        (zero-sum α, exact).

**Step 2 (kernel normal form).** Substituting the Session-7 law,
with L := log K + γ + c* and d = |j−k|:

    −αᵀPα = (2/K²)·[ −Σ α_jα_k d(L − log d) ] + αᵀEα,

E collecting three error kernels: (i) H_{⌊j/d⌋} − (log(j/d) + γ),
(ii) the boundary layer ε(j,d)·(2d/jk), (iii) jk vs K² normalization.
Each is pointwise O(d²/K³·polylog) on the window — bounded in Step 5.

**Step 3 (the two positivity identities).** On zero-sum α with partial
sums S_m (S_w = 0):

  (a) αᵀ(−d)α = 2Σ_m S_m², and ‖α‖² = Σ(S_m − S_{m−1})² ≤ 4ΣS_m²,
      so αᵀ(−d)α ≥ ‖α‖²/2.                          [partial sums]
  (b) For every s ≥ 0: min(d, s) = s − (s−d)₊ and
      (s−|a−b|)₊ = ∫ 𝟙[a, a+s](u)·𝟙[b, b+s](u) du — a box
      autocorrelation, hence PSD as a kernel. Constants die on
      zero-sum, so αᵀ(−min(d, s))α = αᵀ((s−d)₊)α ≥ 0.   [boxes]

**Step 4 (assembling the log).** Using d·log(w/d) = ∫₁^w min(d,s)/s ds
− d + 1 (check: ∫₁^d ds + ∫_d^w (d/s)ds = d − 1 + d log(w/d)):

    −Σαα d(L − log d)
      = (L − log w)·αᵀ(−d)α + αᵀ(−d·log(w/d))α
      = (L − log w)·2ΣS² + ∫₁^w αᵀ((s−d)₊)α ds/s − 2ΣS²
      ≥ (L − log w − 1)·‖α‖²/2.

Every term explicit; the box integral is discarded (it is ≥ 0).

**Step 5 (error bookkeeping — the labor).** |αᵀEα| ≤ Σ|α_j||α_k||E_{jk}|
with |E_{jk}| ≤ C·d²/K³·log K on the window gives
|αᵀEα| ≤ C·w²(log K)/K³·‖α‖² — subdominant to Step 4's
(log K)/K²-scale term as soon as w²·polylog ≪ K. **For the theorem,
fix w = w₀ constant** (Session 3: 20 chains attain the measured floor
within 1%, w = 160 adds < 1%), which trivializes this step; the
w ≤ K^{1/2−δ} version is an optional strengthening.

**Step 6 (non-zero-sum component).** α = β + t·𝟙/√n, β zero-sum. The
𝟙-direction has Rayleigh ≈ (log 2)/(4K) ≫ (log K)/K². Cross term: only
the VARIATION of the row sums couples to β (Σβ = 0), and row sums of M
vary by O(w²·polylog/K²) across the window — so |cross| ≤
O(w²·polylog/K²)·‖β‖·|t|, absorbed by AM-GM into the two diagonal
terms. Conclusion survives: Q(α) ≥ c·(log K)/K²·‖α‖².

**Theorem 2 (target statement).** Fix w₀ ≥ 4. There are explicit
K₀, C₀ such that for all K ≥ K₀ and all α supported on (K−w₀, K]:

    ‖Σ α_k f_k‖²  ≥  (log K − C₀) / (16 K²) · ‖α‖².

With Theorem 1 (≤ (H_{K−1}+1)/(10K(K−1)) for the witness): the
doubling-chain floor is Θ(K⁻² log K), constants explicit on both sides.

**Numerical pre-verification (gate before writing).** With exact M and
P from `kernel_cd.py`: (T1a) Step 1's identity to machine precision;
(T1b) every inequality of Steps 3–4 evaluated on random zero-sum α AND
on the true optimizer, correctly ordered, final bound below the
measured floor 0.59/K² but above zero — target ≈ 0.24/K² at K = 1250,
w = 32; (T1c) |E_{jk}| against the claimed envelope. Results recorded
in `~/rh_output/tier1_verify.txt` before the paper proof is drafted.

**Lean afterwards.** Steps 1, 3a, 3b are finite algebra (the box
autocorrelation is a finite sum identity in disguise for integer
windows) — natural continuation of the formalization track once the
paper proof exists.

Effort: 1 session numerics gate + 1–2 sessions writing. Risk: low —
the only soft spot is Step 5's constants, and fixing w₀ removes it.

---

## Tier 2: λ_min(A_N) ≫ N^{−3−ε}, unconditional (the Mellin ladder)

From `mellin_lower_bound.md` (M0–M1 done: identity anchored to Ehm's
published kernel evaluation; uncertainty lemma proved, ε-free, and
only 29–64× off the true adversary at N ≤ 2000). Remaining ladder:

1. **Source, don't recall**: the smoothed approximate functional
   equation with explicit error terms, and the Montgomery–Vaughan
   mean-value theorem, from primary references; record exact
   statements in the notes before use.
2. Window [T, 2T], T ≍ N^{2+ε}: assemble
   ∫|ζX|² ≳ T·Σ_r s_r²/r ≳ T·N^{−1−ε} through MV diagonal dominance;
   carry the AFE error terms and the χ-factor's second sum honestly.
3. Numerical instantiation at N = 500–1000 against the cached ζ-grid:
   every displayed inequality must hold with measured slack at finite
   N before it enters the writeup.
4. Writeup: **first polynomial lower bound for the NB spectral floor**,
   bracketing λ_min ∈ [N^{−3−ε}, N^{−2}log N] with Theorem 1.

Effort: 2–4 sessions, dominated by error-term sourcing and checking.
Risk: medium — no new ideas needed, but the AFE bookkeeping can eat a
session, and any recalled-constant shortcut is how this project gets
burned.

---

## Tier 3: closing to N^{−2−ε} (research; run kill-tests first)

The structural loss in Tier 2 is the T ≍ N² window (diagonal dominance)
against the t⁻² weight, while the true minimizers' spectral mass sits
at t ≍ N (Session 6 measurement). Two routes, each with a cheap
kill-test that the N = 10⁴ GPU data enables:

- **Route A (mollified Cauchy–Schwarz).** ∫|ζX|²w ≥ (∫ζX·Ḡ·w)²/∫|G|²w
  with G adapted to x's s_r skeleton; first moments of ζ reach windows
  T ≍ N^{1+ε} where second moments cannot. KILL-TEST: on exact data
  (N ≤ 1000, ζ-grid to T = 2000), numerically optimize G over a
  tractable family and measure the achievable exponent; if even the
  numerically-optimal G in the family cannot beat N^{−3}, the route
  dies before any hard analysis is attempted.
- **Route B (localization).** Prove near-null vectors of A_N have
  chain s_r-structure, then invoke Theorem 2's floor on that structure.
  Supported at N = 2000 (top-12 r's carry 70% of Q, all at r ≈ N/2, N).
  KILL-TEST: at N = 10⁴, measure the chain-subspace overlap of the
  λ_min eigenvector and the s_r concentration; if either decays with
  N, the route dies.

If both die: write the obstruction precisely — "a lower bound of type L
for length-N Dirichlet polynomials at height ≍ N" — as the sharpest
form of the open question. That statement plus Tiers 1–2 is still a
complete, publishable arc.

---

## Order of work

1. Tier 1 numerics gate (T1a–c), then the Theorem 2 writeup in
   `chains_theory.md` (upgrade §5 from program to proof).
2. Tier 2 ladder, one rung per session, numerics-anchored.
3. Tier 3 kill-tests as soon as the N = 10⁴ Gram matrix exists
   (GPU build; also feeds the Plan 2 amplitude measurement).
4. Lean formalization of Theorem 2's Steps 1/3a/3b after the paper
   proof stabilizes.
