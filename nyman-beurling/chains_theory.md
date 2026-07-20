# The Doubling-Chain Structure of the Nyman–Beurling Gram Matrix

*Working notes, June 2026. Companion code: `chains.py`. Status: §1–2 and
the **Theorem in §0** are proved (elementary, every step numerically
cross-validated); §4's kernel law is now derived up to one measured
constant; the matching lower bound in §5 remains a program.*

## 0. THEOREM (upper bound on λ_min — proved)

**Theorem 1.** Let K = ⌊N/2⌋ ≥ 3 and let H_m denote the m-th harmonic
number. Then

    λ_min(G_N) ≤ (H_{K−1} + 1) / (10·K(K−1)).

In particular λ_min(G_N) = O(N⁻² log N), hence κ(G_N) ≥ c·N²/log N.

*Proof.* Take the test vector x with coefficients x_{K−1} = −1/2,
x_{2K−2} = 1, x_K = +1/2, x_{2K} = −1 (i.e. Σx_k e_k = f_{K−1} − f_K,
a difference of two adjacent doubling chains; the four indices are
distinct and ≤ N). Then ‖x‖² = 5/2, and by the Rayleigh principle
λ_min ≤ ‖f_{K−1} − f_K‖²/(5/2).

By the square-wave identity (§1), f_k = (1 − ε_k)/4 with
ε_k(u) = (−1)^{⌊u/k⌋} (u = 1/t coordinates, measure u⁻²du). Hence

    ‖f_{K−1} − f_K‖² = (1/16)∫(ε_K − ε_{K−1})² u⁻²du
                     = (1/4)∫_D u⁻²du =: P/8,

where D = {u : ⌊u/(K−1)⌋ − ⌊u/K⌋ is odd} and P := 2∫_D u⁻²du.

**Lemma (disagreement intervals).** For 0 < u < U* := K(K−1),
⌊u/(K−1)⌋ − ⌊u/K⌋ = #{ℓ ≥ 1 : (K−1)ℓ ≤ u < Kℓ} ∈ {0, 1}, equal to 1
exactly on the disjoint union ⋃_{ℓ=1}^{K−1} [(K−1)ℓ, Kℓ). (Disjointness:
the next interval starts at (K−1)(ℓ+1) = (K−1)ℓ + K − 1 ≥ Kℓ iff
ℓ ≤ K−1.) ∎

Therefore

    P = 2 Σ_{ℓ=1}^{K−1} [1/((K−1)ℓ) − 1/(Kℓ)] + P_tail
      = 2H_{K−1}/(K(K−1)) + P_tail,    0 ≤ P_tail ≤ 2/U*,

since each bracket equals 1/(K(K−1)ℓ), and the tail is bounded by the
full measure ∫_{U*}^∞ 2u⁻²du. Assembling:
λ_min ≤ P/20 ≤ (2H_{K−1} + 2)/(20·K(K−1)). ∎

*Numerical validation of every step (N = 2500, K = 1250):* exact interval
enumeration gives P = 1.05635×10⁻⁵; the Gram data gives
8‖f_{K−1} − f_K‖² = 1.05763×10⁻⁵ (0.12%, within the enumeration's tail
allowance); the harmonic term is 9.8738×10⁻⁶ and the actual tail
6.90×10⁻⁷ sits inside its bound 1.281×10⁻⁶ (54% saturated). The theorem
bound evaluates to 5.58×10⁻⁷ against the true λ_min = 2.477×10⁻⁷ —
valid, and within a factor 2.3.

This is the first theorem of the Nyman–Beurling phase of this project:
the NB dilation basis degenerates at least at rate N⁻²log N, witnessed
explicitly by a four-term integer combination of dilations.

## 1. The square-wave identity (proved)

For all integers m, k ≥ 1 and t > 0, writing b = b(t) = ⌊1/(kt)⌋:

    e_{mk}(t) = e_k(t)/m + (b mod m)/m.                          (1)

*Proof.* With y = 1/(kt): ρ(y/m) = y/m − ⌊y/m⌋ and ρ(y)/m = y/m − ⌊y⌋/m,
so e_{mk} − e_k/m = (⌊y⌋ − m⌊y/m⌋)/m = (⌊y⌋ mod m)/m, and ⌊y/m⌋ = ⌊⌊y⌋/m⌋. ∎

So the **chain difference** f^m_k := e_{mk} − e_k/m is the square wave
(b mod m)/m. Since b = 0 exactly when t > 1/k, f^m_k is supported on
(0, 1/k]. Its norm is exact:

    ‖f^m_k‖² = C_m/k,   C_m = (1/m²) Σ_{r=1}^{m−1} r² Σ_{j≡r (m)} 1/(j(j+1)).

For m = 2 the inner sum telescopes to the alternating harmonic series:
**C₂ = (log 2)/4 = 0.17328680…** — matching the constant measured in
Session 1 to seven digits. Verified for m = 2, 3 against the Gram data at
1e−12 (code §1).

**Corollary (closed form for a Vasyunin-type integral).** From
C₂/1 = A(2,2)·… expansion at k = 1: ‖f²₁‖² = A(2,2) + A(1,1)/4 − A(1,2),
with A(2,2) = A(1,1)/2 = (log 2π − γ)/2, hence

    A(1,2) = ∫₀^∞ ρ(1/t)ρ(1/2t) dt = (3/4)(log 2π − γ) − (log 2)/4
           = 0.772209256…,

agreeing with the independently computed Gram entry to all printed digits.

## 2. The zero-sum reduction (proved)

For m = 2: b mod 2 = ½ − ½(−1)^b globally (including b = 0). Hence for any
coefficients α on a set S of base indices, if **Σα_k = 0**,

    Σ_{k∈S} α_k f²_k(t) = −¼ Σ_{k∈S} α_k (−1)^{⌊1/(kt)⌋},

so (in u = 1/t coordinates, measure u⁻²du)

    ‖Σ α_k f²_k‖² = (1/16) ∫₀^∞ [Σ_k α_k (−1)^{⌊u/k⌋}]² u⁻² du.     (2)

The u → 0 divergence is absent precisely because Σα = 0 (all signs +1
there). The NB λ_min problem, restricted to chains, is the minimization of
the explicit oscillatory functional (2).

## 3. The chain subspace attains λ_min up to a stable constant (numerical)

Let M_w(N) be the Gram matrix of {f²_k : k ∈ (N/2 − w, N/2]} (entries from
the full A by bilinearity; coefficient norm 5/4 per chain, disjoint
supports). Generalized minimal Rayleigh quotient λ_chain vs true λ_min(G_N):

| N | λ_min(G_N) | λ_chain (w=20) | ratio |
|------|-----------|----------------|-------|
| 500 | 5.460e−6 | 6.481e−6 | 1.19 |
| 1000 | 1.432e−6 | 1.735e−6 | 1.21 |
| 2000 | 3.816e−7 | 4.642e−7 | 1.22 |
| 2500 | 2.477e−7 | 3.036e−7 | 1.23 |

Twenty chains suffice (w = 160 improves λ_chain by < 1%); adding 3-chains
does not help (mixed 2+3 at w = 80 equals pure-2 to 3 digits). The
optimal α is an **alternating-sign smooth envelope with Σα = 0** (measured
Σα = 0.0000), i.e. exactly the ansatz that activates reduction (2).

Since λ_min ≤ (4/5)λ_min(M_w) is rigorous (restriction of the Rayleigh
quotient), any upper bound on the chain form transfers to G_N.

## 4. The kernel law (derived, June 9 update)

The exact decomposition (immediate from f = (1−ε)/4 and ‖f_k‖² = C₂/k):

    M_{jk} = (log 2/8)(1/j + 1/k) − P_{jk}/16,

with P_{jk} = 2∫_{D_{jk}} u⁻²du the parity-disagreement integral —
diagonal P_{kk} = 0 reproducing M_{kk} = C₂/k exactly. Exact enumeration
of D at K = 1250 gives, for d = k − j,

    P_{jk}·K²/(2d) = log(K/d) + γ + c(d),
    c(d) measured: 0.545 (d=1) → 0.661 (d=32), drifting toward log 2.

So the kernel's log-slope is **exactly 1/8** (= 2/16), not the 0.143 of
the Session-3 window fit — that fit mixed pairs at different scales while
normalizing all by the same K², inflating both constants by ~(K/max)²
heterogeneity. (A good cautionary example of fitted-constant
contamination; the derivation supersedes the fit.)

### 4.1 The kernel constant derived exactly (July 2026; supersedes the c(d) reading above)

Code: `kernel_cd.py`; raw output `~/rh_output/kernel_cd.txt`.

**Exact evaluation of P (new).** After gcd reduction (P(gj′, gk′) =
P(j′,k′)/g, mirroring the A-scaling), the parity pattern of
Δ(u) = ⌊u/j⌋ − ⌊u/k⌋ is *periodic*: Δ(u + jk) = Δ(u) + d, so the parity
period is L = jk for d even and 2L for d odd. Splitting the shifted
copies and summing termwise, with {[a_i, b_i)} the odd-parity segments
of one period,

    P = 2 Σ_i [ (1/a_i − 1/b_i) + (ψ(1 + b_i/Lp) − ψ(1 + a_i/Lp))/Lp ],

ψ = digamma — machine precision at O(K) cost, no truncation. Validated
against brute-force brackets, against Session 4's Gram-derived
P(1249,1250) = 1.05763×10⁻⁵ (rel. diff 9×10⁻⁷), and against the cached
chain-window eigenvalues of `chains.py` at four (K,w) anchors
(rel. diff ≤ 7×10⁻⁵ — this validates the decomposition
M_{jk} = (log2/8)(1/j + 1/k) − P_{jk}/16 end to end, with no reference
to the numerically-integrated Gram matrix).

**Exact special value.** P(1,2) = π/2: the disagreement set for (1,2)
is ⋃_{r≥0}[4r+1, 4r+3), and P telescopes to the Leibniz series.

**The constant, in closed form.** Δ has mean λ = u·d/(jk), and the local
odd-parity fraction converges to the triangle wave f(λ) = dist(λ, 2ℤ).
The disjoint-interval head is exactly a harmonic sum (§0's mechanism for
every d), and the averaged tail evaluates in closed form:
∫₁^∞ f(λ)λ⁻²dλ = 1 + log(2/π), the log arising from the Wallis product
Π(1 − 1/4r²) = 2/π. Hence, for the reduced pair (j, k), gap d:

    P_{jk} · jk/(2d) = H_{⌊j/d⌋} + c* + O(A(θ)·d/j),
    c* = 1 + log(2/π) = 0.548417…,   θ = {j/d}, A(0) = 0.

Confirmed numerically to 5–6 decimals across K = 156…5000, d = 1…64;
the deviation decays like K⁻² at d = 1 (where θ = 0) and like d/j with a
bounded θ-dependent coefficient otherwise. **The "c(d) drifting toward
log 2" reading of the June measurement is superseded:** the drift was
the K²-vs-jk normalization plus harmonic-vs-log discreteness; the
intrinsic constant is universal and equals c*. (The June table is
reproduced exactly by the exact P under the June normalization:
0.5546 → 0.6611 over d = 1…32 at K = 1250, matching the measured
0.545 → 0.661.)

### 4.2 The §5 lemma pre-check: the zero-sum floor is flat in w (July 2026)

With M built from the exact decomposition, the zero-sum-restricted floor
F(w, K) = K²·min_{α⊥1} αᵀMα/‖α‖² over windows (K−w, K]:

| w | K=250 | K=625 | K=1250 |
|-----|-------|-------|--------|
| 8 | 0.5103 | 0.5630 | 0.6054 |
| 32 | 0.5062 | 0.5542 | 0.5928 |
| 128 | 0.5001 | 0.5507 | 0.5903 |
| 200 | 0.4756 | 0.5479 | 0.5885 |

**Flat in w** — the w-independent lower-bound premise of §5's
conditional-positivity lemma survives its cheapest falsification test.
The unconstrained floor coincides with the zero-sum floor to 4 decimals
(the optimizer is automatically zero-sum, as Session 3 observed). The
slow growth of F with K (0.50 → 0.59 over 250 → 1250, ≈ linear in
log K) is where the measured log-drift of λ_min·N² lives: in the
kernel's K-dependence (the H_{⌊j/d⌋} structure), not in the window
width. What remains for §5 is now sharply localized: prove
αᵀΨα ≥ c‖α‖² for the explicit kernel ψ(d) = (2d/16)(H_{⌊j/d⌋} −
H-window-reference + …) — the −|d| part has the elementary
partial-sums proof (αᵀ(−|i−j|)α = 2ΣS_m² ≥ ‖α‖²/2 on zero-sum), and the
d·log d part is the remaining delicate step.

## 5. Proof program for λ_min ≍ N⁻² (· slowly varying)

The pieces assemble as follows. On zero-sum α (forced, else the C₂/max
rank-one-like part dominates):

1. The C₂/max(j,k) part contributes only through its variation across the
   window — a max-type kernel of the same O(|j−k|/K²) size as φ.
2. What remains is (1/K²)·αᵀΨα with Ψ_{jk} = ψ(|j−k|) explicit
   (ψ(d) = φ(d) + max-correction, ψ(d) ~ d(a′ − b′ log d)).
3. Kernels −|j−k| are conditionally positive definite (Brownian-bridge
   covariance on zero-mean vectors), and −d log d likewise on the relevant
   scale; hence αᵀΨα ≳ c‖α‖² on zero-sum α with c bounded below
   independent of w, giving λ_chain ≍ 1/K² with the constant given by a
   one-dimensional variational problem over the envelope (whose slow
   w-dependence plausibly produces the observed log-drift of λ_min·N²:
   1.37 → 1.55 over N = 500…2500).

Each step is elementary analysis. **The upper bound is now Theorem 1
(§0)** — it required only the d = 1 disagreement integral, evaluated
exactly as a harmonic sum, no kernel asymptotics needed. What remains for
λ_min ≍ N⁻²·(slowly varying) is the matching **lower bound**: steps 1–3
above with honest error terms (the conditionally-positive-definite kernel
argument), plus extending the kernel control from the chain subspace to
all of G_N (the measured factor-1.2 gap).

**Consequence if completed:** κ(G_N) ≍ N² (up to slowly varying factors) —
a deterministic, arithmetic conditioning law for the Nyman–Beurling basis,
in contrast to the random extreme-value conditioning of the zeta-zero Gram
matrix studied in the first phase of this repo. To check against the
literature: the ill-conditioning of the NB system is folklore, but we have
not found a stated λ_min ≍ N⁻² law or the doubling-chain mechanism.
(Literature check pending — flag, not claim.)

## 6. THEOREM 2: the chain-subspace lower bound (July 2026)

*Status: Theorem 2′ (crude constants) is fully proved below — every step
elementary, nothing deferred. Theorem 2 (sharp constants, effective at
computational K) is proved modulo one named lemma (§6.7), whose statement
is precise and whose content is verified numerically across the full
measured range. Numerical pre-verification of every inequality:
`~/rh_output/tier1_verify.txt` (identity to 5×10⁻¹⁵; final bound
0.237/K² against measured floor 0.593/K² at K = 1250, w₀ = 32).*

### 6.0 Statement

Fix a window width w₀ ≥ 4. For K ≥ w₀ + 1 let W = {K−w₀+1, …, K} and
let f_j = e_{2j} − e_j/2 be the doubling-chain differences. Write
Q(α) = ‖Σ_{j∈W} α_j f_j‖² = αᵀMα.

**Theorem 2 (sharp form; modulo the Tail Variation Lemma, §6.7).**
There is an explicit K₀(w₀) of polynomial size such that for K ≥ K₀ and
ALL α ∈ ℝ^{w₀} (no zero-sum restriction):

    Q(α) ≥ [ (log K + γ + c* − log w₀ − 1)/2 + 1 − o_K(1) ] · ‖α‖² / (8K²),

c* = 1 + log(2/π). (At K = 1250, w₀ = 32 the bracket/16K²-form
evaluates to 0.237/K², measured floor 0.593/K² — valid, factor 2.5.)

**Theorem 2′ (unconditional, proved in full below).** For every fixed
w₀ ≥ 4 and all K ≥ exp(4w₀²):

    Q(α) ≥ (log K) · ‖α‖² / (64 K²).

**Corollary (with Theorem 1).** The doubling-chain family's spectral
floor is Θ(K⁻² log K): Theorem 1's four-dilation witness is
order-optimal among all chain combinations, and the measured factor
≈ 1.2 between the chain floor and λ_min(G_N) is the entire remaining
gap between this theorem and the full matrix.

*(What Theorem 2 does NOT do: bound λ_min(G_N) below — restriction to a
subspace bounds λ_min only from above. The full-matrix floor is the
Mellin program, plans/05–06 Tiers 2–3.)*

### 6.1 Step 1 — exact reduction on zero-sum vectors

M_{jk} = (log2/8)(1/j + 1/k) − P_{jk}/16 exactly (§4). For any g and
any zero-sum α, Σ_{j,k} α_jα_k (g(j) + g(k)) = 2(Σα)(Σαg) = 0. Hence

    Σα = 0  ⟹  Q(α) = −(1/16) Σ_{j≠k} α_jα_k P_{jk}.       (T2.1)

No approximation; verified at 5×10⁻¹⁵.

### 6.2 Step 2 — two-sided elementary control of P

For j < k in W, d = k − j (1 ≤ d < w₀), m₀ = ⌊j/d⌋: the disagreement
set below U₀ = j(m₀+1) is exactly the disjoint union ⋃_{m≤m₀}[jm, km)
(the Theorem-1 lemma, verbatim with general d in place of 1), so

    P_{jk} = (2d/(jk))·H_{m₀} + tail_{jk},   0 ≤ tail_{jk} ≤ 2/U₀ ≤ 2d/j².   (T2.2)

Writing H_{m₀} = log K + γ − log d + h_{jk}, elementary estimates give
|h_{jk}| ≤ 3w₀/(K − 2w₀) =: h̄ (log(K/j), the floor in m₀, and
H_m − log m − γ ∈ (1/(2m+1), 1/(2m)) each contribute ≤ w₀/(K−2w₀)).

### 6.3 Step 3 — the two positivity identities (zero-sum α, S_m = Σ_{i≤m}α_i)

(a) **Partial sums.** αᵀ(−|j−k|)α = 2Σ_{m<w₀} S_m², and
‖α‖² = Σ(S_m − S_{m−1})² ≤ 4ΣS_m², hence αᵀ(−|j−k|)α ≥ ‖α‖²/2.

(b) **Box overlap.** (s − |a−b|)₊ = ∫ 𝟙[a,a+s](u)·𝟙[b,b+s](u) du, so
the kernel (s−d)₊ is PSD for every s ≥ 0; and min(d, s) = s − (s−d)₊
has zero diagonal, so on zero-sum α:
Σ_{j,k} α_jα_k min(d_{jk}, s) = −αᵀ(s−d)₊α ≤ 0.

### 6.4 Step 4 — assembling the logarithm

For integers 1 ≤ d < w₀: d·log(w₀/d) = ∫₁^{w₀} min(d,s) ds/s − (d−1).
Summing against α_jα_k over j ≠ k (all kernels below have zero
diagonal; Σ_{j≠k}α_jα_k(d−1) = αᵀ(−D)α·(−1)… = −2ΣS² + ‖α‖² on
zero-sum):

    −Σ_{j≠k} α_jα_k d·log(w₀/d)
        = ∫₁^{w₀} αᵀ(s−d)₊α ds/s − 2ΣS_m² + ‖α‖².

Therefore, with L′ = log K + γ (splitting −d(L′−log d) =
−(L′−log w₀)d − d log(w₀/d)):

    −Σ_{j≠k} α_jα_k d(L′ − log d)
        = (L′ − log w₀)·2ΣS² + ∫₁^{w₀} αᵀ(s−d)₊α ds/s − 2ΣS² + ‖α‖²
        ≥ (L′ − log w₀ − 1)·‖α‖²/2 + ‖α‖²,                    (T2.3)

for L′ ≥ log w₀ + 1, discarding the (nonnegative) box integral.

### 6.5 Step 5 — zero-sum assembly

From (T2.1), (T2.2), h̄, and jk = K²(1+η), |η| ≤ 2w₀/K:

    16·Q(α) = Σ (2d/(jk))(L′ − log d + h_{jk}) (−α_jα_k) − Σ α_jα_k·tail_{jk}
      ≥ (2/K²)(1 − 2w₀/K)·[(L′ − log w₀ − 1)/2 + 1]·‖α‖²
        − (2/K²)·h̄·w₀²·‖α‖²  −  E_tail,

using |Σ_{j≠k}α_jα_k d·h| ≤ h̄·w₀²‖α‖² (since Σ_{j≠k}|α_j||α_k|d ≤
w₀·(Σ|α|)² ≤ w₀²‖α‖²). The tail term is where the two versions part:

- **Crude (Theorem 2′):** |E_tail| ≤ (2d/j²-bound) ⟹ ≤ (2/K²)(1+o(1))·
  w₀²‖α‖². For log K ≥ 4w₀² the main term dominates and
  16Q ≥ (2/K²)·(log K)/4·‖α‖², i.e. Q ≥ log K·‖α‖²/(32K²) on zero-sum
  (the stated /64 absorbs §6.6). ∎ (Theorem 2′, zero-sum part.)
- **Sharp (Theorem 2):** the Tail Variation Lemma (§6.7) says
  tail_{jk} = (2d/(jk))·(c* − γ + τ_{jk}) with |τ| ≤ A·d/j. The
  constant part joins L′ (γ cancels, c* enters — this is where the
  sharp constant comes from); the variation part contributes
  ≤ (2/K²)·(A w₀³/K)·‖α‖² — negligible. Assembly then gives the
  Theorem 2 display. ∎ (modulo §6.7)

### 6.6 Step 6 — removing the zero-sum restriction

Split α = β + t·𝟙/√w₀, β ⊥ 𝟙. Since M is a Gram matrix (PSD), and:
(i) Q(𝟙/√w₀) = (log 2/4)(w₀/K)(1 + O(w₀ log K/K)) — order w₀/K, far
above the zero-sum floor's (log K)/K²; (ii) row sums of M vary across
the window by V = O(w₀² log K/K²), so on zero-sum β,
|B(β, 𝟙/√w₀)| ≤ ‖β‖·√w₀·V (only the variation couples: Σβ = 0);
(iii) AM–GM absorbs the cross term: for K larger than an explicit
polynomial threshold in w₀, Q(α) ≥ ½·min(Q_zero-sum-floor,
Q(𝟙)-direction)·‖α‖² ≥ the stated bounds with the /2 absorbed in the
constants. Elementary; constants tracked in the ledger below. ∎

### 6.7 The Tail Variation Lemma — PROVED (§7)

**Lemma.** For coprime j < k with d = k − j ≤ j/20:

    tail_{jk} · jk/(2d) = c* + τ_{jk},   0 ≤ τ_{jk} ≤ A·d/j,

with A absolute and effective (the ledger of §7 gives a crude A;
the measured worst case over all tested pairs is 0.93, and the
observed exact second-order law is τ = (j mod d)/j + O((d/j)²)).
General pairs reduce by the gcd scaling P(gj, gk) = P(j,k)/g.
Equivalently: P·jk/(2d) = H_{⌊j/d⌋} + c* + τ — the Session-7 law
with explicit error. Proof: §7 below. Theorem 2's debt is retired.

*Correction note (repo discipline): an earlier version of this
subsection stated the constant as c* − γ; the correct constant is c*
(the Session-7 anchor already showed tail·jk/2 = 0.5485 = c* at the
Theorem-1 pair). Caught while writing §7; the Theorem 2 assembly in
§6.5 always used the correct bracket (log K + γ + c* − log d) and is
unaffected.*

### 6.8 Validation ledger

- (T2.1) identity: 5×10⁻¹⁵ (3 random zero-sum draws, exact M and P).
- Step 3a identity and inequality: exact on 5 random draws.
- Step 3b PSD: min eigenvalue ≥ −10⁻⁹ at s = 1, 3, 7.5, 20, 31.
- Final sharp bound vs measured zero-sum floor: 0.237/K² ≤ 0.593/K²
  (K = 1250, w₀ = 32) — valid, factor 2.50 slack.
- Error envelope: max |E_{jk}|·K³/d² = 2.02 over the window (the
  claimed O(1)).

## 7. Proof of the Tail Variation Lemma (July 2026)

*Validation of every identity below: `tvl_check.py`, log
`~/rh_output/tvl_check.txt` — the collapse identity at six pairs
(inside rigorous truncation brackets), the tent identity to 3×10⁻¹⁵,
the limit law with measured deviations exactly matching (j mod d)/j.*

Throughout: j, k coprime, d = k − j, m₀ = ⌊j/d⌋, U₀ = j(m₀ + 1),
λ-variable λ = ud/(jk), f(λ) = dist(λ, 2ℤ), c* = 1 + log(2/π) =
∫₁^∞ f(λ)λ⁻² dλ (evaluated exactly in §4.1: telescoping + Wallis).

### 7.1 Interval representation and two boundary miracles

Δ(u) = ⌊u/j⌋ − ⌊u/k⌋ = Σ_{ℓ≥1} 𝟙_{I_ℓ}(u), I_ℓ = [jℓ, kℓ) (length
dℓ): each ℓ counts iff jℓ ≤ u < kℓ. Two exact facts position the
tail integral perfectly:

(a) k·m₀ = jm₀ + dm₀ ≤ jm₀ + j = U₀ — every interval I_ℓ with
    ℓ ≤ m₀ ends at or before U₀;
(b) I_{m₀+1} begins exactly at j(m₀+1) = U₀.

So on [U₀, ∞) only the intervals I_ℓ with ℓ ≥ m₀ + 1 appear, none
clipped at U₀. (This is why Theorem 1/2's head-tail split at U₀ is
the right split.) The multi-intersections: for ℓ₁ < … < ℓᵣ,
⋂ I_{ℓᵢ} = [jℓᵣ, kℓ₁) when jℓᵣ < kℓ₁ (middles impose no constraint),
which for ℓ₁ ≥ m₀ + 1 lies entirely in [U₀, ∞) by (a)–(b).

### 7.2 The product expansion and the binomial collapse

(−1)^{Δ(u)} = Π_ℓ (1 − 2·𝟙_{I_ℓ}(u)) pointwise (finitely many factors
≠ 1). Fix X > U₀ and expand over [U₀, X]:

    ∫_{U₀}^X (−1)^Δ u⁻² du = ∫_{U₀}^X u⁻² du
        + Σ_{S≠∅} (−2)^{|S|} · ν(⋂_{ℓ∈S} I_ℓ ∩ [U₀, X]),

ν = the u⁻²du measure; the sum is finite at finite X. Group subsets by
(min, max) = (ℓ, ℓ+t): middles are free, giving C(t−1, r−2) subsets of
size r, and

    Σ_r (−2)^r C(t−1, r−2) = 4(1−2)^{t−1} = 4(−1)^{t−1}  (t ≥ 1);
    coefficient −2 for t = 0.

With ν(I-intersection) = w(ℓ,t) := 1/(j(ℓ+t)) − 1/(kℓ) (positive iff
t < T_ℓ := dℓ/j), and tail = ∫(1 − (−1)^Δ)u⁻²du:

    tail = − Σ_{ℓ≥m₀+1} [ −2·w(ℓ,0) + 4Σ_{1≤t<T_ℓ} (−1)^{t−1} w(ℓ,t) ]
           + (X-boundary),                                        (7.1)

where the X-boundary (straddling intersections plus the ∫u⁻² mismatch)
is O(d²/(j²k))·(1 + o(1)) → 0 as X → ∞ relative to the target scale
2d/(jk) — the straddler count at height X is ≈ Xd/(jk) with individual
ν-mass ≤ dℓ/X², totalling O(d²/(j²k)) uniformly in X, and one takes
X → ∞ along kℤ to empty it. [Verified: (7.1) reproduces the exact
tail within rigorous truncation brackets at six pairs.]

### 7.3 The tent identity

Write w(ℓ,t) = (d/(jk))·(1/ℓ)·(1 − t/T_ℓ)·(ℓ/(ℓ+t)) and split off the
main part (the factor ℓ/(ℓ+t) → 1; the correction is §7.4). With
G(T) := Σ_{1≤t<T} (−1)^{t−1}(1 − t/T), an exact two-case computation
(n = ⌊T⌋, θ = {T}; Σ(−1)^{t−1} and Σ(−1)^{t−1}t in closed form) gives

    2G(T) − 1 = − f(T)/T,     f = dist(·, 2ℤ)                     (7.2)

— the triangle wave, exactly, not asymptotically. [Verified 3×10⁻¹⁵.]
Hence the main part of (7.1) is

    (2d/(jk)) Σ_{ℓ>m₀} (1/ℓ)·(1 − 2G(T_ℓ)) = (2d/(jk))·(j/d)·Σ_{ℓ>m₀} f(T_ℓ)/ℓ².

### 7.4 The two error sources (both O(d/j) relative)

**(i) The ℓ/(ℓ+t) correction.** Per ℓ, the correction to the t-sum is
Σ(−1)^{t−1}g(t), g(t) = (1 − t/T)·t/(ℓ+t), with g(1) = O(1/ℓ),
g(T⁻) = O(d/j · d/j), and ∫|g″| = O(1/ℓ). The alternating
Euler–Maclaurin bound |Σ(−1)^{t−1}g − ½g(1) ∓ ½g(end)| ≤ C∫|g″|
(pair, mean-value, telescope) gives per-ℓ mass O(1/ℓ), hence total
(4d/(jk))·Σ_{ℓ>m₀} O(1/ℓ²) = (2d/(jk))·O(d/j). ∎

**(ii) Sum vs integral.** h(x) = f(dx/j)/x² has |h′| ≤ (d/j)/x² + 2/x³
(f is 1-Lipschitz), so |Σ_{ℓ>m₀} h(ℓ) − ∫_{m₀+1}^∞ h| ≤ ½∫_{m₀}^∞|h′|
= O((d/j)²). Substituting λ = dx/j: ∫_{m₀+1}^∞ h = (d/j)∫_{λ₁}^∞
f(λ)λ⁻²dλ with λ₁ = d(m₀+1)/j = 1 + (1−θ)d/j, θ = {j/d}; and
∫_{λ₁}^∞ = c* − ∫_1^{λ₁} f λ⁻² = c* + O(d/j) (f ≤ 1). ∎

### 7.5 Assembly

    tail = (2d/(jk))·(j/d)·[(d/j)(c* + O(d/j)) + O((d/j)²)] + errors (7.4)
         = (2d/(jk))·(c* + O(d/j)),

i.e. tail·jk/(2d) = c* + τ, |τ| ≤ A·d/j with A absolute, effective,
and assembled from: the λ₁-offset (≤ 1), the Riemann-sum bound (≤ 2),
the alternating correction (≤ C), the X-boundary (≤ 1) — a generous
ledger gives a one-digit A; the measured worst case is 0.93. Positivity
of τ (observed) and the exact second-order law

    τ = (j mod d)/j + O((d/j)²)     [observed to (d/j)²; not derived]

are recorded from the validation table (every tested (j,d) matches
{j/d}·d/j to the displayed digits). ∎ **(Tail Variation Lemma)**

With §7, Theorem 2 (sharp form) holds unconditionally for K ≥ K₀(w₀)
of polynomial size. The chain-subspace floor Θ(K⁻² log K) is now a
theorem with no outstanding debts.
