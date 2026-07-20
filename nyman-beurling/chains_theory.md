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
