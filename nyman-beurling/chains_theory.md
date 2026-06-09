# The Doubling-Chain Structure of the Nyman–Beurling Gram Matrix

*Working notes, June 2026. Companion code: `chains.py`. Status: the
identities of §1–2 are proved (elementary); the kernel law of §4 is a
numerical fit of striking quality; §5 is a proof program, not a proof.*

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

## 4. The kernel law (numerical fit, three-decimal quality)

Empirically, with K = N/2 and j, k in the window:

    M_{jk} = C₂/max(j,k) − φ(|j−k|)/K²,
    φ(d) = d·(a − b·log d),    a = 1.073 ± 0.002,  b = 0.1430 ± 0.0005.

Fit residuals < 0.001 uniformly over d = 1…159 (e.g. predicted φ(d)/d at
d = 40: 0.545, measured 0.544; at d = 159: 0.348 both). The d·log d form
is exactly what a parity-decorrelation computation predicts: the parities
⌊u/j⌋, ⌊u/k⌋ stay aligned up to u ~ K²/d and decorrelate beyond,
producing a −(d/K²)·log-type deficit. The constants a, b should be
derivable from that computation (open item; b is numerically close to 1/7
but we make no claim).

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

Each step is elementary analysis; what is missing for a theorem is the
error-term bookkeeping in the kernel asymptotics (step 4 → rigorous) and
the uniform lower bound in step 3. **Upper bound status:** exhibiting any
fixed zero-sum alternating α (e.g. the measured envelope) in (2) with the
kernel law gives λ_min(G_N) = O(N⁻² log N) modulo only the kernel
asymptotics — this is the nearest-term rigorous target.

**Consequence if completed:** κ(G_N) ≍ N² (up to slowly varying factors) —
a deterministic, arithmetic conditioning law for the Nyman–Beurling basis,
in contrast to the random extreme-value conditioning of the zeta-zero Gram
matrix studied in the first phase of this repo. To check against the
literature: the ill-conditioning of the NB system is folklore, but we have
not found a stated λ_min ≍ N⁻² law or the doubling-chain mechanism.
(Literature check pending — flag, not claim.)
