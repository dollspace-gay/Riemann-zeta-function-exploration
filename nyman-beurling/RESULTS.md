# Results Log

## Session 1 — June 9, 2026: first computation, N ≤ 500

Code: `compute_dN.py`. Raw output: `~/rh_output/nb_results.txt`; Gram matrix
saved to `~/rh_output/nb_gram.npz`. RTX 3090, float64; the full coprime Gram
build to N = 500 (76,116 pairs) takes 1.1 s.

### Validation (all pass)

| check | scheme | reference | diff |
|-------|--------|-----------|------|
| A(1,1) | 1.260661401498 | log 2π − γ = 1.260661401508 | 1e−11 |
| A(1,2) | 0.772209256 | brute-force integration | 4e−10 |
| A(2,3) | 0.441103509 | brute-force integration | 3e−10 |
| A(2,4) | 0.386104628 | brute force (tests gcd reduction) | 6e−10 |
| ψ₁ at 1, 1.5, 2 | — | π²/6 etc. | ~5e−11 |

Spectral-cutoff sensitivity of d_N²: zero at every N ≤ 500 (κ ≤ 10⁶ —
float64 has headroom to N of several thousand).

### Finding 1: the Burnol/BDBLS rate law, visible by N = 50

d_N² · log N against the conjectured constant C = 2 + γ − log 4π = 0.0461914:

| N | d_N² (0,∞) | d²logN (0,∞) | d²logN (0,1) |
|------|------------|--------------|--------------|
| 10 | 0.0237719 | 0.05474 | 0.05253 |
| 50 | 0.0118691 | 0.04643 | 0.04558 |
| 100 | 0.0101888 | 0.04692 | 0.04618 |
| 200 | 0.0089207 | 0.04727 | 0.04661 |
| 300 | 0.0079949 | 0.04560 | 0.04504 |
| 400 | 0.0077078 | 0.04618 | 0.04563 |
| 500 | 0.0074199 | 0.04611 | 0.04558 |

Both normalizations sit within ~2% of C from N = 50 onward and oscillate
around it. **The fluctuation around C (e.g. the dip at N = 300) is itself
interesting** — the known second-order asymptotics of d_N² carry oscillatory
terms from the zeta zeros; fitting the residual d²logN − C against
zero-frequency oscillations is an obvious next experiment.

### Finding 2: the null directions are (k, 2k) dilation chains — with an exact identity

The λ_min eigenvector of G_500 has PR ≈ 10 and is supported on pairs
(k, 2k) with k near N/2: components at k = 246, 247, 248, 249 paired
against 492, 494, 496, 498, with v_k ≈ −v_{2k}/2.

Mechanism, verified exactly: for t > 1/k both e_k and e_{2k} are pure
power laws (1/(kt), 1/(2kt)), so

    e_{2k} − e_k/2  is supported entirely on (0, 1/k],

and by gcd-scaling its norm is exactly

    ‖e_{2k} − e_k/2‖² = c₀/k,    c₀ = (3/4)(log 2π − γ) − A(1,2) = 0.1732868…

Numerically confirmed to 7 digits at k = 50…249. Pairs with k ∈ (N/2, N]
admissible give single-pair Rayleigh quotients ~ c₀·2/N; the observed
λ_min ≈ 1.3/N² (λ_min·N² = 1.25, 1.31, 1.33, 1.37 at N = 200…500) is a
factor ~N below that — the eigenvector gains the extra power of N by
combining ~10 chain pairs so that the leftover (0, 1/k]-supported
corrections also cancel against each other. This is the direct analog of
the compound-cluster discovery in the zeta-zero Gram project: single-pair
bounds are loose; chains win.

**Conjecture (numerical): λ_min(G_N) ≍ N^{−2}** (possibly with a slowly
varying factor — λ_min·N² drifts 1.25 → 1.37 over N = 200…500), hence
κ(G_N) ≍ N². A proof via the chain structure looks accessible and would
be a clean lemma about the NB basis. Note the contrast with the zeta-zero
Gram matrix, where λ_min was governed by random extreme events; here the
degeneracy is *deterministic and arithmetic* (dilation chains), which is
why the law is clean.

### Finding 3: Möbius structure of the optimal coefficients, sign corrected

The optimal c_k correlate at |r| = 0.90 with −μ(k)(1 − log k / log N) —
the sign is *negative* Möbius, as the identity Σ_{k≤x} μ(k)⌊x/k⌋ = 1
dictates (our first run had the sign backwards in the comparison; the data
corrected us, and the derivation confirms the data). Sample at N = 500:
c₁ = −0.978, c₂ = +0.964, c₃ = +0.958, c₄ = +0.063 (μ(4) = 0), c₆ = −0.779.
Squarefree-with-odd/even prime factors alternate as −μ dictates; c_k at
non-squarefree k are near zero but not exactly zero (c₄ = 0.063) — the
deviation of c from the −μ profile is exactly where the second-order
structure of d_N lives.

### Next steps

1. Extend to N = 2000–5000 (cost ~N³ in the Gram build, ~70 s at N = 2000;
   float64 conditioning still fine). Sharper d²logN − C residual.
2. Fit the residual d_N²logN − C against oscillations cos(γ_j log N) from
   the first zeta zeros — if the zeros are visible in the residual, we are
   literally watching RH data in an RH-equivalent quantity.
3. Prove λ_min ≍ N^{−2} via the (k, 2k)-chain structure (candidate Lean
   target after paper proof: the support identity and the c₀/k norm are
   exact algebra over our existing formalized machinery's style).
4. Residual function profile r_N(t): where the unapproximable mass lives.
5. Deviation field δ_k = c_k + μ(k)(1 − log k/log N): structure?
