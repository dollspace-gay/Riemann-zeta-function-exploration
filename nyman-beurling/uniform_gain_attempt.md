# Uniform residual gain: first proof attempt

September 10, 2026. **The requested uniform gain bound is not proved.**
This records the arguments attempted, an unconditional obstruction to one
shortcut, and the precise missing information. The earlier phrase “the
remaining RH step” denotes a sufficient theorem that would prove RH; it
does not mean that the theorem is a routine consequence of the jump bound.

Use E_N, M_N, q_N, and G_N from [the refinement note](refinement_theory.md).
The proposed sufficient target is

    G_N = q_Nᵀ M_N⁻¹ q_N ≥ c E_N/log N

for every sufficiently large dyadic N, with a fixed c > 0. There is no
claimed proof of this statement here, nor an asymptotic counterexample to it.

## 1. What the coefficient bound does not supply

The proved coercivity estimate bounds the norm of a finite synthesis from
below in terms of its coefficients. It does not bound the correlation of
that synthesis with χ − P_Nχ. Cauchy–Schwarz gives an upper bound on a
correlation. Reversing it without an additional argument would be an error.

The distinction can be made exact. Let W be the closed span of all e_k,
and decompose the actual residual as

    r_N = r_∞ + s_N,
    r_∞ = χ − P_Wχ,   s_N = P_Wχ − P_Nχ.

Then r_∞ is orthogonal to every old and new direction, and

    E_N = ‖r_∞‖² + ‖s_N‖².

Every measured gain comes from s_N. A frame estimate applying only to
vectors in W cannot silently be applied to all of r_N: doing so assumes
away the component whose vanishing is the RH criterion.

Equivalently, Gram–Schmidt gives an orthonormal family v_k and numbers
a_k = ⟨χ,v_k⟩ such that

    E_N = 1 − Σ_(k≤N) |a_k|²,
    E_N − E_2N = Σ_(N<k≤2N) |a_k|².

Bessel gives Σ |a_k|² ≤ 1. The missing saturation Σ |a_k|² = 1 is the
same closure problem. A finite spectral lower bound does not supply it.
The 1024-dimensional run captures 0.9934712795118079 of the target's
squared norm; a finite sum does not determine the infinite remainder.

## 2. An unconditional obstruction to using chains alone

**Proposition.** For every finite real or complex linear combination
g = Σ α_k f_k, where f_k(t) = e_2k(t) − e_k(t)/2,

    ‖χ − g‖² ≥ 1 − 2(log 2)² > 0.

This is a written proof, not an additional Lean theorem or a novelty claim.

**Proof.** Put L = log 2. For 0 < t ≤ 1, write uniquely, apart from
irrelevant endpoints,

    t = 2^(−j) x,    j ≥ 0,    1/2 < x ≤ 1,

and set h(t) = 1 − L/x. Set h(t) = 0 for t > 1. The function is bounded
and supported in (0,1], hence belongs to L². It satisfies

    h(t/2) = h(t)       for 0 < t ≤ 1,
    ∫_(1/2)^1 h(x) dx/x = L − L = 0.

Changing variables u = 2t and using e_k(u) = 1/(ku) for u > 1 gives

    ⟨f_k,h⟩
      = (1/2) ∫₀∞ e_k(u)[h(u/2)−h(u)] du
      = (1/(2k)) ∫₁² h(u/2) du/u
      = 0.

All these integrals exist by L² Cauchy–Schwarz; the final one is on a
compact interval. Summing the geometric weights 2^(−j) gives

    ⟨χ,h⟩ = 2 ∫_(1/2)^1 (1−L/x) dx = 1−2L²,
    ‖h‖² = 2 ∫_(1/2)^1 (1−L/x)² dx = 1−2L².

The latter is positive because h is not zero almost everywhere. Thus
⟨χ−g,h⟩ = ‖h‖², and Cauchy–Schwarz proves the proposition. The complex
case follows by the same inner-product argument. ∎

The lower bound is approximately 0.0390939721635972. Solving the
chain-only projection problem with 16, 32, 64, 128, 256, and 512 chains
gives errors 0.116082, 0.113696, 0.112640, 0.111973, 0.111670, and
0.111426. These finite values respect the proved bound; no limiting value
is inferred from them.

**Scope of the obstruction:** it prevents an approximation scheme starting
from zero and using only the f_k from converging to χ. It does not refute
the proposed gain bound evaluated at the *fully optimized* r_N. Full
reprojection introduces directions outside the chain span. The support
argument is another reason not to replace full reprojection with successive
corrections confined to shrinking tails.

## 3. Explicit directions: checked, but no uniform argument

For any nonzero vector v the best scalar correction in that direction gains

    (q_Nᵀv)²/(vᵀM_Nv).

The script `gain_attempt.py` screens five choices on every integer
16 ≤ N ≤ 512, using the existing 1024 Gram matrix. The following are
minima of log(N) times that gain divided by E_N over this finite range:

| Direction on the new parents | Minimum | N attaining it |
|---|---:|---:|
| Previous optimal coefficients | 0.0149722 | 81 |
| μ(k) | 0.0379382 | 200 |
| μ(k) log(2N/k) | 0.0357401 | 229 |
| q_N | 0.0233224 | 42 |
| diag(M_N)⁻¹q_N | 0.0245236 | 42 |

The full projection's corresponding finite minimum is 0.401684 at N=42.
These are positive finite observations. They neither prove a positive
asymptotic infimum nor refute these directions asymptotically. In particular,
the argument still lacks a lower bound for the signed quantity q_Nᵀv.

Reproduce from `nyman-beurling/`:

```bash
OPENBLAS_NUM_THREADS=1 python gain_attempt.py \
  --gram ../Riemann_Continuation_Verification/nb_gram_refinement_1024.npy \
  --first 16 --last 512 \
  --output ../Riemann_Continuation_Verification/gain_candidate_screen.json
```

The run checks the Cholesky prefix identity, independently solves each old
projection, verifies directional gains do not exceed the full gain, and
checks the three elementary witness integrals by independent quadrature.
It records matrix/source hashes and library versions. No spectral cutoff,
zeta zeros, or RH assumptions enter the numerical calculation.

## 4. A precise prime connection for the tapered trial function

There is an elementary bridge to the new prime-based files, although it
does not identify their kernel with the Gram form. Let

    ψ(x) = Σ_(m≤x) Λ(m),
    θ_N = Σ_(k≤N) μ(k) log(N/k)/k,
    c_k^trial = −μ(k) log(N/k)/log N.

For 1 ≤ u ≤ N, the trial residual is exactly

    R_N^trial(u) = [θ_N u − ψ(u)]/log N.             (1)

To prove this, sum the divisor identities

    Σ_(d|m) μ(d) = 1_(m=1),
    Σ_(d|m) μ(d) log d = −Λ(m)

over m ≤ u. They give Σ μ(k) floor(u/k) = 1 and
Σ μ(k) log(k) floor(u/k) = −ψ(u). Insert them into
R(u) = 1 + Σ c_k floor(u/k) − uΣ c_k/k. Consequently the trial's
head energy is

    ∫₁ᴺ |R_N^trial(u)|² du/u²
      = [1/log²N] ∫₁ᴺ |θ_N − ψ(u)/u|² du.           (2)

The prime sum in the new Suzuki function is also an exact transform of ψ:

    Σ_(m≤e^t) Λ(m)/√m (t−log m)
      = ∫₁^(e^t) ψ(x) x^(−3/2)[1+(t−log x)/2] dx.  (3)

Equation (3) is Stieltjes integration by parts; its endpoint terms vanish.
Thus both constructions involve the Chebyshev error, through different
operations: a square integral in (2), and a signed smoothed integral in (3).
Neither operation bounds the other in the direction needed here without
additional estimates. In particular, (1) concerns the explicit trial
coefficients, not the optimal c_N in the proposed gain bound.

`verify_prime_bridge.py` independently builds the von Mangoldt sieve and
checks (1) and (2) for seven N through 1024, at integers and interior
points. It checks (3) at seven times through t=12 by exact-form integration
on unit intervals. The respective largest floating discrepancies in this
run are 8.58e−17, 1.78e−17, and 3.34e−16. These validate implementations of
the identities, not uniform error bounds. Reproduce with

```bash
python verify_prime_bridge.py \
  --output ../Riemann_Continuation_Verification/prime_bridge_checks.json
```

The standard optimal tapered-Möbius theorem explicitly assumes RH and a
bound on Σ 1/|ζ′(ρ)|². Importing it to prove this unconditional gain bound
would be circular. See [Bettin–Conrey–Farmer, Theorem 1](https://arxiv.org/pdf/1211.5191).

## 5. How strong the proposed uniform estimate is

If E_2N ≤ (1−c/log N)E_N holds on all sufficiently large dyadic scales,
then the already proved exponential-product argument yields

    E_N = O((log N)^(−c/log 2)).

This demands a quantitative logarithmic decay rate, not merely convergence.
The unconditional positive lower bound liminf E_N log N > 0 implies
that any such c must satisfy c ≤ log 2. See
[Burnol's lower-bound theorem](https://arxiv.org/pdf/math/0103058).
This restriction does not produce a positive admissible c.

The unresolved step is still a lower bound on correlations of the actual
optimal residual with new directions, strong enough to give nonsummable
fractional gains. The preceding arguments do not establish that bound.
