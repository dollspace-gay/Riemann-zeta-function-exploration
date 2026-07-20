# Plan 5: The Mellin-Side Lower Bound Attempt

*Target: a rigorous lower bound λ_min(A_N) ≫ N^{−c} for the NB Gram
matrix via the verified spectral identity — the hardest and highest-risk
item on the board, staged so that each milestone is a real result even
if the final push fails. This is genuine analytic number theory: the
recalled-knowledge discipline applies at maximum strength. Every named
theorem below (Montgomery–Vaughan, approximate functional equation,
twisted moments) is RECALLED and must be verified against a source
before its statement is used in anger.*

## The object

Verified numerically (Session 6, 0.06–0.15% on random vectors):

    xᵀA_N x = (1/2π) ∫_ℝ |ζ(½+it)|²/(¼+t²) · |X(t)|² dt,
    X(t) = Σ_{k≤N} x_k k^{−½−it}.

A lower bound inf_{‖x‖=1} ∫ ≥ c·N^{−a} gives λ_min(A_N) ≫ N^{−a}.
Theorem 1 says a = 2 is the truth (up to log). The question is how much
of that known technology can certify.

## Milestone 0: rigor of the identity itself

The identity is currently "derived + numerically verified." Write out
the honest proof: ê_k(s) = −k^{−s}ζ(s)/s on the critical line
(Mellin–Plancherel for L²(0,∞); the literature treats this as standard —
find and cite the precise statement, or prove it for our e_k directly).
This also underwrites Plan 2's Step 2. Small, bounded task.

## Milestone 1 (elementary, provable now): the coefficient uncertainty lemma

The heart of any lower bound is: unit-norm coefficients x cannot make
the Dirichlet-series structure vanish everywhere. Concretely, define for
r ≤ N the divisor partial sums

    s_r = Σ_{d | r} x_d.

For m = q·r with q a prime > N and r ≤ N, the Dirichlet coefficient of
ζ·X at m is exactly s_r (all divisors of m that are ≤ N are the divisors
of r). Möbius inversion x_d = Σ_{r|d} μ(d/r)·s_r plus Cauchy–Schwarz
(x_d² ≤ τ(d)·Σ_{r|d}s_r²) and divisor-sum bounds give

    1 = ‖x‖² ≪ Σ_{r≤N} s_r²·τ(r)·(N/r)·log N   ⟹   Σ_{r≤N} s_r²/r ≫ N^{−1−ε}.

**This lemma is elementary, self-contained, and numerically checkable
today**: compute s_r for the actual λ_min eigenvectors of A_N (the chain
vectors) at N = 500…10⁴ and check Σs_r²/r against N^{−1}. If the chain
vectors saturate the lemma, the lemma is the right skeleton; if they sit
far above it, the lemma is lossy and the loss must be located. Do this
BEFORE any analytic work.

## Milestone 2 (the assembly): a first rigorous polynomial floor

Sketch to be executed with full error terms:

1. Restrict the integral to a window t ∈ [T, 2T]; the weight costs
   ~T⁻². On the window, replace ζ by its approximate-functional-equation
   truncation (length √(T/2π)); ζ·X becomes a Dirichlet polynomial of
   length ~N·√T whose coefficients at the special indices m = qr
   (q prime > N) are the s_r above.
2. Montgomery–Vaughan mean value on the window: for T large relative to
   the polynomial length, the diagonal dominates:
   ∫_T^{2T} |·|² ≈ Σ_m c_m²·m⁻¹·(T + O(m)). Diagonal dominance needs
   T ≫ length ≈ N·√T, i.e. **T ≫ N²**.
3. The prime-times-r indices contribute Σ_r s_r²/r · Σ_{q∈(N, ·]} 1/q,
   and Milestone 1 gives Σ_r s_r²/r ≫ N^{−1−ε}.
4. Totals: λ_min ≳ T⁻² · T · N^{−1−ε} = N^{−1−ε}/T. With T ≍ N^{2+ε}:

       λ_min(A_N) ≫ N^{−3−ε}.

Not the truth (N⁻²), but: **a first unconditional polynomial lower
bound on the spectral floor of the classical NB Gram matrix**, which the
literature check found no trace of at any exponent. Combined with
Theorem 1 it brackets λ_min in [N^{−3−ε}, N^{−2}log N]. This milestone
is a complete, writable result on its own.

Care points (where this can silently break): the AFE error terms
integrated against |X|²; the χ-factor's second AFE sum; the O(m) MV
error against the m ~ N²-sized indices; keeping every constant effective
enough to state the theorem cleanly. Each gets its own numerical
spot-check against the cached ζ-grid before being trusted.

## Milestone 3 (the hard push): N^{−3} → N^{−2−ε}

The loss in Milestone 2 is structural: pushing T up to N² for diagonal
dominance overpays in the t⁻² weight, while the true minimizers (chains)
have their |X|² mass at t ≍ N. Candidate routes, in decreasing
concreteness:

1. **Cauchy–Schwarz with a mollifier:** ∫|ζX|²w ≥ (∫ ζ·X·Ḡ·w)² / ∫|G|²w
   for a chosen G. First moments of ζ against Dirichlet polynomials are
   controllable at much shorter window heights than second moments —
   possibly down to T ≍ N^{1+ε}, recovering the exponent. The game is
   choosing G so the first moment doesn't die against adversarial x
   (G must depend on x — e.g. G built from the s_r skeleton of x).
2. **Twisted second moment technology** (BCHB and successors): known
   ranges for ∫|ζ|²|X|² with X of length N vs window height T — check
   the actual literature for the current admissible length/height
   trade-off before assuming anything (this is exactly the technology
   the outreach draft asks the experts about; we can read it ourselves).
3. **Bilinear/off-diagonal cancellation:** show the off-diagonal MV
   terms are not adversarial for coefficients constrained by unit norm —
   this is research mathematics with real failure probability.

If none close: write up the obstruction precisely ("the missing
ingredient is a lower bound of type L for Dirichlet polynomials of
length N at height N") — that statement is itself the sharpest possible
form of the question for any future expert contact, and Milestone 2
stands as the result.

## Honest expectations

Milestone 0: certain. Milestone 1: near-certain, and independently
interesting (it quantifies "you can't hide from all the primes at
once"). Milestone 2: likely with careful error bookkeeping — this is
the expected headline of the plan. Milestone 3: genuinely uncertain;
timebox attempts and document obstructions. The failure mode to guard
against is narrative momentum around a "nearly complete proof" — the
March lesson. Every analytic input gets source-verified; every
inequality chain gets a numerical instantiation at finite N before
being written into a proof.

## Validation assets already in hand

- Exact A_N to N = 10⁴ and its eigenvectors (the adversarial x's).
- The ζ-grid to T = 2000 (`zeta_grid.npz`) for windowed-integral checks.
- The verified identity itself as the bridge: any claimed lower bound
  on the integral side must, numerically, sit below the true λ_min at
  every accessible N — a hard sanity rail the whole way down.

## Cross-links

Milestone 0 underwrites Plan 2. A Milestone 3 success subsumes Plan 1
Step 3 (full-matrix lower bound). Milestone 1's s_r statistics may also
illuminate Plan 3's residual profile (both are about where x's mass can
hide).

## Effort estimate

M0: half a session. M1: one session including numerics. M2: 2–4
sessions (error bookkeeping dominates). M3: open-ended, timeboxed.
