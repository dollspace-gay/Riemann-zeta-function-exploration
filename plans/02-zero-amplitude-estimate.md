# Plan 2: A Theoretical Amplitude for the Zero Oscillations in d_N²

*Target: close the parked zero hunt properly. Three statistics across two
ranges (N ≤ 10⁴) saw no trace of the zeta zeros in d_N's fine structure,
with a noise floor ≈ 5×10⁻⁵. The missing piece has always been a
prediction: how large SHOULD the cos(γ_j·log N) terms be? Derive that
amplitude. Either outcome closes the question: detectable → design the
experiment that finds them; undetectable → the negative becomes a
theorem-backed statement, not a shrug.*

## The mechanism to make quantitative

In the Mellin picture (verified numerically in Session 6), with
P(s) = Σ_{k≤N} c_k k^{−s} and ê_k(s) = −k^{−s}ζ(s)/s, χ̂(s) = 1/s:

    ‖χ − Σ c_k e_k‖² = (1/2π) ∫ |1 − ζ(½+it)·P(½+it)|² / |½+it|² dt.

Near each zero ρ_j = ½ + iγ_j, ζ vanishes, so the integrand is pinned at
1/|ρ_j|² regardless of P — this is exactly how Burnol's constant
C = Σ 1/|ρ|² arises. The *N-dependence*: a length-N Dirichlet polynomial
has resolution ~1/log N in t, so the width of the uncancellable window
around each γ_j shrinks as N grows, and the local cancellation pattern's
phase rotates like N^{iγ_j} = e^{iγ_j log N}. That rotation is the
candidate source of cos(γ_j log N + φ_j) terms in d_N²·log N − C. The
task is to derive the amplitude A_j(N), not just the frequency.

## Step 1: the trial-function distance (exact, computable, no optimizer noise)

Define D̃_N² = ‖χ − Σ c̃_k e_k‖² for the *explicit* Möbius trial
c̃_k = −μ(k)(1 − log k/log N). This is a plain quadratic form in the
exact Gram data — c̃ᵀAc̃ − 2c̃ᵀb + 1 — cheap at every N ≤ 10⁴ from the
cached Cholesky/Gram files. Properties that make it the right object:

- It upper-bounds d_N² and is known (BDBLS circle of ideas — recalled,
  verify) to achieve the C/log N order.
- It is *smooth in the coefficients*: no minimizer chasing arithmetic
  structure of each N, so its fluctuation should isolate the analytic
  oscillation from the arithmetic noise that drowned the optimal-d_N hunt
  (rms 0.058 on the increments — the killer last time).

**Experiment 1 (cheap, do first):** compute D̃_N²·log N on a dense N-grid
to 10⁴, remove the smooth drift, periodogram in log N. If the zero
frequencies γ_1 = 14.13, γ_2 = 21.02, γ_3 = 25.01 appear HERE, the
mechanism is confirmed and the amplitude is measured directly. If not,
the noise floor of this cleaner statistic is itself informative.

## Step 2: derive A_j(N) analytically

For the trial function, 1 − ζP̃ is explicit enough for classical-style
analysis: write d̃² as a contour integral / Plancherel expression, deform
or expand around each zero, and extract the residue-type contribution.
Expected shape (to be *derived*, not assumed):

    D̃_N²·log N = C + (drift terms in 1/log N) + Σ_j A_j(N)·cos(γ_j log N + φ_j) + …

with A_j(N) presumably carrying both a 1/|ρ_j|² factor and a decay in N
(power of N, or power of log N — this is the whole question). Every step
numerically anchored: evaluate the derived leading term against Step 1's
measured curve at multiple N before trusting any asymptotic manipulation.

## Step 3: the detectability verdict

Compare the derived A_j(N) to:
- the measured arithmetic-noise floor of the optimal-d_N statistics
  (rms 0.058 on increments, ~5×10⁻⁵ periodogram floor at 800 blocks);
- the cleaner floor of the trial-function statistic from Step 1;
- the N-scaling of both under block averaging (noise ∝ 1/√blocks).

Branches:
- **A_j decays like a power of N** → no feasible N reaches it; write the
  closure note in RESULTS.md: "the zeros are provably invisible at any
  accessible scale by these statistics," with the derived bound. Hunt
  closed permanently, honorably.
- **A_j ~ 1/log^a N** → compute the required N and averaging schedule;
  if feasible (even at overnight-run cost), spec the experiment
  (statistic, N-grid, block schedule, expected SNR) before running it.
- **A_j matches Step 1's measurement** → we have already detected the
  zeros in an RH-equivalent quantity's finite-N structure; verify hard
  (phase coherence across N-ranges, all three γ_j, amplitude ratio test
  A_1:A_2:A_3 against the derived 1/|ρ_j|² law) before believing it.

## Honest expectations

Step 1 is an afternoon and cannot fail to produce information. Step 2 is
real analysis with room for wrong recalled asymptotics — the discipline
is the same as the Mellin session: derive, then check every intermediate
object numerically (the zeta grid to T = 2000 is already cached). The
most likely overall outcome is the first branch (a clean impossibility
note), which is worth having in the repo either way.

## Deliverables

- `nyman-beurling/amplitude.py` (Step 1 + validation of Step 2's formula).
- RESULTS.md session entry; a short `amplitude_theory.md` if Step 2's
  derivation is substantial enough to stand alone.

## Cross-links

Uses Plan 5's Step 1 rigor (the Mellin identity's exactness) as its
foundation; shares the cached ζ-grid. Independent of Plans 1, 3, 4.
