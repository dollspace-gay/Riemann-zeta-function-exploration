# The Amplitude Law of the Zero Oscillations in the Trial-Function Distance

*Working notes, July 2026 (Plan 2 Step 2). Status: derivation at
explicit-formula rigor (RH + simple zeros assumed, convergence
manipulations not yet theorem-grade), confirmed numerically with NO
free parameters. Companion check: `~/rh_output/amplitude_theory_check.txt`.*

## Result

For the Möbius trial coefficients c̃_k = −μ(k)(1 − log k/log N), the
statistic D̃_N²·log N oscillates around its drift with per-zero
amplitude

    A_j(N) = 4·log(2π) · N^{−1/2} / (|ρ_j|² · |ζ′(ρ_j)|),

plus weaker second-order beat lines at frequencies γ_j − γ_k. The same
computation reproduces Burnol's constant C = Σ_ρ 1/|ρ|² as the leading
(non-oscillatory) term — the derivation self-anchors against the known
rate law.

**Measured targets hit (N ≤ 10⁴ GPU data, Session 10):**
- envelope exponent: predicted exactly ½; measured 0.524 (4-octave fit);
- intensity law: predicted 1/(|ρ|²|ζ′|); the law-screen winner at
  log-rms 0.089 (vs 0.430 for 1/|ρ|²);
- absolute normalization 4 log 2π: γ₁ measured/predicted = 1.06, 1.12,
  1.05, 0.97 across four N-octaves (3–12%, no fitted constants); γ₂, γ₃
  consistent within their (low) SNR;
- the stray peak at ≈ 10.9–11.2 in the 10⁴ spectrum matches the
  predicted γ₃ − γ₁ = 10.876 beat line at 2.2× floor (suggestive).

## Derivation

**1. Mellin form.** ê_k(s) = −k^{−s}ζ(s)/s, χ̂ = 1/s, so the residual
transform is (1/s)(1 − ζ(s)M_N(s)) with M_N(s) = Σ_{k≤N} μ(k)
(1 − log k/log N) k^{−s}, and D̃² = (1/2π)∫|1 − ζM_N|² dt/(¼+t²) on
s = ½ + it.

**2. The tapered Möbius sum.** The weight log(N/k)/log N gives
M_N(s) = (1/log N)·(1/2πi)∫ N^w/(w²·ζ(s+w)) dw. Shifting left: the
double pole at w = 0 yields 1/ζ(s)·log N − ζ′(s)/ζ(s)² (main), each
zero contributes N^{ρ−s}/((ρ−s)²ζ′(ρ)). Hence

    1 − ζ(s)M_N(s) = (1/log N)·B(s) + (trivial/truncation errors),
    B(s) = (ζ′/ζ)(s) − Σ_ρ ζ(s)·N^{ρ−s}/((ρ−s)²·ζ′(ρ)).

Near each ρ_j the two singularities cancel: B ≈ (1 − e^{−iτ·log N})/(iτ),
τ = t − γ_j — a Fejér bump of width 1/log N and height log N.

**3. Leading term = Burnol.** ∫|Fejér_j|²·dt/(¼+t²) = 2π·log N/|ρ_j|²
per zero, so D̃² ⊇ (1/log N)·Σ_ρ 1/|ρ|² = C/log N. ✓ (Matches BDBLS/
Bettin–Conrey–Farmer optimality of these coefficients.)

**4. The oscillation.** Terms linear in N^{ρ_j−s} = e^{iγ_j log N}·N^{−it}
in |B|² produce the cos(γ_j log N) component. Using the functional
equation to write conj((ζ′/ζ)(½+it)) = (ζ′/ζ)(1−s), the cross-term
integrand extends meromorphically and the factor N^{−it} = e^{−(s−½)log N}
directs a contour shift to the right. The dominant captured singularity
is **s = 1** (pole of ζ(s) against the weight's 1/(1−s)):

- distance ½ from the critical line ⇒ the envelope factor N^{−1/2};
- the double-pole residue carries a factor log N (cancelling one
  1/log N) and the constant (ζ′/ζ)(0) = log 2π (verified numerically
  to 2e−16);
- the 1/(ρ_j − 1)² = 1/|ρ_j|² and 1/ζ′(ρ_j) factors ride along.

Collecting the conjugate zero ρ̄_j (same frequency, cos even) and the
2Re of the cross term gives the combinatorial factor 4. Result: in
D̃²·log N the j-th line has amplitude 4 log(2π)·N^{−1/2}/(|ρ_j|²|ζ′(ρ_j)|).
Poles of (ζ′/ζ)(1−s) at s = ρ̄_k on the contour generate the
second-order γ_j − γ_k beat lines; s = 1 + 2n (trivial zeros) give
O(N^{−5/2}).

## Honesty box

1. The ρ-sum manipulations (interchange, conditional convergence,
   truncation of the w-contour) are heuristic-explicit-formula grade,
   not theorem grade. The numerical agreement (parameter-free, four
   octaves, three targets) says the answer is right; a rigorous proof
   would follow the BDBLS §-style arguments and is a separate project.
2. The factor 4 is derived from bookkeeping (conjugate pair × 2Re) but
   was CONFIRMED by measurement before that bookkeeping was fully
   trusted — flagged so no one mistakes the order of events.
3. Assumes RH + simple zeros in the usual formal way (as does every
   statement about individual γ_j oscillations).
4. Phase prediction φ_j is derivable from the same residue but has not
   been checked against the fitted phases yet — open item.

## Consequences

- The June zero hunts were doomed twice over: the optimal curve's
  arithmetic noise (rms 0.058) sat ~10³× above these amplitudes at the
  N then available — AND the optimal coefficients differ from the trial
  ones precisely near the zeros, altering (plausibly suppressing) the
  line amplitudes. The trial curve was the right instrument.
- Detecting γ_j at height γ requires N^{1/2} ≳ |ρ_j|²|ζ′(ρ_j)|/floor —
  with our floor ~2e−4, γ₄–γ₅ need N ~ 10⁵–10⁶: feasible overnight on
  the GPU, predicted in advance. A clean falsifiable test of the law.
- The N^{−1/2} is the weight-pole distance ½ — the same ½ as the
  critical line itself: the envelope exponent is literally the width of
  the critical strip crossing, which is a satisfying place for the
  spectroscopy story to land.
