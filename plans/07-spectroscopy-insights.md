# Plan 7: Spectroscopy Insights — Envelope Classification and Inverse Line Intensities

*July 2026, from doll's sidebar question: "now that RH is related to
spectroscopy, any insights to help solve it?" The proof-relevant answer
is Plans 5/6 (positivity = no perfect absorption). This plan captures
the two INSTRUMENT-grade insights as experiments. Neither claims
anything about ζ itself; both sharpen what our detector can do and what
it could never do.*

## Part A: envelope spectroscopy — what RH-failure would look like

**Idea.** A zero at ½ + iγ produces a line at frequency γ with the
common (decaying) envelope. A zero at β + iγ, β = ½ + δ, would produce
a line whose envelope is anomalous by the factor N^δ — a *brightening*
line. RH, in this instrument's terms: every line has the common
envelope. So: build a synthetic residual curve with RH-consistent lines
at γ₁…γ₅ (envelope exponent matched to the measured γ₁ decay, noise
floor matched to measured) plus ONE planted off-line pair at
γ_f between the real lines with envelope offset δ; run the actual
detector (same drift removal, same LSQ periodogram, windowed
envelopes); classify lines by fitted envelope exponent.

**Deliverables.**
1. Classifier demo: the planted line stands off the common envelope by
   ≈ δ; real-line exponents cluster.
2. Resolution curve: minimum detectable δ vs N-range (contrast N^δ vs
   noise) — the quantitative form of "the spectrometer's β-resolution
   grows only logarithmically with effort," for the process paper.
3. Honesty note in the writeup: synthetic-only; tests the instrument,
   not ζ.

## Part B: inverse spectroscopy — line intensities know ζ′(ρ)

**Idea.** Line intensities carry weights like 1/|ρ_j ζ′(ρ_j)|^p
("transition strengths"). We have measured A₁:A₂:A₃ = 1 : 0.31 : 0.22.
Compute |ζ′(ρ_j)| DIRECTLY (mpmath, never recalled), test the candidate
intensity laws (1/|ρ|², 1/(|ρ||ζ′|), 1/(|ρ||ζ′|)², 1/(|ρ|²|ζ′|)) against
the measured ratios, and — for whichever fits — invert one line to
"measure" |ζ′(ρ₁)| from the distance curve and compare to truth.

**Honesty guard:** three data points against several one-parameter-free
laws is model *screening*, not confirmation; the fitted law becomes the
target for Plan 2 Step 2's derivation, which must then predict it, not
the reverse. Also report the frequency side: the curve already measures
γ₁ itself to 3–4 digits (14.140 vs 14.1347).

## Upgrade path

Both parts rerun on the N = 10⁴ GPU data when available: 5× envelope
range for Part A's classifier and real-data envelope exponents; smaller
error bars on Part B's ratios (γ₄, γ₅ may become measurable, adding
data points against the laws).
