# Plans — July 2026

Five pathways selected after the June 2026 sessions (external contact
deliberately excluded for now, per doll). Each plan is self-contained:
current state, staged milestones with validation gates, honest
expectations, effort estimates. The repo's standing rules apply to all
of them: derive rather than recall, cross-check every quantity, log
negatives with the same prominence as positives.

| # | Plan | Type | Risk | First session's deliverable | Status (July 20, 2026) |
|---|------|------|------|------------------------------|------------------------|
| 1 | [λ_min lower bound](01-lambda-min-lower-bound.md) | theorem target | medium | c(d) exact + Ψ positivity numerical pre-check | **Session 1 done.** c* = 1 + log(2/π) derived + confirmed (`kernel_cd.py`); exact digamma-period P; zero-sum floor flat in w (lemma premise survives). Next: prove the d·log d positivity step. |
| 2 | [Zero-oscillation amplitude](02-zero-amplitude-estimate.md) | analysis + experiment | low | trial-function periodogram (Experiment 1) | **COMPLETE (detection + derivation).** γ₁–γ₃ detected (SNR 12.8 at 10⁴); Step 2 derived A_j = 4log(2π)N^{−1/2}/(|ρ_j|²|ζ′(ρ_j)|), parameter-free, hitting all three measured targets (`amplitude_theory.md`). Open: phase check; γ₄–γ₅ prediction test at N ~ 10⁵–10⁶. |
| 3 | [Residual profile + Vasyunin](03-residual-profile-and-vasyunin.md) | computation + validation | low | r_N(t) profile with energy-closure gate | **Both parts done.** Mass spread at u ~ N^0.44, not u ~ N; closure 2–7e−4. Vasyunin sourced verbatim, verified 3 ways; both literature flags resolved. |
| 4 | [Lean Theorem 1 end-to-end](04-lean-theorem1-end-to-end.md) | formalization | medium | Mathlib scouting pass + M1 (e_k ∈ L²) | **M1 done.** `NymanBeurlingL2.lean` compiles, zero sorries, in build roots. Scouting: substitution via JacobianOneDim; log-2 series via integral route. Next: M2 (square-wave identity in L²). |
| 5 | [Mellin lower bound](05-mellin-lower-bound.md) | research math | high | coefficient uncertainty lemma + numerics (M1) | **M0 + M1 done.** Lemma proved (τ-weighted, ε-free) in `mellin_lower_bound.md`; saturation vs true eigenvectors only 29–64×. Next: M2 assembly with sourced AFE/MV statements. |
| 6 | [Lower-bound proof](06-lower-bound-proof.md) | theorem target | tiered | Tier 1 numerics gate | **Theorem 2 WRITTEN** (`chains_theory.md` §6): 2′ fully proved; sharp form owes only the Tail Variation Lemma. Chain floor = Θ(K⁻²log K). Next: prove the TVL. |
| 7 | [Spectroscopy insights](07-spectroscopy-insights.md) | instrument | low | classifier + ζ′ screen | **Done.** δ=0.1 classifier at 3.2σ; law A ∝ 1/(\|ρ\|²\|ζ′\|) screened, then DERIVED and confirmed out-of-sample to 10⁶ (Sessions 11–12). |
| 8 | [Redheffer](08-redheffer.md) | exploration | gated | literature gate + Part A | **Gate passed (Hilberdink flagged).** Part A done: left null space = refined Möbius; σ_min flat ≈ 0.005; geometric mult = n/2 − 1 measured. |

## Suggested order

Cheap and load-bearing first: **3A** (residual, never been looked at),
**2 Step 1** (trial-function periodogram, an afternoon), **5 M1** (the
elementary lemma with its numerical check), **1 Step 0** (c(d) exact).
Each of these is one session, each produces a RESULTS.md entry, and
together they de-risk the three big pushes (1 Steps 2–3, 4, 5 M2–M3)
before committing serious effort to any of them.

Interdependencies: 3B (Vasyunin) feeds 1 Step 0; 5 M0 underwrites 2;
a full success of 5 would subsume 1 Step 3. Plans 2, 3, 4 are mutually
independent and parallelizable.
