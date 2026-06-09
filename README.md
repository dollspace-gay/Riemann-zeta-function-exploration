**A GED-holder and a language model investigated the Riemann Hypothesis.**

Here's what we found, what we got wrong, and what survived.

---

## What This Is

Started in one evening (March 2026) by a trans woman with no formal math education and Claude (Anthropic), revisited and extended in June 2026. The investigation has produced:

- **One genuinely new observation:** The null space of the prime-zero Gram matrix is dominated by tight *clusters* of zeros, located exactly by a 3-zero window eigenvalue. This connects two previously separate phenomena — anomalously close zeros (Lehmer's phenomenon) and the resolution limit of the explicit formula. (v3 said "Lehmer pairs"; v4 refines this — the famous Lehmer pair actually *loses* to a compound cluster at N=8000. See the revision history in the paper.)

- **One mechanistic theory, tested:** The invisible directions are k-th divided differences over zero clusters — the prime sum cannot resolve the k-th derivative of the phase at small scales. Planted-cluster experiments confirm the predicted δ², δ⁴, δ⁶ scaling for pairs/triples/quadruples (measured exponents 1.95/4.01/6.05). Combined with GUE extreme-gap statistics this predicts λ_min ~ N^(−2/(k+2)) per channel, bracketing every measured exponent, and conjectures λ_min = N^(−2/3+o(1)) asymptotically, conditional on the GUE Hypothesis. See [`papers/scaling_theory.md`](papers/scaling_theory.md).

- **One robust geometric property:** The Xi-weighted Gram matrix has minimum condition number at σ = 1/2, confirmed across 10 L-functions, at scales up to N = 20,000, and at zero heights up to 600,000. Machine-precision symmetry under σ ↔ 1−σ. The second-order minimum condition is verified to N = 2000 with 4–5 decimal agreement.

- **Two machine-verified proofs:** The symmetry κ(σ) = κ(1−σ), and the interlacing pair bound λ_min(G) ≤ Σ_p w(p,σ)(1 − cos(δ·log p)) — the latter ties the smallest eigenvalue to close zero pairs. Both formally proven in Lean 4, zero sorries, zero custom axioms.

- **One bug, and one revised claim:** The initial Rust implementation doubled the diagonal, inflating results by 10×; the "bounded conditioning" result in early drafts was wrong. And v3's "Lehmer pair localization" claim was refined to tight-cluster localization in v4 after sliding-window analysis. Both are documented in full — the corrections are part of the record.

**This is not a proof of the Riemann Hypothesis.** It's a computational exploration with a few new findings, a tested mechanism, corrected mistakes, and an honest writeup of all of it.

---

## Quick Start

### Verify the Lean proof (the only mechanically certain part)

```bash
cd lean
lake update
lake exe cache get   # Downloads prebuilt Mathlib (~3GB, one-time)
lake build           # Should complete cleanly (zero sorries)
```

If it builds, both machine-verified results hold: the symmetry theorem κ(σ) = κ(1−σ), and the interlacing pair bound λ_min(G) ≤ Σ_p w(p,σ)(1 − cos((γ_i−γ_j) log p)). Check `#print axioms` output for no hidden assumptions.

### Run the GPU computation (requires NVIDIA GPU + PyTorch)

```bash
# Download 2 million precomputed zeros
bash compute/download_zeros.sh

# Install dependencies
pip install torch numpy sympy

# Main analysis
python compute/rh_gpu.py

# Eigenvector analysis + height dependence
python compute/rh_postfix.py

# Perturbation test + complex characters
python compute/rhgap.py

# v4 experiments: interlacing bound, control ensembles, curvature, height/cutoff
python compute/rh_progress.py

# Sliding-window interlacing (tight-cluster localization)
python compute/rh_windows.py

# Fixed prime-count sweep
python compute/rh_fixedP.py

# Scaling-law theory tests: planted clusters, stencils, GUE ensemble
python compute/rh_theory.py
```

### Read the papers

- [`papers/rh_crystal_v4.md`](papers/rh_crystal_v4.md) — The mathematical findings (current version)
- [`papers/scaling_theory.md`](papers/scaling_theory.md) — Toward deriving the scaling law: stencil theory + GUE extreme-value channels
- [`papers/rh_crystal_v3_final.md`](papers/rh_crystal_v3_final.md) — v3, kept as the historical record
- [`papers/progress_notes_2026-06-09.md`](papers/progress_notes_2026-06-09.md) — Lab notes for the v4 experiments
- [`papers/rh_process_paper.md`](papers/rh_process_paper.md) — How we did it, what went wrong, and who gets to do math

---

## Repo Structure

```
rh-crystal/
├── papers/
│   ├── rh_crystal_v4.md          # Math paper (current, v4)
│   ├── scaling_theory.md         # Scaling-law derivation: stencils + GUE channels
│   ├── rh_crystal_v3_final.md    # v3 (historical record)
│   ├── progress_notes_2026-06-09.md  # Lab notes for v4 experiments
│   └── rh_process_paper.md       # Process, methodology, access
├── figures/
│   ├── fig1_corrected_scaling.png
│   ├── fig2_corrected_sigma.png
│   ├── fig3_eigenvectors.png
│   └── fig4_perturbation.png
├── lean/
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── RHCrystal/
│       └── RHCrystal.lean        # Symmetry theorem + interlacing pair bound
├── compute/
│   ├── precompute_zeros.py       # Generate zeros via mpmath (slow)
│   ├── rh_gpu.py                 # GPU analysis (PyTorch, main results)
│   ├── rh_postfix.py             # Eigenvector + height analysis
│   ├── rhgap.py                  # Perturbation + complex characters
│   ├── rh_progress.py            # v4: interlacing, ensembles, curvature, cutoff
│   ├── rh_windows.py             # v4: sliding-window interlacing
│   ├── rh_fixedP.py              # v4: fixed prime-count sweep
│   ├── rh_theory.py              # Scaling theory: planted clusters, GUE ensemble
│   └── download_zeros.sh         # Download Odlyzko's zeros (Linux)
├── rust/
│   ├── Cargo.toml
│   ├── src/main.rs
│   └── BUGFIX.md                 # Documents the diagonal doubling bug
└── README.md                     # You are here
```

---

## Key Results (Corrected)

| Claim | Status | Evidence |
|-------|--------|----------|
| κ(σ) = κ(1−σ) | **Proven** (Lean 4) | `lean/RHCrystal/RHCrystal.lean` |
| σ = 1/2 minimizes κ | **Observed** (N ≤ 20,000) | GPU results, 10 L-functions |
| σ = 1/2 minimum holds at all heights | **Observed** (γ up to 600k) | `rh_postfix.py` Test 5 |
| Bounded conditioning (κ < 4) | **WRONG** (bug) | See `rust/BUGFIX.md` |
| κ grows as N^{0.61} | **Observed** | `rh_gpu.py` Test 1 |
| λ_min eigenvector localized on tight clusters (v4; v3 said "Lehmer pairs") | **Observed** (new) | `rh_windows.py` |
| λ_min ≤ pair bound B(δ) (interlacing/Rayleigh) | **Proven** (Lean 4) | `lean/RHCrystal/RHCrystal.lean` |
| κ scaling reproduced by GUE, destroyed by Poisson | **Observed** (v4) | `rh_progress.py` Test B |
| Height degradation removable by raising prime cutoff | **Observed** (v4) | `rh_progress.py` Test D |
| Invisible directions = divided-difference stencils (δ², δ⁴, δ⁶ channels) | **Theory + confirmed** (exponents 1.95/4.01/6.05) | `rh_theory.py` Test A |
| λ_min ~ N^(−2/(k+2)) per channel; N^(−2/3) asymptotic | **Conjectured** (GUE ensemble: −0.55…−0.58, crossover) | `rh_theory.py` Test C, `scaling_theory.md` |
| Off-line zeros increase κ | **Observed** | `rhgap.py` Gap 2 |

---

## The Bug

The initial Rust implementation had this inner loop:

```rust
for i in 0..n {
    for j in i..n {
        let val = w * (phases[i] * phases[j].conj()).re;
        g[(i, j)] += val;
        g[(j, i)] += val;  // When i==j, this doubles the diagonal
    }
}
```

When `i == j`, both lines write to `g[(i,i)]`, doubling the diagonal. This is equivalent to adding a scaled identity matrix, which compresses the condition number by ~10×. The GPU implementation uses `wcos @ wcos.T + wsin @ wsin.T` which computes the outer product correctly.

The bug was discovered when scaling from N=5,000 (Rust) to N=20,000 (GPU) produced a 10× discrepancy in condition numbers. All results in the final paper use the corrected GPU implementation.

---

## Data Sources

- **Zeros:** [Odlyzko's tables](https://www-users.cse.umn.edu/~odlyzko/zeta_tables/) — 2,001,052 zeros, free to download
- **Primes:** Generated by sieve, up to 10^7 (664,579 primes)
- **Zero verification:** [LMFDB](https://www.lmfdb.org/zeros/zeta/) — 103.8 billion zeros computed by David Platt

---

## Requirements

- **Lean proof:** [elan](https://github.com/leanprover/elan) (Lean version manager)
- **GPU computation:** Python 3.10+, PyTorch with CUDA, numpy, sympy
- **Zero precomputation (optional):** mpmath (`pip install mpmath`)

---

## License

All code and writing in this repository is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Do whatever you want with it. Cite if you're inclined.

---

## FAQ

**Did you prove the Riemann Hypothesis?**
No.

**Did you make progress toward proving it?**
Toward RH itself, probably not. We found one new observation (tight zero clusters as null space), one geometric characterization of the critical line (isotropy minimum), and a tested mechanism for *why* the clusters dominate (divided-difference stencils), with a conjectured scaling law conditional on the GUE Hypothesis. That's progress on understanding this particular construction — not on RH.

**Is the math correct?**
The Lean-verified parts are mechanically certain. The numerical observations are reproducible. The initial "bounded conditioning" claim was wrong due to a bug, and this is documented.

**Can someone without a math degree do mathematical research?**
That's the real question this project raises. See [`papers/rh_process_paper.md`](papers/rh_process_paper.md).
