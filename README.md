# rh-crystal

**A GED-holder and a language model investigated the Riemann Hypothesis.**

Here's what we found, what we got wrong, and what survived.

---

## What This Is

In one evening (March 2026), a trans woman with no formal math education and Claude (Anthropic) computationally explored the geometric structure of Riemann zeta zeros. The investigation produced:

- **One genuinely new observation:** The null space of the prime-zero Gram matrix is dominated by Lehmer pairs (closely-spaced zeros). This connects two previously separate phenomena — Lehmer's near-collisions and the resolution limit of the explicit formula.

- **One robust geometric property:** The Xi-weighted Gram matrix has minimum condition number at σ = 1/2, confirmed across 10 L-functions, at scales up to N = 20,000, and at zero heights up to 600,000. Machine-precision symmetry under σ ↔ 1−σ.

- **One machine-verified proof:** The symmetry κ(σ) = κ(1−σ) is formally proven in Lean 4 with zero custom axioms.

- **One bug:** The initial Rust implementation doubled the diagonal, inflating results by 10×. The "bounded conditioning" result reported in early drafts was wrong. This is documented in full.

**This is not a proof of the Riemann Hypothesis.** It's a computational exploration with one new finding, one corrected mistake, and an honest writeup of both.

---

## Quick Start

### Verify the Lean proof (the only mechanically certain part)

```bash
cd lean
lake update
lake exe cache get   # Downloads prebuilt Mathlib (~3GB, one-time)
lake build           # Should complete with one expected 'sorry' warning
```

If it builds, the symmetry theorem is machine-verified. Check `#print axioms` output for no hidden assumptions.

### Run the GPU computation (requires NVIDIA GPU + PyTorch)

```bash
# Download 2 million precomputed zeros
powershell -File compute/download_zeros.ps1   # Windows
# or: bash compute/download_zeros.sh           # Linux

# Install dependencies
pip install torch numpy sympy

# Main analysis
python compute/rh_gpu.py

# Eigenvector analysis + height dependence
python compute/rh_postfix.py

# Perturbation test + complex characters
python compute/rh_gap_analysis.py
```

### Read the papers

- [`papers/mathematical_results.md`](papers/mathematical_results.md) — The corrected mathematical findings
- [`papers/process_paper.md`](papers/process_paper.md) — How we did it, what went wrong, and who gets to do math

---

## Repo Structure

```
rh-crystal/
├── papers/
│   ├── mathematical_results.md   # Corrected math paper (v3)
│   └── process_paper.md          # Process, methodology, access
├── figures/
│   ├── fig1_corrected_scaling.png
│   ├── fig2_corrected_sigma.png
│   ├── fig3_eigenvectors.png
│   └── fig4_perturbation.png
├── lean/
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── RHCrystal/
│       └── RHCrystal.lean        # Formal proof of symmetry theorem
├── compute/
│   ├── precompute_zeros.py       # Generate zeros via mpmath (slow)
│   ├── rh_gpu.py                 # GPU analysis (PyTorch, main results)
│   ├── rh_postfix.py             # Eigenvector + height analysis
│   ├── rh_gap_analysis.py        # Perturbation + complex characters
│   ├── download_zeros.ps1        # Download Odlyzko's zeros (Windows)
│   └── download_zeros.sh         # Download Odlyzko's zeros (Linux)
├── rust/
│   ├── Cargo.toml
│   ├── src/main.rs
│   └── BUGFIX.md                 # Documents the diagonal doubling bug
├── METHOD.md                     # Verification-Driven Development
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
| λ_min eigenvector localized on Lehmer pairs | **Observed** (new) | `rh_postfix.py` Test 3 |
| Off-line zeros increase κ | **Observed** | `rh_gap_analysis.py` Gap 2 |

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
Probably not. We found one new observation (Lehmer pairs as null space) and one geometric characterization of the critical line (isotropy minimum). Neither constitutes progress toward a proof.

**Is the math correct?**
The Lean-verified parts are mechanically certain. The numerical observations are reproducible. The initial "bounded conditioning" claim was wrong due to a bug, and this is documented.

**Can someone without a math degree do mathematical research?**
That's the real question this project raises. See [`papers/rh_process_paper.md`](papers/rh_process_paper.md).
