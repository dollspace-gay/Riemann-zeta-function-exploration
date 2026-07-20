# RH Crystal / Nyman–Beurling: Lean 4 Formalization

Machine-verified results across both phases of the project. **Zero
sorries in every module; every theorem's axiom audit prints exactly
[propext, Classical.choice, Quot.sound].**

## Modules and what's proven

### `RHCrystal/RHCrystal.lean` (phase 1, March–June 2026)
| Theorem | Meaning |
|---------|---------|
| `xi_weight_symm` | w(p,σ) = w(p,1−σ) |
| `gram_matrix_symm` | G(σ) = G(1−σ), hence κ(σ) = κ(1−σ) |
| `cosh_ge_one` | cosh x ≥ 1 |
| `exists_eigenvalue_mul_le_rayleigh` | Rayleigh: ∃k, λ_k·(x⬝x) ≤ xᵀAx |
| `exists_eigenvalue_le_pair_bound` | pair bound 2λ_k ≤ A_ii + A_jj − 2A_ij |
| `gram_eigenvalue_le_pair_bound` | the paper-v4 §3.3 interlacing bound |

### `RHCrystal/NymanBeurling.lean` (June 2026)
| Theorem | Meaning |
|---------|---------|
| `interval_of_odd` | the disagreement-interval lemma |
| `disagreement_integral_le` | ∫_D u⁻² ≤ (H_J+1)/(J(J+1)) |

### `RHCrystal/NymanBeurlingL2.lean` (July 2026 — Milestones 1–2)
| Theorem | Meaning |
|---------|---------|
| `eDil_memLp_two` | the dilation family e_k ∈ L²(0,∞) |
| `fract_half` | fract(y/2) = fract(y)/2 + (⌊y⌋ mod 2)/2, all real y |
| `chainDiff_eq_squareWave` | f_k = e_{2k} − e_k/2 is the square wave |
| `chainDiff_values`, `chainDiff_eq_zero` | values {0, ½}; support (0, 1/k] |
| `chainDiff_memLp_two` | chain differences ∈ L² |

### `RHCrystal/NymanBeurlingMain.lean` (July 2026 — Milestones 3–4)
| Theorem | Meaning |
|---------|---------|
| `Dt_volume_le` | volume of the t-space disagreement set ≤ (H_J+1)/(J(J+1)) |
| `witness_integral_le` | ∫ (f_J − f_{J+1})² ≤ (H_J+1)/(4J(J+1)) |
| `nb_gram_eigenvalue_le` | **THEOREM 1 END TO END**: some eigenvalue of the N×N Nyman–Beurling Gram matrix is ≤ (H_J+1)/(10·J·(J+1)) |

The last entry is the complete machine verification of the project's
Theorem 1 (`nyman-beurling/chains_theory.md` §0): from the L²
definition of the dilation family, through the square-wave identity and
the measure-theoretic core (in t-coordinates the u⁻²du weight is plain
Lebesgue measure — no change of variables needed), to the spectral
conclusion via the Rayleigh principle.

## Build and verify

```bash
# Install elan (Lean version manager) if you don't have it
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

cd lean
lake update
lake exe cache get    # prebuilt Mathlib (~3GB, one-time)
lake build            # completes cleanly: zero sorries
```

The `#print axioms` commands at the bottom of each module print the
audit; you should see ONLY propext, Classical.choice, Quot.sound. If
anything else appears (in particular `sorryAx`), something is wrong.
