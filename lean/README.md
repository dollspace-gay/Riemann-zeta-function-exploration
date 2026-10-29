# RH Crystal / Nyman–Beurling: Lean 4 Formalization

Machine-verified results across both phases of the project. **Zero
sorries in every module; the full project audit permits only
propext, Classical.choice, and Quot.sound.**

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
elan toolchain install leanprover/lean4:v4.33.0-rc1
lake exe cache get    # prebuilt Mathlib (~3GB, one-time)
lake build
lake env lean check_axioms.lean
```

Use the checked-in `lean-toolchain` and `lake-manifest.json`; `lake update`
changes the pinned dependencies and is not part of reproduction. On a machine
under memory pressure, build the new modules sequentially with
`lake build +JumpLocal`, `lake build +JumpArithmetic`,
`lake build +JumpSynthesis`, and `lake build +Refinement` before the final
`lake build`.

## September 2026: jump estimates and residual convergence

| Module | Machine-checked result |
|---|---|
| `JumpLocal` | Exact integral of the affine jump, lower bound s²/16, complex square completion |
| `JumpArithmetic` | Finite Schur estimate; weighted Möbius inversion; coefficient recovery loses at most N²H_N in squared norm |
| `JumpSynthesis` | **Real jump bound end to end:** actual synthesis integral ≥ Σc²/(36N²H_N), including fractional-part jumps, integrability, substitution, and weighted cell sum |
| `Refinement` | Scalar correction identity; removing an orthogonal old component improves gain; trial-correlation transfer bound and excess-energy identity; exponential cumulative energy bound; divergent guaranteed fractional gains force error → 0 |
| `Annihilator` | Fractional-part residue identity for every positive integer modulus; identity for the actual dilations; finite summation by parts with its boundary term; invariant-subspace projection correlation; nonzero residual from an annihilator witness |

The convergence theorem takes the gain inequality and divergence as explicit
hypotheses. It does not assert them for Nyman–Beurling residuals or prove RH.
The full written jump-bound argument and its formalization boundary are in
[`refinement_theory.md`](../nyman-beurling/refinement_theory.md). The new audit
checks every declaration in all listed project modules and fails on any axiom
other than `propext`, `Classical.choice`, or `Quot.sound`.

The September 23 residual-capture run builds all eight modules and audits 196
declarations successfully. The new projection/transfer lemmas are general
Hilbert-space facts with explicit hypotheses; they do not prove the arithmetic
correlation bound. Their application and the numerical boundary are recorded in
[`residual_capture_theory.md`](../nyman-beurling/residual_capture_theory.md).

The September 24 follow-up adds `gain_le_visible_energy` and
`cost_lower_of_half_capture`. They prove that a component orthogonal to all
corrections cannot be removed, and quantify the correction cost needed for
half-capture when the visible component is small. These are general
Hilbert-space facts. The associated fixed-N Möbius correlation limit is a
written argument using an external L1 theorem, not a Lean arithmetic theorem;
see [`mobius_correlation_identity.md`](../nyman-beurling/mobius_correlation_identity.md).
The full build passes, and the updated audit checks 198 declarations across
eight modules with only the three permitted standard axioms.

The subsequent September 24 annihilator investigation adds a ninth module,
`Annihilator`, to the default build and audit. Its three finite identities do
not formalize infinite Möbius inversion, the conditional summability theorem,
or the vanishing of the surviving remainder. See
[`annihilator_rigidity.md`](../nyman-beurling/annihilator_rigidity.md) for those
written arguments and their unproved regularity hypothesis.
That stage's full build succeeded, and its audit passed for 201 declarations
across nine modules with only the three permitted standard axioms.

The [decay follow-up](../nyman-beurling/autocorrelation_decay.md) adds
`projection_residual_correlation` and `projection_residual_ne_zero_of_witness`
to the same module. These use Mathlib's actual orthogonal projection and
prove two general Hilbert-space implications. They do not formalize the
analytic controls or establish decay. The current full build succeeds
(3505 jobs), and the audit passes for **203 declarations across nine modules**
with only the three permitted standard axioms.

The `#print axioms` commands at the bottom of each module print the
audit; you should see ONLY propext, Classical.choice, Quot.sound. If
anything else appears (in particular `sorryAx`), something is wrong.
