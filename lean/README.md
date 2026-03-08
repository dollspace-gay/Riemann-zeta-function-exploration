# RH Crystal: Lean 4 Formalization

Formal verification of the algebraic core of the Xi-weighted Gram matrix framework.

## What's Proven

| Theorem | Status | Meaning |
|---------|--------|---------|
| `xi_weight_symm` | **PROVEN** | w(p,σ) = w(p,1-σ) |
| `xi_weight_half` | **PROVEN** | w(p,1/2) = log(p)/√p |
| `gram_entry_symm` | **PROVEN** | G_{ij}(σ) = G_{ij}(1-σ) |
| `gram_matrix_symm` | **PROVEN** | G(σ) = G(1-σ) as matrices, hence κ(σ) = κ(1-σ) |
| `xi_weight_min` | `sorry` | w(p,1/2) ≤ w(p,σ) — mechanically provable but tedious |
| Condition number minimized at σ=1/2 | NOT ATTEMPTED | Requires spectral theory |
| Condition number bounded | NOT ATTEMPTED | This is essentially RH |

## Axiom Audit

The `#print axioms` commands at the bottom of `Basic.lean` will show exactly which
foundational axioms each theorem depends on. You should see ONLY:

- `propext` (propositional extensionality)
- `Quot.sound` (quotient soundness)
- `Classical.choice` (classical logic)

These are standard Lean 4 foundations. If you see ANYTHING else (especially anything
with "riemann" or "hypothesis" in the name), something is wrong.

The one `sorry` in `xi_weight_min` will show up as the `sorryAx` axiom if you
print its axioms. This is expected and flagged explicitly.

## Build and Verify

```bash
# Install elan (Lean version manager) if you don't have it
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build (first build downloads Mathlib — may take a while)
cd rh-lean
lake update
lake exe cache get    # Download prebuilt Mathlib (MUCH faster)
lake build
```

If it compiles with no errors (only the one expected sorry warning),
the proofs are machine-verified.

## What This Means

If `lake build` succeeds, Lean has machine-checked that:

1. The Xi weight function is symmetric under σ to 1-σ
2. The Gram matrix satisfies G(σ) = G(1-σ)
3. Therefore κ(σ) = κ(1-σ)

No human trust required for these steps. The hard parts
(minimization, boundedness) remain unproven and are explicitly
marked as such.
