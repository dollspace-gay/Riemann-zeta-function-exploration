# Bug: Diagonal Doubling in Gram Matrix Construction

## What Happened

The Rust inner loop for building the Gram matrix contained this code:

```rust
for i in 0..n {
    for j in i..n {
        let val = w * (phases[i] * phases[j].conj()).re;
        g[(i, j)] += val;
        g[(j, i)] += val;
    }
}
```

When `i == j`, both assignments write to `g[(i, i)]`, doubling the diagonal entries.

## Effect

Adding an extra copy of the diagonal is equivalent to adding a scaled identity matrix to G. This compresses the condition number because it shifts all eigenvalues up by a constant, making the ratio λ_max/λ_min closer to 1.

The effect was approximately 10×: the Rust code reported κ ≈ 3.4 at N=2000 while the correct value is κ ≈ 33.8.

## How It Was Found

When the computation was reimplemented in PyTorch for GPU acceleration using vectorized outer products (`wcos @ wcos.T + wsin @ wsin.T`), the results disagreed by an order of magnitude. The zeros were verified to be identical, isolating the bug to the matrix construction.

## What It Invalidated

The claim that the condition number is "bounded" (κ < 4 at N=2000) was entirely an artifact of this bug. The actual condition number grows as approximately N^{0.61}.

## What It Did NOT Invalidate

- The σ = 1/2 isotropy minimum (the diagonal doubling is the same at every σ)
- The symmetry κ(σ) = κ(1−σ) (structural, proven in Lean)
- The universality across L-functions (same reasoning)
- The Lean proofs (purely algebraic, no numerical code involved)

## Fix

Either add a guard:

```rust
for i in 0..n {
    for j in i..n {
        let val = w * (phases[i] * phases[j].conj()).re;
        g[(i, j)] += val;
        if i != j {
            g[(j, i)] += val;
        }
    }
}
```

Or use the vectorized formulation as in the GPU code.

## Lesson

Scale before celebrating. The bug was invisible at N=500 because the inflated results looked plausible. It only became apparent when a completely independent implementation at 10× larger scale produced different numbers.
