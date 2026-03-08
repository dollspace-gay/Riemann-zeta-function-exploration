# Riemann Hypothesis Crystal Analysis (Rust)

## Setup

### Step 1: Precompute zeros (Python, one-time)

```bash
pip install mpmath numpy sympy
python precompute_zeros.py
```

This saves zeros to `~/rh_data/`. Takes ~20-30 min for 5000 Riemann zeros
plus Dirichlet L-function zeros. You only need to do this once.

Increase `N_ZEROS` in the script if you want more (10000 zeros ≈ 1-2 hours).

### Step 2: Build and run the Rust analysis

```bash
cd rh-crystal
cargo build --release
cargo run --release
```

The release build with LTO and native CPU targeting will be dramatically
faster than Python. The Gram matrix construction is parallelized across
all CPU cores using rayon.

## What it does

- **Sieve primes** up to 1,000,000 (78,498 primes) in milliseconds
- **Test A**: σ-dependence of condition number for Riemann ζ and 7 Dirichlet L-functions
- **Test B**: Scaling law κ(N) at prime ratios 20, 50, 100
- **Test C**: Prime ratio sweep at fixed N
- **Test D**: Large-scale σ-dependence (N=50, 100, 200 with 10,000 primes)

## Key question

Does the Xi-weighted normalized Gram matrix consistently minimize its
condition number at σ=1/2 across:
- All L-functions (universality)
- All matrix sizes (scaling)
- All prime-to-zero ratios

## Performance

| Operation | Python | Rust (release) |
|-----------|--------|----------------|
| Gram N=100, P=1000 | ~2s | ~0.01s |
| Gram N=500, P=10000 | ~minutes | ~1-5s |
| σ sweep (60 points) | ~2 min | ~1-5s |
| Full analysis | ~30 min | ~1-5 min |

The bottleneck shifts entirely to zero precomputation (Python/mpmath),
which only needs to be done once.
