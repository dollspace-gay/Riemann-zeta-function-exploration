"""
RH Crystal: GPU Edition (PyTorch)
==================================
Uses Odlyzko's 2,001,052 precomputed zeros + GPU acceleration.
Targets N=10,000 to N=20,000+ Gram matrices.

Requirements:
  pip install torch numpy sympy
  Download Odlyzko zeros: 
    wget https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6.gz
    gunzip zeros6.gz
  Place zeros6 in ~/rh_data/

Hardware: Needs GPU with 16GB+ VRAM for N=10000+
"""

import torch
import numpy as np
import time
import os
import sys

# =====================================================================
# CONFIG
# =====================================================================
DATA_DIR = os.path.expanduser("~/rh_data")
OUTPUT_DIR = os.path.expanduser("~/rh_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # Need double precision for large matrices

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Dtype: {DTYPE}")

# =====================================================================
# LOAD ZEROS
# =====================================================================
print("\nLoading zeros...")

# Try Odlyzko format first (space-separated, may have index column)
zeros_path = os.path.join(DATA_DIR, "zeros6")
if not os.path.exists(zeros_path):
    zeros_path = os.path.join(DATA_DIR, "zeta_zeros.csv")

if not os.path.exists(zeros_path):
    print(f"ERROR: No zeros file found. Download Odlyzko's zeros:")
    print(f"  cd {DATA_DIR}")
    print(f"  wget https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6.gz")
    print(f"  gunzip zeros6.gz")
    sys.exit(1)

t0 = time.time()
# Odlyzko's format: one number per line (just the imaginary part)
zeros_list = []
with open(zeros_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Handle possible whitespace-separated columns
        parts = line.split()
        try:
            val = float(parts[-1])  # Take last column
            if val > 0:
                zeros_list.append(val)
        except ValueError:
            continue

zeros_np = np.array(zeros_list, dtype=np.float64)
print(f"Loaded {len(zeros_np)} zeros in {time.time()-t0:.1f}s")
print(f"Range: [{zeros_np[0]:.4f}, {zeros_np[-1]:.4f}]")
print(f"First 5 zeros: {zeros_np[:5]}")
print(f"Zero #100: {zeros_np[99]}")
print(f"Zero #1000: {zeros_np[999]}")
# =====================================================================
# GENERATE PRIMES (use numpy sieve for speed)
# =====================================================================
print("\nGenerating primes...")

def sieve_primes(limit):
    """Simple but fast numpy sieve."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

t0 = time.time()
MAX_PRIME = 10_000_000  # 10 million -> ~664k primes
all_primes = sieve_primes(MAX_PRIME)
print(f"Sieved {len(all_primes)} primes up to {MAX_PRIME} in {time.time()-t0:.1f}s")

log_primes_np = np.log(all_primes.astype(np.float64))

# =====================================================================
# GPU GRAM MATRIX CONSTRUCTION
# =====================================================================
def build_gram_gpu(n_zeros, zeros, primes, log_primes, sigma=0.5,
                   n_primes=None, normalize=True, batch_size=1000):
    """
    Build Xi-weighted Gram matrix on GPU using batched outer products.

    For each batch of primes, compute:
      phases = exp(-i * gamma * log_p)  [N x batch]
      weights = log_p * cosh((sigma-0.5)*log_p) / sqrt(p)  [batch]
      G += sum_batch( w * real(outer(phase, conj(phase))) )

    This vectorizes over primes within each batch.
    """
    N = min(n_zeros, len(zeros))
    P = min(n_primes or len(primes), len(primes))

    gamma = torch.tensor(zeros[:N], dtype=DTYPE, device=DEVICE)  # [N]

    G = torch.zeros((N, N), dtype=DTYPE, device=DEVICE)

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch_p = torch.tensor(primes[start:end], dtype=DTYPE, device=DEVICE)
        batch_logp = torch.tensor(log_primes[start:end], dtype=DTYPE, device=DEVICE)

        # Weights: log(p) * cosh((sigma-0.5)*logp) / sqrt(p)
        w = batch_logp * torch.cosh((sigma - 0.5) * batch_logp) / torch.sqrt(batch_p)  # [B]

        # Phases: exp(-i * gamma_j * logp_k)
        # gamma: [N], batch_logp: [B]  -> phase_args: [N, B]
        phase_args = -gamma.unsqueeze(1) * batch_logp.unsqueeze(0)  # [N, B]
        cos_phases = torch.cos(phase_args)  # [N, B]
        sin_phases = torch.sin(phase_args)  # [N, B]

        # For each prime k, contribution to G is:
        # w[k] * (cos_i*cos_j + sin_i*sin_j)  [this is Re(phase_i * conj(phase_j))]
        # = w[k] * cos_phases[:,k] @ cos_phases[:,k].T + w[k] * sin_phases[:,k] @ sin_phases[:,k].T
        #
        # Vectorized: G += (cos_phases * sqrt(w)) @ (cos_phases * sqrt(w)).T
        #                + (sin_phases * sqrt(w)) @ (sin_phases * sqrt(w)).T

        sqrt_w = torch.sqrt(w).unsqueeze(0)  # [1, B]
        wcos = cos_phases * sqrt_w  # [N, B]
        wsin = sin_phases * sqrt_w  # [N, B]

        G += wcos @ wcos.T + wsin @ wsin.T

    if normalize:
        tr = torch.trace(G)
        if tr > 1e-15:
            G *= N / tr

    return G

def gram_stats_gpu(G):
    """Compute condition number on GPU."""
    eigvals = torch.linalg.eigvalsh(G)
    lmin = eigvals.min().item()
    lmax = eigvals.max().item()
    cond = lmax / lmin if lmin > 1e-15 else float('inf')
    return cond, lmin, lmax

# =====================================================================
# ESTIMATE MEMORY
# =====================================================================
def estimate_memory(N, P, batch_size=1000):
    """Estimate GPU memory usage in GB."""
    matrix = N * N * 8 / 1e9  # G matrix (float64)
    phases = N * batch_size * 8 * 2 / 1e9  # cos + sin phases
    eigvec = N * N * 8 / 1e9  # eigenvector workspace
    total = matrix * 2 + phases + eigvec  # G + workspace + phases + eigvecs
    return total

print("\nMemory estimates:")
for N in [1000, 5000, 10000, 15000, 20000]:
    mem = estimate_memory(N, 200000)
    print(f"  N={N:6d}: ~{mem:.1f} GB VRAM")

# =====================================================================
# TEST 1: MASSIVE SCALING
# =====================================================================
print("\n" + "=" * 66)
print("TEST 1: SCALING TO N=10000+ (ratio=20, sigma=0.5)")
print("=" * 66)

sizes = [100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]

for N in sizes:
    if N > len(zeros_np):
        print(f"  N={N}: not enough zeros (have {len(zeros_np)})")
        break

    n_primes = min(20 * N, len(all_primes))
    mem_est = estimate_memory(N, n_primes)

    if DEVICE.type == "cuda":
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        if mem_est > free_mem * 0.9:
            print(f"  N={N}: would need ~{mem_est:.1f}GB, only {free_mem:.1f}GB free. Skipping.")
            continue

    t0 = time.time()
    try:
        G = build_gram_gpu(N, zeros_np, all_primes, log_primes_np,
                          sigma=0.5, n_primes=n_primes, batch_size=2000)
        cond, lmin, lmax = gram_stats_gpu(G)
        elapsed = time.time() - t0
        print(f"  N={N:6d}, P={n_primes:7d}: cond={cond:10.4f}  lmin={lmin:.8f}  lmax={lmax:.6f}  ({elapsed:.1f}s)")
        del G
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  N={N}: GPU OOM. Stopping scaling test.")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            break
        raise

# =====================================================================
# TEST 2: SCALING WITH RATIO=50
# =====================================================================
print("\n" + "=" * 66)
print("TEST 2: SCALING ratio=50")
print("=" * 66)

for N in sizes:
    if N > len(zeros_np):
        break
    n_primes = min(50 * N, len(all_primes))
    mem_est = estimate_memory(N, n_primes)
    if DEVICE.type == "cuda":
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        if mem_est > free_mem * 0.9:
            print(f"  N={N}: skipping (memory)")
            continue
    t0 = time.time()
    try:
        G = build_gram_gpu(N, zeros_np, all_primes, log_primes_np,
                          sigma=0.5, n_primes=n_primes, batch_size=2000)
        cond, lmin, lmax = gram_stats_gpu(G)
        elapsed = time.time() - t0
        print(f"  N={N:6d}, P={n_primes:7d}: cond={cond:10.4f}  lmin={lmin:.8f}  lmax={lmax:.6f}  ({elapsed:.1f}s)")
        del G
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  N={N}: GPU OOM.")
            torch.cuda.empty_cache()
            break
        raise

# =====================================================================
# TEST 3: SIGMA DEPENDENCE AT LARGEST FEASIBLE N
# =====================================================================
print("\n" + "=" * 66)
print("TEST 3: SIGMA DEPENDENCE AT LARGE N")
print("=" * 66)

# Test at N=5000 first (should be fast)
for N_test in [5000, 10000]:
    if N_test > len(zeros_np):
        break

    n_primes = min(20 * N_test, len(all_primes))
    mem_est = estimate_memory(N_test, n_primes)

    print(f"\n  --- N={N_test}, P={n_primes} ---")

    if DEVICE.type == "cuda":
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        if mem_est > free_mem * 0.8:
            print(f"  Skipping (need ~{mem_est:.1f}GB, have {free_mem:.1f}GB)")
            continue

    sigmas = np.linspace(0.2, 0.8, 25)
    min_cond = float('inf')
    min_sigma = 0.5
    results = []

    for sigma in sigmas:
        try:
            G = build_gram_gpu(N_test, zeros_np, all_primes, log_primes_np,
                              sigma=sigma, n_primes=n_primes, batch_size=2000)
            cond, lmin, lmax = gram_stats_gpu(G)
            results.append((sigma, cond, lmin, lmax))
            if cond < min_cond:
                min_cond = cond
                min_sigma = sigma
            del G
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError:
            torch.cuda.empty_cache()
            break

    print(f"  Min condition: {min_cond:.6f} at sigma={min_sigma:.4f}")
    for sigma, cond, lmin, lmax in results:
        if abs(sigma - 0.5) < 0.03 or abs(round(sigma, 1) - sigma) < 0.015:
            print(f"    s={sigma:.4f}: cond={cond:.6f}  lmin={lmin:.8f}  lmax={lmax:.8f}")

    # Check symmetry
    print("  Symmetry check:")
    for i in range(min(5, len(results) // 2)):
        s1, c1, _, _ = results[i]
        s2, c2, _, _ = results[-(i+1)]
        if abs(s1 + s2 - 1.0) < 0.03:
            sym_err = abs(c1 - c2) / max(c1, c2) * 100
            print(f"    sigma={s1:.3f} vs {s2:.3f}: cond={c1:.4f} vs {c2:.4f} (diff={sym_err:.4f}%)")

# =====================================================================
# TEST 4: PRIME RATIO SWEEP AT LARGE N
# =====================================================================
print("\n" + "=" * 66)
print("TEST 4: PRIME RATIO SWEEP AT N=5000")
print("=" * 66)

N_sweep = min(5000, len(zeros_np))
ratios = [5, 10, 20, 30, 50, 75, 100, 130]

for r in ratios:
    n_primes = min(r * N_sweep, len(all_primes))
    try:
        G = build_gram_gpu(N_sweep, zeros_np, all_primes, log_primes_np,
                          sigma=0.5, n_primes=n_primes, batch_size=2000)
        cond, lmin, lmax = gram_stats_gpu(G)
        print(f"  ratio={r:4d} ({n_primes:7d} primes): cond={cond:10.4f}  lmin={lmin:.8f}")
        del G
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError:
        print(f"  ratio={r}: OOM")
        torch.cuda.empty_cache()
        break

# =====================================================================
# TEST 5: ZEROS AT EXTREME HEIGHTS (if available)
# =====================================================================
if len(zeros_np) > 100000:
    print("\n" + "=" * 66)
    print("TEST 5: CONDITIONING AT DIFFERENT HEIGHTS")
    print("=" * 66)
    print("Testing whether the conditioning changes for zeros at different heights")

    # Test blocks of 1000 zeros at different positions
    block_size = 1000
    n_primes = 20000

    for start in [0, 10000, 50000, 100000, 500000, 1000000]:
        if start + block_size > len(zeros_np):
            break
        block_zeros = zeros_np[start:start + block_size]
        G = build_gram_gpu(block_size, block_zeros, all_primes, log_primes_np,
                          sigma=0.5, n_primes=n_primes, batch_size=2000)
        cond, lmin, lmax = gram_stats_gpu(G)
        print(f"  Zeros {start:>8d}-{start+block_size}: height~{block_zeros[0]:.0f}-{block_zeros[-1]:.0f}  cond={cond:.4f}  lmin={lmin:.6f}")
        del G
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

# =====================================================================
print("\n" + "=" * 66)
print(f"DONE")
print("=" * 66)
