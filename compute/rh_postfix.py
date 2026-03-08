"""
RH Crystal: Post-Bug-Fix Analysis
===================================
1. Confirm σ=1/2 minimum survives with correct code
2. Eigenvector analysis: WHICH zeros become invisible?
3. Height dependence: does the crystal structure change?
4. Growth rate: fit the exact exponent of κ(N)

Uses Odlyzko's 2M zeros + GPU acceleration.
"""

import torch
import numpy as np
import time
import os
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
DATA_DIR = os.path.expanduser("~/rh_data")
OUTPUT_DIR = os.path.expanduser("~/rh_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =====================================================================
# LOAD DATA
# =====================================================================
print("\nLoading zeros...")
zeros_path = os.path.join(DATA_DIR, "zeros6")
if not os.path.exists(zeros_path):
    zeros_path = os.path.join(DATA_DIR, "zeta_zeros.csv")

zeros_list = []
with open(zeros_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            val = float(parts[-1])
            if val > 0:
                zeros_list.append(val)
        except ValueError:
            continue

zeros_np = np.array(zeros_list, dtype=np.float64)
print(f"Loaded {len(zeros_np)} zeros, range [{zeros_np[0]:.2f}, {zeros_np[-1]:.2f}]")

print("Generating primes...")
def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

all_primes = sieve_primes(10_000_000)
log_primes_np = np.log(all_primes.astype(np.float64))
print(f"{len(all_primes)} primes up to {all_primes[-1]}")

# =====================================================================
# GPU GRAM MATRIX (correct implementation)
# =====================================================================
def build_gram_gpu(n_zeros, zeros, primes, log_primes, sigma=0.5,
                   n_primes=None, normalize=True, batch_size=2000):
    N = min(n_zeros, len(zeros))
    P = min(n_primes or len(primes), len(primes))
    gamma = torch.tensor(zeros[:N], dtype=DTYPE, device=DEVICE)
    G = torch.zeros((N, N), dtype=DTYPE, device=DEVICE)

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch_p = torch.tensor(primes[start:end], dtype=DTYPE, device=DEVICE)
        batch_logp = torch.tensor(log_primes[start:end], dtype=DTYPE, device=DEVICE)
        w = batch_logp * torch.cosh((sigma - 0.5) * batch_logp) / torch.sqrt(batch_p)
        phase_args = -gamma.unsqueeze(1) * batch_logp.unsqueeze(0)
        cos_phases = torch.cos(phase_args)
        sin_phases = torch.sin(phase_args)
        sqrt_w = torch.sqrt(w).unsqueeze(0)
        wcos = cos_phases * sqrt_w
        wsin = sin_phases * sqrt_w
        G += wcos @ wcos.T + wsin @ wsin.T

    if normalize:
        tr = torch.trace(G)
        if tr > 1e-15:
            G *= N / tr
    return G

def gram_eigensystem(G):
    """Return eigenvalues AND eigenvectors."""
    eigenvalues, eigenvectors = torch.linalg.eigh(G)
    return eigenvalues, eigenvectors

# =====================================================================
# TEST 1: CONFIRM σ=1/2 MINIMUM (with correct code)
# =====================================================================
print("\n" + "=" * 66)
print("TEST 1: σ-DEPENDENCE (correct code)")
print("=" * 66)

for N_test in [100, 500, 1000, 2000]:
    if N_test > len(zeros_np):
        break

    n_primes = min(20 * N_test, len(all_primes))
    sigmas = np.linspace(0.2, 0.8, 25)

    print(f"\n  N={N_test}, P={n_primes}:")

    min_cond = float('inf')
    min_sigma = 0.5
    results = []

    for sigma in sigmas:
        G = build_gram_gpu(N_test, zeros_np, all_primes, log_primes_np,
                          sigma=sigma, n_primes=n_primes)
        eigvals = torch.linalg.eigvalsh(G)
        lmin = eigvals.min().item()
        lmax = eigvals.max().item()
        cond = lmax / lmin if lmin > 1e-15 else float('inf')
        results.append((sigma, cond, lmin, lmax))
        if cond < min_cond:
            min_cond = cond
            min_sigma = sigma
        del G
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print(f"  Min condition: {min_cond:.4f} at σ={min_sigma:.4f}")

    # Print key values
    for sigma, cond, lmin, lmax in results:
        if abs(sigma - 0.5) < 0.015 or abs(round(sigma, 1) - sigma) < 0.015:
            print(f"    σ={sigma:.4f}: κ={cond:.4f}  λ_min={lmin:.6f}  λ_max={lmax:.6f}")

    # Symmetry check
    print("  Symmetry:")
    for i in range(min(4, len(results) // 2)):
        s1, c1, _, _ = results[i]
        s2, c2, _, _ = results[-(i + 1)]
        if abs(s1 + s2 - 1.0) < 0.03:
            print(f"    σ={s1:.3f}: κ={c1:.4f}  |  σ={s2:.3f}: κ={c2:.4f}  |  match={abs(c1-c2)/max(c1,c2)*100:.4f}%")

# =====================================================================
# TEST 2: PRECISE SCALING LAW
# =====================================================================
print("\n" + "=" * 66)
print("TEST 2: SCALING LAW (correct code)")
print("=" * 66)

sizes = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
conds = []
lmins = []
actual_sizes = []

for N in sizes:
    if N > len(zeros_np):
        break
    n_primes = min(20 * N, len(all_primes))
    t0 = time.time()
    try:
        G = build_gram_gpu(N, zeros_np, all_primes, log_primes_np,
                          sigma=0.5, n_primes=n_primes)
        eigvals = torch.linalg.eigvalsh(G)
        lmin = eigvals.min().item()
        lmax = eigvals.max().item()
        cond = lmax / lmin if lmin > 1e-15 else float('inf')
        elapsed = time.time() - t0
        print(f"  N={N:5d}, P={n_primes:6d}: κ={cond:10.4f}  λ_min={lmin:.8f}  ({elapsed:.1f}s)")
        conds.append(cond)
        lmins.append(lmin)
        actual_sizes.append(N)
        del G
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  N={N}: OOM")
            torch.cuda.empty_cache()
            break
        raise

# Fit power law
if len(actual_sizes) > 3:
    log_N = np.log(actual_sizes)
    log_kappa = np.log(conds)
    log_lmin = np.log(np.abs(lmins))

    # kappa ~ a * N^alpha
    coeffs = np.polyfit(log_N, log_kappa, 1)
    alpha = coeffs[0]
    print(f"\n  Power law fit: κ ~ N^{alpha:.4f}")

    # lmin ~ b * N^beta
    coeffs2 = np.polyfit(log_N, log_lmin, 1)
    beta = coeffs2[0]
    print(f"  λ_min ~ N^{beta:.4f}")
    print(f"  (α=0.5 means √N growth, α=0 means bounded)")

# =====================================================================
# TEST 3: EIGENVECTOR ANALYSIS — WHICH ZEROS ARE INVISIBLE?
# =====================================================================
print("\n" + "=" * 66)
print("TEST 3: EIGENVECTOR ANALYSIS")
print("=" * 66)
print("Which zeros contribute to the smallest eigenvalue?")
print("If the 'invisible' zeros have structure, that's informative.\n")

for N_eig in [500, 1000, 2000]:
    if N_eig > len(zeros_np):
        break

    n_primes = min(20 * N_eig, len(all_primes))
    print(f"  --- N={N_eig}, P={n_primes} ---")

    G = build_gram_gpu(N_eig, zeros_np, all_primes, log_primes_np,
                      sigma=0.5, n_primes=n_primes)
    eigenvalues, eigenvectors = gram_eigensystem(G)

    eigvals_cpu = eigenvalues.cpu().numpy()
    eigvecs_cpu = eigenvectors.cpu().numpy()

    # The eigenvector for the SMALLEST eigenvalue
    min_vec = eigvecs_cpu[:, 0]  # eigvalsh returns sorted ascending
    max_vec = eigvecs_cpu[:, -1]

    # Where is the weight concentrated?
    min_vec_sq = min_vec ** 2  # probability distribution over zeros
    max_vec_sq = max_vec ** 2

    # Statistics of the "invisible direction"
    # Which zero indices have the largest weight?
    top_indices_min = np.argsort(min_vec_sq)[-10:][::-1]
    top_indices_max = np.argsort(max_vec_sq)[-10:][::-1]

    print(f"  λ_min = {eigvals_cpu[0]:.6f}, λ_max = {eigvals_cpu[-1]:.6f}")
    print(f"  κ = {eigvals_cpu[-1]/eigvals_cpu[0]:.4f}")

    print(f"\n  Top 10 zeros in λ_min eigenvector (the 'invisible' direction):")
    print(f"  {'Index':>8s} {'γ value':>14s} {'Weight':>10s} {'Local spacing':>14s}")
    gammas = zeros_np[:N_eig]
    spacings = np.diff(gammas)
    for idx in top_indices_min:
        spacing = spacings[idx] if idx < len(spacings) else spacings[-1]
        print(f"  {idx:8d} {gammas[idx]:14.4f} {min_vec_sq[idx]:10.6f} {spacing:14.4f}")

    # Is the invisible direction concentrated in specific regions?
    # Divide zeros into 10 blocks and see where the weight lives
    block_size = N_eig // 10
    print(f"\n  Weight distribution across zero blocks:")
    print(f"  {'Block':>8s} {'Zeros':>16s} {'Height range':>24s} {'λ_min weight':>14s} {'λ_max weight':>14s}")
    for b in range(10):
        start = b * block_size
        end = (b + 1) * block_size
        w_min = np.sum(min_vec_sq[start:end])
        w_max = np.sum(max_vec_sq[start:end])
        h_lo = gammas[start]
        h_hi = gammas[end - 1] if end - 1 < len(gammas) else gammas[-1]
        print(f"  {b:8d} {start:6d}-{end:6d}   {h_lo:10.1f} - {h_hi:10.1f} {w_min:14.6f} {w_max:14.6f}")

    # Participation ratio: how many zeros contribute to the eigenvector?
    # PR = 1 / Σ(v_i^4), where v_i are eigenvector components
    # PR = N means fully delocalized, PR = 1 means localized on one zero
    pr_min = 1.0 / np.sum(min_vec_sq ** 2)
    pr_max = 1.0 / np.sum(max_vec_sq ** 2)
    print(f"\n  Participation ratio (1=localized, N=delocalized):")
    print(f"    λ_min eigenvector: PR = {pr_min:.1f} / {N_eig} = {pr_min/N_eig:.4f}")
    print(f"    λ_max eigenvector: PR = {pr_max:.1f} / {N_eig} = {pr_max/N_eig:.4f}")

    if pr_min / N_eig < 0.1:
        print(f"    >>> λ_min direction is LOCALIZED — specific zeros are invisible <<<")
    elif pr_min / N_eig > 0.5:
        print(f"    >>> λ_min direction is DELOCALIZED — invisibility is spread out <<<")
    else:
        print(f"    >>> λ_min direction is PARTIALLY localized <<<")

    # Check if invisible zeros have unusual spacings
    # Weight the spacings by the eigenvector
    weighted_spacing = np.sum(min_vec_sq[:-1] * spacings) / np.sum(min_vec_sq[:-1])
    mean_spacing = np.mean(spacings)
    print(f"\n  Spacing analysis:")
    print(f"    Mean spacing (all zeros): {mean_spacing:.4f}")
    print(f"    Weighted spacing (λ_min direction): {weighted_spacing:.4f}")
    print(f"    Ratio: {weighted_spacing/mean_spacing:.4f}")
    if weighted_spacing / mean_spacing > 1.2:
        print(f"    >>> Invisible zeros tend to be in WIDE gaps <<<")
    elif weighted_spacing / mean_spacing < 0.8:
        print(f"    >>> Invisible zeros tend to be in NARROW gaps <<<")
    else:
        print(f"    >>> No strong spacing preference <<<")

    del G, eigenvalues, eigenvectors
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# =====================================================================
# TEST 4: HEIGHT DEPENDENCE
# =====================================================================
print("\n" + "=" * 66)
print("TEST 4: CONDITIONING AT DIFFERENT HEIGHTS")
print("=" * 66)
print("Same N=1000, but sampled from different parts of the zero sequence.")
print("If κ is constant, the crystal structure is stationary.\n")

block_size = 1000
n_primes = 20000

starts = [0, 5000, 10000, 50000, 100000, 500000, 1000000]

for start in starts:
    if start + block_size > len(zeros_np):
        break
    block = zeros_np[start:start + block_size]
    try:
        G = build_gram_gpu(block_size, block, all_primes, log_primes_np,
                          sigma=0.5, n_primes=n_primes)
        eigvals = torch.linalg.eigvalsh(G)
        lmin = eigvals.min().item()
        lmax = eigvals.max().item()
        cond = lmax / lmin if lmin > 1e-15 else float('inf')
        print(f"  Zeros {start:>8d}-{start+block_size:>8d}  "
              f"height {block[0]:>12.1f}-{block[-1]:>12.1f}  "
              f"κ={cond:8.4f}  λ_min={lmin:.6f}")
        del G
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError:
        torch.cuda.empty_cache()
        break

# =====================================================================
# TEST 5: DOES σ=1/2 MINIMUM HOLD AT DIFFERENT HEIGHTS?
# =====================================================================
print("\n" + "=" * 66)
print("TEST 5: σ-DEPENDENCE AT DIFFERENT HEIGHTS")
print("=" * 66)

for start in [0, 100000, 1000000]:
    if start + 500 > len(zeros_np):
        break
    block = zeros_np[start:start + 500]
    n_primes = 10000

    print(f"\n  Zeros {start}-{start+500} (height ~{block[0]:.0f}-{block[-1]:.0f}):")

    sigmas = np.linspace(0.3, 0.7, 15)
    min_cond = float('inf')
    min_sigma = 0.5

    for sigma in sigmas:
        try:
            G = build_gram_gpu(500, block, all_primes, log_primes_np,
                              sigma=sigma, n_primes=n_primes)
            eigvals = torch.linalg.eigvalsh(G)
            lmin = eigvals.min().item()
            lmax = eigvals.max().item()
            cond = lmax / lmin if lmin > 1e-15 else float('inf')
            if cond < min_cond:
                min_cond = cond
                min_sigma = sigma
            if abs(sigma - 0.5) < 0.02 or abs(sigma - 0.3) < 0.02 or abs(sigma - 0.7) < 0.02:
                print(f"    σ={sigma:.3f}: κ={cond:.4f}")
            del G
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError:
            torch.cuda.empty_cache()
            break

    print(f"  Min: κ={min_cond:.4f} at σ={min_sigma:.4f}")
    if abs(min_sigma - 0.5) < 0.03:
        print(f"  >>> σ=1/2 IS the minimum <<<")
    else:
        print(f"  >>> Minimum displaced to σ={min_sigma:.3f} <<<")

# =====================================================================
print("\n" + "=" * 66)
print("DONE")
print("=" * 66)
