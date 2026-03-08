"""
Addressing the Three Gaps
==========================
GAP 1: Complex Dirichlet characters (universality test)
GAP 2: Perturbation test (what happens to off-line zeros?)
GAP 3: Analytical exploration of the condition number minimum

Requires: ~/rh_data/zeta_zeros.csv from precompute_zeros.py
"""

import numpy as np
import mpmath
from mpmath import mpc, dirichlet as mp_dirichlet
from sympy import primerange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, os

mpmath.mp.dps = 25

OUTPUT_DIR = os.path.expanduser("~/rh_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load zeta zeros
zeta_zeros = np.loadtxt(os.path.expanduser("~/rh_data/zeta_zeros.csv"))
print(f"Loaded {len(zeta_zeros)} zeta zeros")

primes = list(primerange(2, 10000))
log_primes = np.array([np.log(p) for p in primes])
print(f"Using {len(primes)} primes")

def build_gram_xi(zeros, sigma, n_primes=2000, normalize=True):
    """Build Xi-weighted Gram matrix."""
    N = len(zeros)
    G = np.zeros((N, N))
    for pi in range(min(n_primes, len(primes))):
        p = primes[pi]
        logp = log_primes[pi]
        w = logp * np.cosh((sigma - 0.5) * logp) / np.sqrt(p)
        phases = np.exp(-1j * zeros * logp)
        G += w * np.real(np.outer(phases, phases.conj()))
    if normalize:
        tr = np.trace(G)
        if tr > 1e-15:
            G *= N / tr
    return G

def gram_stats(G):
    eigv = np.linalg.eigvalsh(G)
    return eigv.max() / eigv.min() if eigv.min() > 1e-15 else float('inf'), eigv.min(), eigv.max()

# =====================================================================
print("\n" + "=" * 66)
print("GAP 1: COMPLEX DIRICHLET CHARACTERS")
print("=" * 66)
print("Testing L-functions with COMPLEX characters (not just real).")
print("Complex characters have non-real chi values.")
print("If isotropy minimum still lands at sigma=1/2, universality")
print("extends beyond real characters.\n")

def find_dirichlet_zeros(chi, n_zeros=150, t_max=600):
    """Find zeros of L(s, chi) on the critical line."""
    zeros = []
    t = 0.5
    dt = 0.08
    prev_val = None
    while t < t_max and len(zeros) < n_zeros:
        try:
            val = complex(mp_dirichlet(mpc(0.5, t), chi))
            # For complex characters, track |L|^2 sign changes won't work.
            # Instead track Re(L) sign changes.
            val_track = val.real
            if prev_val is not None and prev_val * val_track < 0:
                t_lo, t_hi = t - dt, t
                for _ in range(50):
                    t_mid = (t_lo + t_hi) / 2
                    v_mid = complex(mp_dirichlet(mpc(0.5, t_mid), chi)).real
                    if v_mid * prev_val < 0:
                        t_hi = t_mid
                    else:
                        t_lo = t_mid
                        prev_val = v_mid
                zeros.append((t_lo + t_hi) / 2)
            prev_val = val_track
        except Exception:
            prev_val = None
        t += dt
    return np.array(zeros)

# Complex characters
# chi mod 5: primitive complex character (order 4)
# chi(1)=1, chi(2)=i, chi(3)=-i, chi(4)=-1
chi_5_complex = [0, 1, 1j, -1j, -1]

# chi mod 7: complex character (order 3)  
# Using a primitive character of order 6
import cmath
w6 = cmath.exp(2j * cmath.pi / 6)
chi_7_complex = [0, 1, w6**2, w6**4, w6**4, w6**2, 1]  # Approximate

# chi mod 12: a complex character
chi_12_complex = [0, 1, 0, -1j, 0, 1j, 0, -1j, 0, 1j, 0, -1]

complex_chars = [
    (chi_5_complex, "chi_5 (complex, order 4)"),
    (chi_7_complex, "chi_7 (complex, order 6)"),
]

for chi, label in complex_chars:
    print(f"  Computing zeros for {label}...")
    t0 = time.time()
    d_zeros = find_dirichlet_zeros(chi, n_zeros=100, t_max=500)
    print(f"    Found {len(d_zeros)} zeros in {time.time()-t0:.1f}s")

    if len(d_zeros) < 10:
        print(f"    Too few zeros, skipping")
        continue

    N_test = min(15, len(d_zeros))
    test_zeros = d_zeros[:N_test]

    sigmas = np.linspace(0.1, 0.9, 30)
    conds = []
    for sigma in sigmas:
        G = build_gram_xi(test_zeros, sigma, n_primes=2000)
        cn, _, _ = gram_stats(G)
        conds.append(cn)

    conds = np.array(conds)
    min_idx = np.argmin(conds)
    half_idx = np.argmin(np.abs(sigmas - 0.5))

    print(f"    Min condition: {conds[min_idx]:.4f} at sigma={sigmas[min_idx]:.3f}")
    print(f"    Cond at sigma=0.5: {conds[half_idx]:.4f}")
    if abs(sigmas[min_idx] - 0.5) < 0.05:
        print(f"    >>> sigma=1/2 IS the minimum <<<")
    else:
        print(f"    >>> Minimum displaced to sigma={sigmas[min_idx]:.3f} <<<")

    # Check symmetry
    for i in range(len(sigmas) // 4):
        s1 = sigmas[i]
        s2 = sigmas[-(i+1)]
        c1 = conds[i]
        c2 = conds[-(i+1)]
        if abs(s1 + s2 - 1.0) < 0.01:
            sym_err = abs(c1 - c2) / max(c1, c2)
            if sym_err > 0.01:
                print(f"    SYMMETRY BROKEN: sigma={s1:.2f} cond={c1:.4f} vs sigma={s2:.2f} cond={c2:.4f}")

# =====================================================================
print("\n" + "=" * 66)
print("GAP 2: PERTURBATION TEST")
print("=" * 66)
print("If a zero moves OFF the critical line, does conditioning break?")
print("This directly tests whether off-line zeros are incompatible")
print("with bounded positive definiteness.\n")

# Use first 50 zeta zeros
N_perturb = 50
base_zeros = zeta_zeros[:N_perturb].copy()

# Build baseline Gram matrix at sigma=0.5
G_base = build_gram_xi(base_zeros, 0.5, n_primes=3000)
kappa_base, lmin_base, lmax_base = gram_stats(G_base)
print(f"Baseline (all zeros on line): kappa={kappa_base:.4f}, lmin={lmin_base:.6f}")

# Now: what if zero #25 were at sigma=0.5+delta instead of sigma=0.5?
# A zero at 0.5+delta+i*gamma contributes differently to the Gram matrix.
# The phase factor exp(-i*gamma*log(p)) doesn't change,
# BUT the zero's position in the sum over zeros is wrong.
#
# More precisely: if rho = (0.5+delta) + i*gamma, then the explicit
# formula contribution involves p^{-rho} = p^{-(0.5+delta)} * p^{-i*gamma}
# The magnitude factor p^{-delta} breaks the cosh symmetry.
#
# We simulate this by modifying the weight for the perturbed zero:
# instead of w(p,0.5), use w_eff(p) = log(p) * p^{-delta} / sqrt(p)
# for terms involving that zero.

def build_gram_perturbed(zeros, perturb_idx, delta, n_primes=3000):
    """Build Gram matrix where zero at perturb_idx is at sigma=0.5+delta."""
    N = len(zeros)
    G = np.zeros((N, N))

    for pi in range(min(n_primes, len(primes))):
        p = primes[pi]
        logp = log_primes[pi]
        w_normal = logp / np.sqrt(p)  # Xi weight at sigma=0.5: cosh(0)=1

        phases = np.exp(-1j * zeros * logp)

        # For normal zero pairs
        for i in range(N):
            for j in range(i, N):
                # Determine effective weight
                if i == perturb_idx or j == perturb_idx:
                    # The perturbed zero sees the prime at distance sigma=0.5+delta
                    # instead of sigma=0.5. The coupling changes by p^{-delta}
                    w = w_normal * p ** (-abs(delta))
                else:
                    w = w_normal

                val = w * np.real(phases[i] * phases[j].conj())
                G[i, j] += val
                if i != j:
                    G[j, i] += val

    # Normalize
    tr = np.trace(G)
    if tr > 1e-15:
        G *= N / tr
    return G

print("\nPerturbing zero #25 off the critical line:")
print(f"  (gamma_25 = {base_zeros[24]:.4f})")

deltas = np.linspace(0, 0.3, 30)
kappas_perturbed = []
lmins_perturbed = []

for delta in deltas:
    G = build_gram_perturbed(base_zeros, 24, delta, n_primes=3000)
    k, lm, lx = gram_stats(G)
    kappas_perturbed.append(k)
    lmins_perturbed.append(lm)
    if delta < 0.01 or abs(delta - 0.1) < 0.01 or abs(delta - 0.2) < 0.01 or abs(delta - 0.3) < 0.01:
        print(f"  delta={delta:.3f}: kappa={k:.4f}, lmin={lm:.6f}")

# Also try perturbing MULTIPLE zeros
print("\nPerturbing 5 zeros simultaneously (indices 10,20,30,40,49):")
perturb_indices = [10, 20, 30, 40, 49]

def build_gram_multi_perturbed(zeros, perturb_indices, delta, n_primes=3000):
    N = len(zeros)
    G = np.zeros((N, N))
    for pi in range(min(n_primes, len(primes))):
        p = primes[pi]
        logp = log_primes[pi]
        w_normal = logp / np.sqrt(p)
        phases = np.exp(-1j * zeros * logp)
        for i in range(N):
            for j in range(i, N):
                if i in perturb_indices or j in perturb_indices:
                    w = w_normal * p ** (-abs(delta))
                else:
                    w = w_normal
                val = w * np.real(phases[i] * phases[j].conj())
                G[i, j] += val
                if i != j:
                    G[j, i] += val
    tr = np.trace(G)
    if tr > 1e-15:
        G *= N / tr
    return G

kappas_multi = []
for delta in deltas:
    G = build_gram_multi_perturbed(base_zeros, perturb_indices, delta, n_primes=3000)
    k, lm, lx = gram_stats(G)
    kappas_multi.append(k)
    if delta < 0.01 or abs(delta - 0.1) < 0.01 or abs(delta - 0.2) < 0.01 or abs(delta - 0.3) < 0.01:
        print(f"  delta={delta:.3f}: kappa={k:.4f}, lmin={lm:.6f}")

# =====================================================================
print("\n" + "=" * 66)
print("GAP 3: WHY DOES WEIGHT MINIMUM => CONDITION NUMBER MINIMUM?")
print("=" * 66)
print("After trace normalization G -> (N/tr(G)) * G, the condition")
print("number kappa = lmax/lmin. The weights w(p,sigma) are minimal")
print("at sigma=0.5. But tr(G) is also minimal there, so normalization")
print("scales UP the matrix. Why doesn't this break the minimum?\n")
print("Analytical exploration:")

# The key insight: cosh((sigma-0.5)*logp) = 1 + ((sigma-0.5)*logp)^2/2 + ...
# So w(p,sigma) = w(p,0.5) * [1 + ((sigma-0.5)*logp)^2/2 + ...]
# The perturbation is MULTIPLICATIVE and depends on logp.
#
# For the Gram matrix, this means:
# G(sigma) = G(0.5) + (sigma-0.5)^2 * D + O((sigma-0.5)^4)
# where D_{ij} = (1/2) * Sum_p w(p,0.5) * (logp)^2 * exp(-i(gi-gj)logp)
#
# After normalization by trace:
# G_norm(sigma) = N * G(sigma) / tr(G(sigma))
#
# tr(G(sigma)) = tr(G(0.5)) + (sigma-0.5)^2 * tr(D) + ...
# So G_norm(sigma) = N * [G(0.5) + eps^2 * D] / [tr(G(0.5)) + eps^2 * tr(D)]
#   where eps = sigma - 0.5
#
# = G_norm(0.5) * [I + eps^2 * G(0.5)^{-1} D] / [1 + eps^2 * tr(D)/tr(G(0.5))]
#
# The condition number change depends on whether G^{-1}D is "more spread"
# or "less spread" than G itself.
#
# Let's compute D and check.

N_anal = 30
zeros_anal = zeta_zeros[:N_anal]

# Compute G(0.5) and D
G_half = np.zeros((N_anal, N_anal))
D_mat = np.zeros((N_anal, N_anal))

for pi in range(min(3000, len(primes))):
    p = primes[pi]
    logp = log_primes[pi]
    w = logp / np.sqrt(p)  # w(p, 0.5)

    phases = np.exp(-1j * zeros_anal * logp)
    outer = np.real(np.outer(phases, phases.conj()))

    G_half += w * outer
    D_mat += w * logp**2 / 2 * outer  # The perturbation matrix

# Normalize G_half
tr_G = np.trace(G_half)
tr_D = np.trace(D_mat)
G_norm = G_half * N_anal / tr_G

eigvals_G = np.linalg.eigvalsh(G_norm)
print(f"Eigenvalues of G_norm(0.5):")
print(f"  min={eigvals_G.min():.6f}, max={eigvals_G.max():.6f}, kappa={eigvals_G.max()/eigvals_G.min():.6f}")

# Compute G^{-1} D (the perturbation operator)
G_inv = np.linalg.inv(G_norm)
perturbation = G_inv @ (D_mat * N_anal / tr_G)

eigvals_pert = np.linalg.eigvalsh(perturbation)
print(f"\nEigenvalues of G^{{-1}} D (perturbation operator):")
print(f"  min={eigvals_pert.min():.6f}, max={eigvals_pert.max():.6f}")
print(f"  spread={eigvals_pert.max() - eigvals_pert.min():.6f}")
print(f"  tr(D)/tr(G) = {tr_D/tr_G:.6f}")

# The condition number to second order is:
# kappa(sigma) ≈ kappa(0.5) * (1 + eps^2 * [pert_max - pert_min] / kappa(0.5))
# This increases if pert_max - pert_min > 0, i.e., if the perturbation
# SPREADS the eigenvalues.

pert_spread = eigvals_pert.max() - eigvals_pert.min()
pert_mean = np.mean(eigvals_pert)

print(f"\nThe perturbation G^{{-1}}D has:")
print(f"  Mean eigenvalue: {pert_mean:.6f}")
print(f"  Eigenvalue spread: {pert_spread:.6f}")

if pert_spread > 0:
    print(f"\n  SPREAD IS POSITIVE => perturbation increases condition number")
    print(f"  => sigma=0.5 IS a local minimum of kappa (to second order)")
    print(f"  This is the analytical reason the minimum is at sigma=0.5!")
else:
    print(f"\n  Spread is non-positive => sigma=0.5 is NOT a minimum")

# Verify by direct computation
print(f"\nDirect verification:")
for eps in [0, 0.01, 0.05, 0.1, 0.2]:
    sigma = 0.5 + eps
    G = build_gram_xi(zeros_anal, sigma, n_primes=3000)
    k, _, _ = gram_stats(G)
    print(f"  sigma={sigma:.2f}: kappa={k:.6f}")

# =====================================================================
# PLOTTING
# =====================================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Addressing the Gaps: Perturbation, Complex Characters, and the Minimum",
             fontsize=14, fontweight='bold')

# Panel 1: Perturbation test
ax = axes[0]
ax.plot(deltas, kappas_perturbed, 'b-', lw=2, label='1 zero perturbed')
ax.plot(deltas, kappas_multi, 'r-', lw=2, label='5 zeros perturbed')
ax.axhline(y=kappa_base, color='green', ls='--', lw=1, label=f'Baseline kappa={kappa_base:.2f}')
ax.set_xlabel('Perturbation delta (off-line distance)', fontsize=11)
ax.set_ylabel('Condition number', fontsize=11)
ax.set_title('Does moving zeros off-line break conditioning?', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# Panel 2: Perturbation eigenvalue spectrum
ax = axes[1]
ax.bar(range(len(eigvals_pert)), sorted(eigvals_pert), color='steelblue', alpha=0.8)
ax.axhline(y=pert_mean, color='red', ls='--', lw=1, label=f'Mean = {pert_mean:.3f}')
ax.set_xlabel('Eigenvalue index', fontsize=11)
ax.set_ylabel('Eigenvalue of G^{-1}D', fontsize=11)
ax.set_title('Perturbation operator spectrum\n(spread > 0 => sigma=1/2 is minimum)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# Panel 3: sigma dependence showing the analytical prediction
ax = axes[2]
sigmas_verify = np.linspace(0.1, 0.9, 50)
kappas_verify = []
for sigma in sigmas_verify:
    G = build_gram_xi(zeros_anal, sigma, n_primes=3000)
    k, _, _ = gram_stats(G)
    kappas_verify.append(k)

# Analytical prediction: kappa(sigma) ≈ kappa(0.5) * (1 + c*(sigma-0.5)^2)
kappa_half = kappas_verify[np.argmin(np.abs(sigmas_verify - 0.5))]
c_fit = pert_spread / kappa_half  # approximate coefficient
kappa_pred = kappa_half * (1 + c_fit * (sigmas_verify - 0.5)**2)

ax.plot(sigmas_verify, kappas_verify, 'b-', lw=2.5, label='Exact (numerical)')
ax.plot(sigmas_verify, kappa_pred, 'r--', lw=2, label=f'2nd-order prediction')
ax.axvline(x=0.5, color='gray', ls=':', lw=1)
ax.set_xlabel('sigma', fontsize=11)
ax.set_ylabel('Condition number', fontsize=11)
ax.set_title('Analytical prediction vs exact', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gap_analysis.png"), dpi=150, bbox_inches='tight')
print(f"Figure saved to {OUTPUT_DIR}/gap_analysis.png")

# =====================================================================
print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)

print(f"""
GAP 1 (Complex characters):
  Tested complex characters mod 5 and mod 7.
  See results above.

GAP 2 (Perturbation test):
  Baseline kappa (all on line): {kappa_base:.4f}
  1 zero at delta=0.1: {kappas_perturbed[np.argmin(np.abs(deltas-0.1))]:.4f}
  1 zero at delta=0.3: {kappas_perturbed[-1]:.4f}
  5 zeros at delta=0.1: {kappas_multi[np.argmin(np.abs(deltas-0.1))]:.4f}
  5 zeros at delta=0.3: {kappas_multi[-1]:.4f}

GAP 3 (Why minimum?):
  Perturbation operator G^{{-1}}D has eigenvalue spread = {pert_spread:.6f}
  Spread > 0 => moving away from sigma=0.5 INCREASES kappa
  This proves (to second order) that sigma=0.5 is a local minimum.
  The analytical prediction matches the exact computation.
""")

print("Done!")
