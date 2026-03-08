"""
Precompute Riemann zeta zeros and Dirichlet L-function zeros.
Saves to CSV files for the Rust program to load.
Run this ONCE, then use Rust for all analysis.
"""

import numpy as np
import mpmath
from mpmath import zetazero, mpc
import time, os

mpmath.mp.dps = 30

OUTPUT_DIR = os.path.expanduser("~/rh_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# Riemann zeta zeros — compute as many as patience allows
# =====================================================================
N_ZEROS = 5000  # ~20-30 min depending on your machine

print(f"Computing {N_ZEROS} Riemann zeta zeros...")
t0 = time.time()
zeros = []

for n in range(1, N_ZEROS + 1):
    z = zetazero(n)
    zeros.append(float(z.imag))
    if n % 500 == 0:
        elapsed = time.time() - t0
        rate = n / elapsed
        remaining = (N_ZEROS - n) / rate
        print(f"  {n}/{N_ZEROS} in {elapsed:.0f}s ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

np.savetxt(os.path.join(OUTPUT_DIR, "zeta_zeros.csv"), zeros, fmt='%.20f')
print(f"Saved {len(zeros)} zeta zeros to {OUTPUT_DIR}/zeta_zeros.csv")

# =====================================================================
# Dirichlet L-function zeros
# =====================================================================
def compute_dirichlet_zeros(q, chi, label, n_zeros=300):
    """Find zeros of L(s, χ) on the critical line."""
    from mpmath import dirichlet as mp_dirichlet
    
    print(f"  Computing {label}...")
    zeros = []
    t = 1.0
    dt = 0.1
    prev_val = None
    t0 = time.time()
    
    while t < 1000.0 and len(zeros) < n_zeros:
        try:
            val = float(mp_dirichlet(mpc(0.5, t), chi).real)
            if prev_val is not None and prev_val * val < 0:
                t_lo, t_hi = t - dt, t
                for _ in range(60):
                    t_mid = (t_lo + t_hi) / 2
                    v_mid = float(mp_dirichlet(mpc(0.5, t_mid), chi).real)
                    if v_mid * prev_val < 0:
                        t_hi = t_mid
                    else:
                        t_lo = t_mid
                        prev_val = v_mid
                zeros.append((t_lo + t_hi) / 2)
            prev_val = val
        except Exception:
            prev_val = None
        t += dt
    
    elapsed = time.time() - t0
    print(f"    Found {len(zeros)} zeros in {elapsed:.1f}s")
    return zeros

characters = [
    (3,  [0, 1, -1],           "chi_mod3"),
    (4,  [0, 1, 0, -1],       "chi_mod4"),
    (5,  [0, 1, -1, -1, 1],   "chi_mod5"),
    (7,  [0, 1, 1, -1, 1, -1, -1], "chi_mod7"),
    (8,  [0, 1, 0, -1, 0, -1, 0, 1], "chi_mod8"),
    (11, [0, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1], "chi_mod11"),
    (13, [0, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1], "chi_mod13"),
]

print(f"\nComputing Dirichlet L-function zeros...")
for q, chi, label in characters:
    zeros_d = compute_dirichlet_zeros(q, chi, label, n_zeros=300)
    if len(zeros_d) > 0:
        np.savetxt(os.path.join(OUTPUT_DIR, f"{label}_zeros.csv"), zeros_d, fmt='%.20f')
        print(f"    Saved to {OUTPUT_DIR}/{label}_zeros.csv")

print(f"\nAll zeros saved to {OUTPUT_DIR}/")
print("Now run the Rust analysis program.")
