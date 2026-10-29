"""Probe the fixed-residual Mobius correlation identity and its norm cost.

All NB evaluations are floating point. The fixed-N limit is a written theorem
using the classical L1 convergence theorem, not a conclusion of this scan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.linalg import cholesky, solve_triangular

from block_refinement import validate_problem
from nb_gram_np import b_vector, mobius


def investigate(gram, sizes):
    gram, target = validate_problem(gram, b_vector(len(gram)))
    sizes = sorted(set(sizes))
    if not sizes or sizes[0] < 2 or 2 * sizes[-1] > len(gram):
        raise ValueError("Require 2 <= N <= matrix_size/2")
    factor = cholesky(gram, lower=True)
    coordinates = solve_triangular(factor, target, lower=True)
    errors = 1 - np.cumsum(coordinates**2)
    if np.any(errors <= 0):
        raise ArithmeticError("Nonpositive projection energy")
    mu = mobius(len(gram)).astype(float)
    rows = []
    for n in sizes:
        old_factor = factor[:n, :n]
        c = solve_triangular(old_factor.T, coordinates[:n], lower=False)
        error = float(errors[n-1])
        all_correlations = target - c @ gram[:n, :]
        q = all_correlations[n:]
        normal_error = float(np.max(abs(gram[:n, :n] @ c - target[:n])))
        if normal_error > 1e-10:
            raise ArithmeticError("Old normal equations failed")
        slope = float(c @ (1 / np.arange(1, n+1)))
        sup_bound = float(1 + abs(c).sum())
        chain_limit = (1-error+c[0])/2 - np.log(2)*slope
        blocks = []
        m = 2*n
        while m <= len(gram):
            v = mu[n:m]
            old_rhs = gram[:n, n:m] @ v
            old_coordinates = solve_triangular(old_factor, old_rhs, lower=True)
            raw_cost = float(v @ gram[n:m, n:m] @ v)
            cost = raw_cost - float(old_coordinates @ old_coordinates)
            correlation = float(q[:m-n] @ v)
            if cost <= 0:
                raise ArithmeticError("Nonpositive projected cost")
            gain = correlation**2 / cost
            full_gain = error - float(errors[m-1])
            if gain > full_gain + 5e-10:
                raise ArithmeticError("Direction exceeds full projection gain")
            # Direct pairing with chi + S_M, retaining all old Mobius terms.
            direct_defect = float(1 - c @ target[:n] + target[:m] @ mu[:m]
                                  - c @ (gram[:n, :m] @ mu[:m]))
            if abs(direct_defect - (error + correlation)) > 5e-10:
                raise ArithmeticError("Finite signed-correlation identity failed")
            parents = np.arange(1, m//2+1)
            chain_correlation = float(mu[parents-1] @
                                      (all_correlations[2*parents-1] - all_correlations[parents-1]/2))
            blocks.append({"M": m, "correlation": correlation,
                           "relative_capture_minus_C_over_E": -correlation / error,
                           "relative_defect_abs_C_plus_E": abs(correlation + error) / error,
                           "raw_cost": raw_cost, "projected_cost": cost,
                           "cost_over_E_logN": cost / (error * np.log(n)),
                           "gain": gain, "eta_logN": np.log(n) * gain / error,
                           "full_gain": full_gain,
                           "chain_parent_cutoff": m//2,
                           "cumulative_chain_correlation": chain_correlation,
                           "chain_limit_defect_over_E": abs(chain_correlation-chain_limit)/error,
                           "identity_discrepancy": abs(direct_defect-error-correlation)})
            m *= 2
        rows.append({"N": n, "energy": error, "residual_sup_bound": sup_bound,
                     "residual_slope": slope,
                     "L1_transfer_factor": sup_bound + abs(slope) / np.log(2),
                     "fixed_residual_chain_limit": chain_limit,
                     "normal_equation_max_error": normal_error, "blocks": blocks})
    return {"status": "Finite numerical checks; no uniform estimate or RH proof",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gram', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', default='32,128,512,1024,2048,4096')
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(',')]
    started = time.monotonic()
    result = investigate(np.load(args.gram, allow_pickle=False), sizes)
    with args.gram.open('rb') as stream:
        gram_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    result.update(elapsed_seconds=time.monotonic()-started,
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  gram_sha256=gram_hash,
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ['mobius_correlation_probe.py', 'block_refinement.py',
                                              'nb_gram_np.py']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Checked {sum(len(row['blocks']) for row in result['rows'])} (N,M) pairs")
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
