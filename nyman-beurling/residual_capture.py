"""Investigate arithmetic correction of the fully optimized NB residual.

Finite floating-point evidence only. No uniform gain estimate is asserted.
Reports cancellation, removal of old-span components, and the trial-transfer
bound; exact unit-cell identities diagnose where a correction meets the error.
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
from gain_attempt import chain_matrix
from nb_gram_np import b_vector, mobius


def direction_stats(correlation, cost, old_coordinates, error, full_gain, n):
    old_cost = float(old_coordinates @ old_coordinates)
    projected_cost = cost - old_cost
    if not 0 < projected_cost <= cost or cost <= 0:
        raise ArithmeticError("Invalid correction cost after orthogonal cleanup")
    raw_gain = correlation**2 / cost
    projected_gain = correlation**2 / projected_cost
    if projected_gain > full_gain + 5e-10:
        raise ArithmeticError("A single correction exceeds the full projection gain")
    factor = float(np.log(n) / error)
    return {"correlation": float(correlation), "raw_cost": float(cost),
            "old_span_cost": old_cost, "projected_cost": float(projected_cost),
            "raw_gain": float(raw_gain), "projected_gain": float(projected_gain),
            "raw_eta_logN": float(factor * raw_gain),
            "projected_eta_logN": float(factor * projected_gain),
            "old_span_fraction_of_cost": old_cost / cost}


def cell_arrays(coefficients, parents, weights, stop):
    """Exact-in-form unit-cell integrals, evaluated in long double.

    R(u)=B_j-a*u, H(u)=H_j on (j,j+1), j>=1. The correction is
    sum weights[k]*(e_(2k)-e_k/2). Array index zero is the interval (0,1).
    """
    if not isinstance(stop, (int, np.integer)) or stop < 2:
        raise ValueError("stop must be an integer >= 2")
    c = np.asarray(coefficients, dtype=np.longdouble)
    parents = np.asarray(parents, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.longdouble)
    if parents.shape != weights.shape or np.any(parents <= 0):
        raise ValueError("positive parents and matching weights required")
    slope = np.sum(c / np.arange(1, len(c) + 1, dtype=np.longdouble))
    increments = np.zeros(stop, dtype=np.longdouble)
    for k, value in enumerate(c, 1):
        increments[k::k] += value
    levels = 1 + np.cumsum(increments)
    changes = np.zeros(stop, dtype=np.longdouble)
    for k, value in zip(parents, weights):
        changes[k::2*k] += value / 2
        changes[2*k::2*k] -= value / 2
    h = np.cumsum(changes)
    j = np.arange(1, stop, dtype=np.longdouble)
    inverse_cell = 1 / (j * (j + 1))
    logarithm = np.log1p(1 / j)
    correlation = np.zeros(stop, dtype=np.longdouble)
    cost = np.zeros(stop, dtype=np.longdouble)
    energy = np.zeros(stop, dtype=np.longdouble)
    correlation[1:] = h[1:] * (levels[1:] * inverse_cell - slope * logarithm)
    cost[1:] = h[1:]**2 * inverse_cell
    # Mean-square decomposition avoids cancellation between three large terms.
    energy[0] = slope**2
    energy[1:] = ((levels[1:] * logarithm - slope)**2
                  + levels[1:]**2 * (inverse_cell - logarithm**2))
    if np.min(energy) < -1e-16:
        raise ArithmeticError("Unit-cell energy lost numerical positivity")
    return {"correlation": correlation, "cost": cost, "energy": energy,
            "correction_levels": h, "residual_intercepts": levels, "slope": slope}


def spatial_profile(c, parents, weights, n, error, stats, multiplier):
    stop = multiplier * n
    cells = cell_arrays(c, parents, weights, stop)
    starts = [0, int(parents[0]), n]
    while starts[-1] < stop:
        starts.append(min(2 * starts[-1], stop))
    bands = []
    for left, right in zip(starts, starts[1:]):
        bands.append({"u_from": left, "u_to": right,
                      **{name: float(cells[name][left:right].sum())
                         for name in ("correlation", "cost", "energy")}})
    tail = {"correlation": stats["correlation"] - float(cells['correlation'].sum()),
            "cost": stats["raw_cost"] - float(cells['cost'].sum()),
            "energy": error - float(cells['energy'].sum())}
    if tail['cost'] <= 0 or tail['energy'] <= 0:
        raise ArithmeticError("Nonpositive tail remainder in profile")
    tail['cauchy_schwarz_bound'] = float(np.sqrt(tail['cost'] * tail['energy']))
    if abs(tail['correlation']) > tail['cauchy_schwarz_bound'] + 5e-10:
        raise ArithmeticError("Tail correlation exceeds its Cauchy--Schwarz bound")
    tail['u_from'] = stop
    partial_correlation = float(cells['correlation'].sum())
    correlation_lower = max(0., abs(partial_correlation) - tail['cauchy_schwarz_bound'])
    if correlation_lower > abs(stats['correlation']) + 5e-10:
        raise ArithmeticError("Head-minus-tail correlation bound failed")
    available = error - float(cells['energy'][:parents[0]].sum())
    capture = stats['raw_gain'] / available
    np.testing.assert_allclose(stats['raw_eta_logN'], np.log(n) * available / error * capture,
                               atol=1e-11, rtol=1e-9)
    return {"bands": bands, "tail_from_matrix_minus_cells": tail,
            "available_tail_energy": available,
            "available_tail_fraction_times_logN": float(np.log(n) * available / error),
            "direction_capture_of_available_tail": capture,
            "finite_head_minus_tail_correlation_lower_bound": correlation_lower,
            "eta_lower_bound_expression": float(np.log(n) * correlation_lower**2 / (error * stats['raw_cost'])),
            "absolute_cell_correlations_up_to_stop": float(abs(cells['correlation']).sum()),
            "status": "Floating-point unit-cell identities and remainder checks, not enclosures"}


def investigate(gram, sizes, profile_sizes=(), multiplier=64):
    gram, target = validate_problem(gram, b_vector(len(gram)))
    sizes = sorted(set(sizes))
    if not sizes or sizes[0] < 2 or 2 * sizes[-1] > len(gram):
        raise ValueError("Sizes must satisfy 2 <= N <= matrix_size/2")
    factor = cholesky(gram, lower=True)
    coordinates = solve_triangular(factor, target, lower=True)
    errors = 1 - np.cumsum(coordinates**2)
    if np.any(errors <= 0):
        raise ArithmeticError("Nonpositive projection error")
    mu = mobius(len(gram)).astype(float)
    rows = []
    for index, n in enumerate(sizes):
        old_factor = factor[:n, :n]
        c = solve_triangular(old_factor.T, coordinates[:n], lower=False)
        error = float(errors[n - 1])
        full_gain = float(errors[2*n - 1] * -1 + error)
        if full_gain <= 0:
            raise ArithmeticError("Nonpositive full refinement gain")
        normal_error = float(np.max(abs(gram[:n, :n] @ c - target[:n])))
        if normal_error > 1e-10:
            raise ArithmeticError("Old normal equations are not accurately solved")
        parents = np.arange(n // 2 + 1, n + 1)
        matrix = chain_matrix(gram, parents)
        cross = gram[:n, 2*parents - 1] - gram[:n, parents - 1] / 2
        chain_target = np.log(2) / (2 * parents)
        q = chain_target - cross.T @ c
        candidates = {"mobius_chain": mu[parents - 1],
                      "tapered_mobius_chain": mu[parents - 1] * np.log(2*n/parents)}
        directions = {}
        for name, v in candidates.items():
            old_rhs = cross @ v
            old_coordinates = solve_triangular(old_factor, old_rhs, lower=True)
            correlation = float(q @ v)
            stats = direction_stats(correlation, float(v @ matrix @ v), old_coordinates,
                                    error, full_gain, n)
            terms = q * v
            stats.update(absolute_arithmetic_contributions=float(abs(terms).sum()),
                         signed_alignment=float(abs(correlation) / abs(terms).sum()),
                         positive_contributions=float(terms[terms > 0].sum()),
                         negative_contributions=float(terms[terms < 0].sum()))
            if name == 'mobius_chain':
                trial = -mu[:n] * np.log(n / np.arange(1, n + 1)) / np.log(n)
                difference = trial - c
                excess_energy = float(difference @ gram[:n, :n] @ difference)
                trial_correlation = float(chain_target @ v - old_rhs @ trial)
                transfer_bound = float(np.sqrt(excess_energy * stats['old_span_cost']))
                if abs(trial_correlation - correlation) > transfer_bound + 5e-10:
                    raise ArithmeticError("Trial-transfer inequality failed")
                stats['trial_transfer'] = {
                    "trial_energy": error + excess_energy,
                    "excess_energy_above_optimum": excess_energy,
                    "trial_correlation": trial_correlation,
                    "actual_correlation_discrepancy": abs(trial_correlation - correlation),
                    "transfer_error_upper_bound": transfer_bound,
                    "guaranteed_absolute_correlation_lower_bound": max(0., abs(trial_correlation) - transfer_bound),
                }
                if n in profile_sizes:
                    stats['spatial_profile'] = spatial_profile(c, parents, v, n, error, stats, multiplier)
            directions[name] = stats
        # A second arithmetic family: use every new dilation, then remove its old-span part.
        v = mu[n:2*n]
        old_rhs = gram[:n, n:2*n] @ v
        old_coordinates = solve_triangular(old_factor, old_rhs, lower=True)
        correlation = float(target[n:2*n] @ v - old_rhs @ c)
        directions['mobius_full_block'] = direction_stats(
            correlation, float(v @ gram[n:2*n, n:2*n] @ v), old_coordinates, error, full_gain, n)
        rows.append({"N": n, "error_N": error, "error_2N": float(errors[2*n - 1]),
                     "full_gain": full_gain, "full_eta_logN": float(np.log(n) * full_gain / error),
                     "normal_equation_max_error": normal_error, "directions": directions})
        if index % 64 == 0 or index + 1 == len(sizes):
            print(f"Checked {index + 1}/{len(sizes)} sizes; N={n}", flush=True)
    minima = {name: {kind: min(({"N": row['N'], "value": row['directions'][name][kind]}
                               for row in rows), key=lambda x: x['value'])
                     for kind in ['raw_eta_logN', 'projected_eta_logN']}
              for name in rows[0]['directions']}
    return {"status": "Finite numerical investigation; no uniform gain proof", "rows": rows,
            "direction_minima": minima,
            "projection_errors_at_dyadic_sizes": {str(n): float(errors[n - 1])
                                                   for n in [2**j for j in range(1, len(gram).bit_length())]
                                                   if n <= len(gram)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gram', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--last', type=int, default=4096)
    parser.add_argument('--step', type=int, default=16)
    parser.add_argument('--sizes', help='optional comma-separated sizes overriding the default scan')
    parser.add_argument('--profiles', default='32,128,512,1024,2048,4096')
    args = parser.parse_args()
    if args.step < 1 or args.last < 16:
        parser.error('--step must be positive and --last at least 16')
    profile_sizes = {int(x) for x in args.profiles.split(',') if int(x) <= args.last}
    sizes = (sorted({int(x) for x in args.sizes.split(',')}) if args.sizes else
             sorted(set(range(16, min(512, args.last) + 1))
                    | set(range(512, args.last + 1, args.step)) | profile_sizes | {args.last}))
    started = time.monotonic()
    result = investigate(np.load(args.gram, allow_pickle=False), sizes, profile_sizes)
    result.update(elapsed_seconds=time.monotonic() - started, python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__,
                  gram_sha256=hashlib.sha256(args.gram.read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ['residual_capture.py', 'gain_attempt.py',
                                              'block_refinement.py', 'nb_gram_np.py']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['direction_minima'], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == '__main__':
    main()
