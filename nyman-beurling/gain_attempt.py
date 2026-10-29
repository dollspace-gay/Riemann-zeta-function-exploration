"""Screen explicit residual-gain directions and check the chain-only obstruction.

This is a finite floating-point experiment, not a uniform gain proof. The
analytic obstruction is proved in uniform_gain_attempt.md. No zeros are used.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import cholesky, solve_triangular

from block_refinement import project, validate_problem
from nb_gram_np import b_vector, mobius


def chain_matrix(gram, parents):
    p, q = parents - 1, 2*parents - 1
    return (gram[np.ix_(q, q)] - gram[np.ix_(q, p)]/2
            - gram[np.ix_(p, q)]/2 + gram[np.ix_(p, p)]/4)


def run(gram, first, last):
    target = b_vector(len(gram))
    gram, target = validate_problem(gram, target)
    if not 2 <= first <= last or 2*last > len(gram):
        raise ValueError("Require 2 <= first <= last and a matrix of size >= 2*last")
    factor = cholesky(gram, lower=True)
    coordinates = solve_triangular(factor, target, lower=True)
    errors = 1 - np.cumsum(coordinates**2)
    if np.any(errors <= 0):
        raise ValueError("Nonpositive residual energy; cannot form fractional gains")
    mu = mobius(len(gram))
    rows = []
    for n in range(first, last+1):
        projection = project(gram[:n, :n], target[:n])
        np.testing.assert_allclose(projection.error, errors[n-1], atol=5e-12, rtol=0)
        parents = np.arange(n//2+1, n+1)
        p, q = parents-1, 2*parents-1
        matrix = chain_matrix(gram, parents)
        correlations = (target[q] - target[p]/2
                        - (gram[:n, q] - gram[:n, p]/2).T @ projection.coefficients)
        directions = {
            "old_coefficients": projection.coefficients[p],
            "mobius": mu[p].astype(float),
            "tapered_mobius": mu[p]*np.log(2*n/parents),
            "correlations": correlations,
            "diagonal_correlations": correlations/np.diag(matrix),
        }
        scores = {}
        for name, vector in directions.items():
            numerator = float(correlations @ vector)
            cost = float(vector @ matrix @ vector)
            if cost <= 0:
                raise ValueError(f"Nonpositive correction cost at N={n}: {name}")
            gain = numerator**2 / cost
            full_gain = float(errors[n-1] - errors[2*n-1])
            if gain > full_gain + 5e-11:
                raise ValueError("Directional gain exceeds the full projection gain")
            scores[name] = {"eta_logN": gain / projection.error * float(np.log(n)),
                            "correlation": numerator, "cost": cost}
        rows.append({"N": n, "error_N": projection.error,
                     "full_eta_logN": float((errors[n-1]-errors[2*n-1]) /
                                            projection.error*np.log(n)),
                     "directions": scores})

    log_two = float(np.log(2))
    floor = 1 - 2*log_two**2
    base = lambda x: 1-log_two/x
    norm_squared = 2*quad(lambda x: base(x)**2, .5, 1, epsabs=1e-13)[0]
    target_correlation = 2*quad(base, .5, 1, epsabs=1e-13)[0]
    weighted_mean = quad(lambda x: base(x)/x, .5, 1, epsabs=1e-13)[0]
    np.testing.assert_allclose([norm_squared, target_correlation, weighted_mean],
                               [floor, floor, 0], atol=1e-13, rtol=0)
    chain_only = []
    for n in (16, 32, 64, 128, 256, 512):
        if 2*n <= len(gram):
            parents = np.arange(1, n+1)
            error = project(chain_matrix(gram, parents), log_two/(2*parents)).error
            if error < floor - 5e-11:
                raise ValueError("Chain-only error violates the analytic obstruction")
            chain_only.append({"chain_count": n, "squared_error": error})
    return {
        "status": "Finite floating-point screen; no uniform gain proof or asymptotic refutation",
        "rows": rows,
        "orthogonalized_captured_target_energy": float(np.sum(coordinates**2)),
        "direction_minima": {
            name: min((row["directions"][name] | {"N": row["N"]} for row in rows),
                      key=lambda score: score["eta_logN"])
            for name in rows[0]["directions"]
        },
        "chain_only_obstruction": {
            "analytic_floor": floor, "base_interval_quadrature_norm_squared": norm_squared,
            "base_interval_quadrature_target_correlation": target_correlation,
            "base_interval_quadrature_weighted_mean": weighted_mean,
            "finite_projections": chain_only,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gram", type=Path, required=True)
    parser.add_argument("--first", type=int, default=16)
    parser.add_argument("--last", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ("gain_attempt.py", "block_refinement.py", "nb_gram_np.py")}
    result = run(np.load(args.gram, allow_pickle=False), args.first, args.last)
    result.update(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  source_sha256=sources, gram_path=str(args.gram.resolve()),
                  gram_sha256=hashlib.sha256(args.gram.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({"direction_minima": result["direction_minima"],
                      "chain_only_obstruction": result["chain_only_obstruction"]}, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
