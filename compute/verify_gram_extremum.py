#!/usr/bin/env python3
"""Check the September 2026 finite Gram claims on a CPU.

Requires numpy and mpmath. This is numerical corroboration, NOT interval
certification: even agreement between precisions is not a rigorous error bound.
The sufficient theorem and its limitations are in papers/gram_extremum_review.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import mpmath as mp
import numpy as np


def first_primes(count: int) -> list[int]:
    primes = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def gram(zeros, primes, displacement):
    """Entrywise construction at arbitrary precision; no trace normalization."""
    logs = [mp.log(p) for p in primes]
    weights = [l / mp.sqrt(p) * mp.cosh(displacement * l)
               for p, l in zip(primes, logs)]
    return mp.matrix([[mp.fsum(w * mp.cos((x - y) * l)
                              for w, l in zip(weights, logs))
                       for y in zeros] for x in zeros])


def condition(matrix):
    ev = mp.eigsy(matrix, eigvals_only=True)
    if ev[0] <= 0:
        raise ValueError("The condition-number criterion requires positive definiteness")
    return ev[matrix.rows - 1] / ev[0]


def stop_loss(logs, difference, threshold):
    return mp.fsum(d * max(l - threshold, 0) for l, d in zip(logs, difference))


def check_case(zeros, primes):
    n = len(zeros)
    logs = [mp.log(p) for p in primes]
    weights = [l / mp.sqrt(p) for p, l in zip(primes, logs)]
    center = gram(zeros, primes, mp.mpf(0))
    ev, vectors = mp.eigsy(center)
    assert ev[0] > 0
    assert ev[1] > ev[0] and ev[n - 1] > ev[n - 2]
    masses = []
    for column in [0, n - 1]:
        u = [vectors[i, column] for i in range(n)]
        masses.append([
            w * (mp.fsum(v * mp.cos(z * l) for v, z in zip(u, zeros)) ** 2
                 + mp.fsum(v * mp.sin(z * l) for v, z in zip(u, zeros)) ** 2)
            / ev[column] for w, l in zip(weights, logs)
        ])
    error_limit = mp.power(10, -mp.mp.dps + 10)
    assert max(abs(mp.fsum(m) - 1) for m in masses) < error_limit
    difference = [top - bottom for bottom, top in zip(*masses)]
    knots = [stop_loss(logs, difference, l) for l in logs]
    h0 = stop_loss(logs, difference, mp.mpf(0))
    assert abs(h0 - knots[0]) < error_limit
    assert knots[-1] == 0
    r = mp.fsum(d * l**2 / 2 for d, l in zip(difference, logs))

    # Independent checks of the integrated-tail identity and fixed-mode bound.
    t = mp.mpf("0.25")
    response = [mp.fsum(m * mp.cosh(t * l) for m, l in zip(mass, logs))
                for mass in masses]
    integral = mp.fsum(mp.quad(lambda u: mp.cosh(t * u)
                              * stop_loss(logs, difference, u), [left, right])
                      for left, right in zip([mp.mpf(0)] + logs[:-1], logs))
    identity_error = abs(response[1] - response[0] - t**2 * integral)
    assert identity_error < error_limit
    rayleigh_bound = ev[n - 1] / ev[0] * response[1] / response[0]
    assert condition(gram(zeros, primes, t)) + error_limit >= rayleigh_bound

    sigma_values = ["0.25", "0.40", "0.50", "0.60", "0.75", "0.90"]
    table = {sigma: str(condition(gram(zeros, primes, mp.mpf(sigma) - mp.mpf("0.5"))))
             for sigma in sigma_values}
    return {
        "zero_count": n, "prime_count": len(primes), "largest_prime": primes[-1],
        "condition_numbers": table, "local_log_condition_coefficient": str(r),
        "minimum_nontrivial_stop_loss": str(min(knots[:-1])),
        "stop_loss_at_zero": str(h0), "stop_loss_at_log_primes": list(map(str, knots)),
        "all_nontrivial_knots_positive_numerically": all(h > 0 for h in knots[:-1]),
        "smallest_eigenvalue": str(ev[0]),
        "extreme_eigenvalue_gaps": [str(ev[1] - ev[0]), str(ev[n - 1] - ev[n - 2])],
        "eigenpair_residual_max": str(max(abs(x) for x in center * vectors
                                          - vectors * mp.diag(ev))),
        "mass_normalization_error_max": str(max(abs(mp.fsum(m) - 1) for m in masses)),
        "integrated_tail_identity_error": str(identity_error),
    }


def geometry_check(zeros, primes):
    """Independent feature factorization and precisely defined PCA diagnostics."""
    z = np.array([float(x) for x in zeros])
    p = np.array(primes)
    logs = np.log(p)
    weights = logs / np.sqrt(p)
    phases = z[:, None] * logs
    features = np.concatenate([np.cos(phases), np.sin(phases)], axis=1)
    features *= np.sqrt(np.tile(weights, 2))
    g = features @ features.T
    direct = np.array(gram(zeros, primes, mp.mpf(0)).tolist(), dtype=float)
    entry_error = np.max(np.abs(g - direct))
    assert entry_error < 1e-10
    centered_features = features - features.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered_features @ centered_features.T)
    positive = eigenvalues[eigenvalues > eigenvalues[-1] * 1e-12]
    fractions = positive / positive.sum()
    distances_squared = np.sum((features[:, None, :] - features[None, :, :])**2, axis=2)
    chordal_formula = np.array([
        [4 * np.sum(weights * np.sin((x - y) * logs / 2)**2) for y in z] for x in z])
    chordal_error = np.max(np.abs(distances_squared - chordal_formula))
    assert chordal_error < 1e-10
    return {
        "feature_vs_high_precision_gram_max_error": float(entry_error),
        "chordal_distance_squared_max_error": float(chordal_error),
        "centered_rank_at_relative_tolerance_1e_minus_12": len(positive),
        "top_three_variance_fraction": float(fractions[-3:].sum()),
        "top_four_variance_fraction": float(fractions[-4:].sum()),
        "participation_ratio": float(1 / np.sum(fractions**2)),
        "entropy_effective_rank": float(np.exp(-np.sum(fractions * np.log(fractions)))),
    }


def xi(z):
    if z == 0 or z == 1:
        return mp.mpf("0.5")
    return z * (z - 1) / 2 * mp.power(mp.pi, -z / 2) * mp.gamma(z / 2) * mp.zeta(z)


def xi_prime(z):
    prefactor = z * (z - 1) / 2 * mp.power(mp.pi, -z / 2) * mp.gamma(z / 2)
    logarithmic_derivative = 1 / z + 1 / (z - 1) - mp.log(mp.pi) / 2 + mp.digamma(z / 2) / 2
    return prefactor * (mp.zeta(z, derivative=1) + logarithmic_derivative * mp.zeta(z))


def check_xi():
    """Truncated quadrature only; the infinite-integral result has a written proof."""
    with mp.workdps(30):
        def energy(sigma, derivative=False, cutoff=60):
            f = xi_prime if derivative else xi
            return 2 * mp.quad(lambda t: abs(f(mp.mpc(sigma, t)))**2,
                               [0, 5, 10, 20, 40, cutoff])

        center = mp.mpf("0.5")
        e0 = energy(center)
        sides = [energy(mp.mpf(s)) for s in ["0.25", "0.75"]]
        e1 = energy(center, derivative=True)
        h = mp.mpf("0.005")
        finite_difference = 2 * (energy(center + h) - e0) / h**2
        assert sides[0] > e0 and abs(sides[0] - sides[1]) < mp.mpf("1e-25")
        assert abs(finite_difference - 4 * e1) < mp.mpf("1e-5")
        return {"precision_digits": 30, "cutoff": 60, "energy_at_half": str(e0),
                "energy_at_quarter": str(sides[0]), "derivative_energy_at_half": str(e1),
                "four_times_derivative_energy": str(4 * e1),
                "second_difference_step": str(h), "second_difference": str(finite_difference),
                "cutoff_60_to_80_change": str(abs(energy(center, cutoff=80) - e0)),
                "status": "Numerical quadrature; no certified tail or rounding bound"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xi", action="store_true", help="also run the slower completed-xi quadrature")
    args = parser.parse_args()
    results = {"status": "Numerical checks only; not an interval certificate or an RH proof",
               "source": "papers/gram_extremum_review.md",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "python": platform.python_version(), "numpy": np.__version__, "mpmath": mp.__version__}
    primes = first_primes(120)
    runs = []
    for digits in [45, 75]:
        with mp.workdps(digits):
            zeros = [mp.im(mp.zetazero(n)) for n in range(1, 21)]
            cases = [check_case(zeros[:n], primes[:p]) for n, p in [(2, 5), (20, 30), (20, 120)]]
            assert all(mp.mpf(case["local_log_condition_coefficient"]) < 0 for case in cases[:2])
            assert cases[2]["all_nontrivial_knots_positive_numerically"]
            for case in cases[:2]:
                assert mp.mpf(case["condition_numbers"]["0.60"]) < mp.mpf(case["condition_numbers"]["0.50"])
            runs.append({"digits": digits, "cases": cases})
            if digits == 75:
                results["geometry"] = geometry_check(zeros, primes)
            print(f"{digits} digits: all assertions passed", flush=True)
    with mp.workdps(80):
        errors = [abs(mp.mpf(a) - mp.mpf(b))
                  for low, high in zip(runs[0]["cases"], runs[1]["cases"])
                  for a, b in zip(low["stop_loss_at_log_primes"], high["stop_loss_at_log_primes"])]
        condition_errors = [abs(mp.mpf(low["condition_numbers"][s]) - mp.mpf(high["condition_numbers"][s]))
                            for low, high in zip(runs[0]["cases"], runs[1]["cases"])
                            for s in low["condition_numbers"]]
        assert max(errors + condition_errors) < mp.mpf("1e-35")
        results["precision_comparison"] = {"maximum_stop_loss_difference": str(max(errors)),
                                           "maximum_condition_number_difference": str(max(condition_errors))}
    results["runs"] = runs
    if args.xi:
        results["xi_energy"] = check_xi()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
