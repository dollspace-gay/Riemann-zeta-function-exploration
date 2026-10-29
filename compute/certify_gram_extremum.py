#!/usr/bin/env python3
"""Reconstruct the finite interval proof described in the supplied REPORT.md.

Only zero-bracket endpoints are read from --certificate; no claimed eigenvalue,
root sign, residual, or hinge bound is trusted. Without that option, generate
fresh brackets. Uses mpmath.iv (experimental), not a proof-assistant kernel.
See papers/gram_extremum_review.md for the mathematical justification.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform

import mpmath as mp
import numpy as np

from verify_gram_extremum import first_primes

iv = mp.iv


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def bounds(x):
    """Serialize with deliberate outward padding, then verify that padding."""
    low, high = mp.mpf(x.a), mp.mpf(x.b)
    if low == high == 0:
        return ["0", "0"]
    scale = mp.power(10, mp.floor(mp.log10(max(abs(low), abs(high)))) - 48)
    lower = mp.nstr((mp.floor(low / scale) - 1) * scale, 100)
    upper = mp.nstr((mp.ceil(high / scale) + 1) * scale, 100)
    require(iv.mpf(lower).b <= x.a and iv.mpf(upper).a >= x.b,
            "Outward serialization failed")
    return [lower, upper]


def bernoulli_even(count):
    """Exact rational Bernoulli numbers; the B1 sign is immaterial here."""
    work = []
    result = []
    for m in range(2 * count + 1):
        work.append(Fraction(1, m + 1))
        for j in range(m, 0, -1):
            work[j - 1] = j * (work[j - 1] - work[j])
        if m > 0 and m % 2 == 0:
            result.append(work[0])
    return result


def hardy_z(endpoint, bernoulli):
    """Euler--Maclaurin zeta with an explicit complex remainder rectangle."""
    n, order = 128, len(bernoulli)
    t = iv.mpf(endpoint)
    s = iv.mpc(iv.mpf(1) / 2, t)
    rising = [iv.mpc(1)]
    for j in range(2 * order):
        rising.append(rising[-1] * (s + j))
    value = sum(iv.mpf(k)**(-s) for k in range(1, n))
    value += iv.mpf(n)**(1 - s) / (s - 1) + iv.mpf(n)**(-s) / 2
    for k, b in enumerate(bernoulli, 1):
        coefficient = iv.mpf(b.numerator) / b.denominator / math.factorial(2 * k)
        value += coefficient * rising[2 * k - 1] * iv.mpf(n)**(1 - s - 2 * k)
    remainder = (4 * abs(rising[2 * order]) / (2 * iv.pi)**(2 * order)
                 * iv.mpf(n)**(iv.mpf(1) / 2 - 2 * order) / (2 * order - iv.mpf(1) / 2))
    error = iv.mpf([-remainder.b, remainder.b])
    value += iv.mpc(error, error)
    theta = iv.loggamma(iv.mpc(iv.mpf(1) / 4, t / 2)).imag - t * iv.log(iv.pi) / 2
    z = (iv.exp(iv.mpc(0, theta)) * value).real
    return z, remainder


def root_checks(enclosures):
    bernoulli = bernoulli_even(32)
    records = []
    for index, enclosure in enumerate(enclosures, 1):
        left, error_left = hardy_z(enclosure[0], bernoulli)
        right, error_right = hardy_z(enclosure[1], bernoulli)
        require((left.b < 0 and right.a > 0) or (left.a > 0 and right.b < 0),
                f"Hardy Z sign change not certified in bracket {index}")
        records.append({"index": index, "enclosure": enclosure,
                        "hardy_z_left": bounds(left), "hardy_z_right": bounds(right),
                        "remainder_left": bounds(error_left), "remainder_right": bounds(error_right)})
    return records


def interval_gram(zeros, primes, displacement=0):
    logs = [iv.log(p) for p in primes]
    # iv has no cosh convenience function in some releases.
    weights = [l / iv.sqrt(p) * (iv.exp(displacement * l) + iv.exp(-displacement * l)) / 2
               for p, l in zip(primes, logs)]
    return iv.matrix([[sum(w * iv.cos((x - y) * l) for w, l in zip(weights, logs))
                       for y in zeros] for x in zeros])


def inverse_rational_candidates(rows):
    """Invert the decimal candidate basis exactly, avoiding interval pivot ordering."""
    n = len(rows)
    augmented = [[Fraction(x) for x in row] + [Fraction(i == j) for j in range(n)]
                 for i, row in enumerate(rows)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if augmented[i][j]), None)
        require(pivot is not None, "Candidate basis is singular")
        augmented[j], augmented[pivot] = augmented[pivot], augmented[j]
        denominator = augmented[j][j]
        augmented[j] = [x / denominator for x in augmented[j]]
        for i in range(n):
            if i != j:
                factor = augmented[i][j]
                augmented[i] = [a - factor * b for a, b in zip(augmented[i], augmented[j])]
    return iv.matrix([[iv.mpf(x.numerator) / x.denominator for x in row[n:]]
                      for row in augmented])


def spectral_enclosures(matrix):
    n = matrix.rows
    midpoint = np.array([[float((mp.mpf(matrix[i, j].a) + mp.mpf(matrix[i, j].b)) / 2)
                          for j in range(n)] for i in range(n)])
    values, vectors = np.linalg.eigh(midpoint)
    # Decimal strings define exact rational candidates. They need not be orthogonal.
    candidate_rows = [[str(vectors[i, j]) for j in range(n)] for i in range(n)]
    v = iv.matrix(candidate_rows)
    orthogonality_error = v.T * v - iv.eye(n)
    require(max(sum(abs(orthogonality_error[i, j]) for j in range(n)).b for i in range(n)) < 1,
            "Candidate basis invertibility not certified")
    transformed = inverse_rational_candidates(candidate_rows) * matrix * v
    intervals = []
    for i in range(n):
        radius = sum(abs(transformed[i, j]) for j in range(n) if i != j)
        intervals.append(transformed[i, i] + iv.mpf([-radius.b, radius.b]))
    require(intervals[0].a > 0, "Positive definiteness not certified")
    require(all(a.b < b.a for a, b in zip(intervals, intervals[1:])),
            "Gershgorin intervals do not isolate all eigenvalues")
    modes, records = [], []
    for column in [0, n - 1]:
        candidate = v[:, column]
        candidate /= iv.sqrt(sum(x**2 for x in candidate))
        mu = iv.mpf(str(values[column]))
        distances = [abs(mu - x) for i, x in enumerate(intervals) if i != column]
        separation = min(x.a for x in distances)
        require(separation > 0, "Candidate is not separated from the other eigenvalues")
        residual = matrix * candidate - mu * candidate
        epsilon = iv.sqrt(sum(x**2 for x in residual))
        eta = 2 * epsilon.b / separation
        require(eta.b < 1, "Eigenvector error bound too large")
        modes.append([x + iv.mpf([-eta.b, eta.b]) for x in candidate])
        records.append({"column": column, "residual_norm": bounds(epsilon),
                        "separation_lower": bounds(separation), "vector_error_bound": bounds(eta)})
    return intervals, modes, records


def certify(enclosures, prime_count=120):
    zeros = [iv.mpf(e) for e in enclosures]
    primes = first_primes(prime_count)
    logs = [iv.log(p) for p in primes]
    matrix = interval_gram(zeros, primes)
    eigenvalues, modes, records = spectral_enclosures(matrix)
    masses = []
    for vector, eigenvalue in zip(modes, [eigenvalues[0], eigenvalues[-1]]):
        mass = []
        for p, l in zip(primes, logs):
            cosine = sum(v * iv.cos(z * l) for v, z in zip(vector, zeros))
            sine = sum(v * iv.sin(z * l) for v, z in zip(vector, zeros))
            mass.append(l / iv.sqrt(p) * (cosine**2 + sine**2) / eigenvalue)
        masses.append(mass)
    differences = [top - bottom for bottom, top in zip(*masses)]
    hinges = [sum(differences[k] * (logs[k] - logs[j]) for k in range(j + 1, len(primes)))
              for j in range(len(primes) - 1)]
    require(all(x.a > 0 for x in hinges), "Some hinge inequalities were not certified positive")
    counterexample = []
    for sigma in ["0.5", "0.9"]:
        g = interval_gram(zeros[:2], primes[:5], iv.mpf(sigma) - iv.mpf(1) / 2)
        a, b = g[0, 0], abs(g[0, 1])
        require((a - b).a > 0, "Counterexample matrix not certified positive definite")
        counterexample.append((a + b) / (a - b))
    require(counterexample[1].b < counterexample[0].a, "Counterexample intervals overlap")
    return {"condition_at_center": bounds(eigenvalues[-1] / eigenvalues[0]),
            "eigenvalue_intervals": [bounds(x) for x in eigenvalues],
            "eigenvector_bounds": records,
            "hinge_intervals": [{"prime": p, "interval": bounds(x)} for p, x in zip(primes, hinges)],
            "minimum_hinge_lower_bound": bounds(min(x.a for x in hinges))[0],
            "counterexample_conditions_at_half_and_point9": [bounds(x) for x in counterexample]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, help="supplied JSON; only its zero intervals are inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mp.mp.dps = 100
    iv.dps = 65
    if args.certificate:
        source = json.loads(args.certificate.read_text())
        enclosures = [x["enclosure"] for x in source["zero_intervals"]]
    else:
        radius = mp.mpf("1e-20")
        enclosures = [[mp.nstr(mp.im(mp.zetazero(n)) - radius, 80),
                       mp.nstr(mp.im(mp.zetazero(n)) + radius, 80)] for n in range(1, 21)]
    require(len(enclosures) == 20, "This fixed example requires 20 brackets")
    require(all(iv.mpf(e[0]).b < iv.mpf(e[1]).a for e in enclosures), "Invalid bracket")
    require(all(iv.mpf(a[1]).b < iv.mpf(b[0]).a for a, b in zip(enclosures, enclosures[1:])),
            "Brackets must be ordered and disjoint")
    roots = root_checks(enclosures)
    print("20 Hardy Z sign changes certified", flush=True)
    result = certify(enclosures)
    result.update({"status": "PASS: interval verification of this finite family, not RH or a Lean proof",
                   "root_checks": roots, "interval_decimal_precision": iv.dps,
                   "root_indexing": "No independent zero count; all real ordinates in these brackets are covered",
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "python": platform.python_version(), "numpy": np.__version__, "mpmath": mp.__version__,
                   "input_certificate_sha256": (hashlib.sha256(args.certificate.read_bytes()).hexdigest()
                                                 if args.certificate else None)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("119 positive hinges and the two-zero counterexample certified", flush=True)
    print(f"Minimum hinge lower bound: {result['minimum_hinge_lower_bound']}", flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
