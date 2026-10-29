#!/usr/bin/env python3
"""Numerical prime-side compression of Weil's form using the cos^4 probes.

Unlike the planted-spectrum experiment, this computes zeta's explicit
formula from primes, prime powers, its poles, and its gamma factor.
It is floating-point quadrature, NOT an interval-certified RH test.

  python compute/weil_prime_side.py --output compute/results/weil_prime_side.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma

from weil_signature import window_transform

# Rounded ordinates already recorded in nyman-beurling/symbol_zero.py.
# Used for a low-height cross-check only; never used to assemble prime-side W.
# The validation sum assigns unit multiplicity to each recorded zero.
KNOWN_ORDINATES = np.array([14.134725142, 21.022039639, 25.010857580,
                            30.424876126, 32.935061588, 37.586178159,
                            40.918719012])


def von_mangoldt_terms(bandwidth):
    limit = int(math.floor(math.exp(2*bandwidth)))
    sieve = np.ones(limit+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, math.isqrt(limit)+1):
        if sieve[p]:
            sieve[p*p::p] = False
    terms = []
    for p in np.flatnonzero(sieve):
        power = int(p)
        while power <= limit:
            terms.append((power, math.log(int(p))))
            power *= int(p)
    return sorted(terms)


def prime_side(probes, bandwidth, cutoff):
    probes = np.asarray(probes, dtype=float)
    if not 0 < bandwidth <= 3:
        raise ValueError("Use 0 < bandwidth <= 3 for this CPU reference instrument")
    if cutoff <= np.max(np.abs(probes))+8/bandwidth:
        raise ValueError("The integration cutoff must extend well past every probe")
    terms = von_mangoldt_terms(bandwidth)
    dimension = len(probes)
    poles = np.zeros((dimension, dimension))
    gamma = np.zeros_like(poles)
    primes = np.zeros_like(poles)
    reported_quad_error = np.zeros_like(poles)

    def phi(u):
        return (4/(3*bandwidth))*math.cos(math.pi*u/(2*bandwidth))**4

    # Short panels resolve the transform's oscillatory tail; convergence
    # is independently checked below at three integration cutoffs.
    edges = np.linspace(-cutoff, cutoff, math.ceil(2*cutoff*bandwidth/math.pi)+1)
    for i in range(dimension):
        for j in range(i, dimension):
            poles[i, j] = 2*np.real(window_transform(0.5j-probes[i], bandwidth)
                                    * window_transform(0.5j-probes[j], bandwidth))

            def integrand(t):
                density = (digamma(0.25+0.5j*t).real-math.log(math.pi))/(2*math.pi)
                return float((window_transform(t-probes[i], bandwidth)
                              * window_transform(t-probes[j], bandwidth)).real)*density

            parts = [quad(integrand, lo, hi, epsabs=1e-13, epsrel=1e-12)
                     for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
            gamma[i, j] = math.fsum(x[0] for x in parts)
            reported_quad_error[i, j] = math.fsum(x[1] for x in parts)

            def convolution_real(s):
                lo, hi = max(-bandwidth, s-bandwidth), min(bandwidth, s+bandwidth)
                return quad(lambda u: phi(u)*phi(s-u)
                            * math.cos(probes[i]*u+probes[j]*(s-u)),
                            lo, hi, epsabs=1e-13, epsrel=1e-12)[0]

            primes[i, j] = -2*math.fsum(weight/math.sqrt(n)*convolution_real(math.log(n))
                                       for n, weight in terms)
            for matrix in [poles, gamma, primes, reported_quad_error]:
                matrix[j, i] = matrix[i, j]

    matrix = poles+gamma+primes
    return dict(matrix=matrix.tolist(), eigenvalues=np.linalg.eigvalsh(matrix).tolist(),
                poles=poles.tolist(), gamma_integral=gamma.tolist(),
                prime_power_term=primes.tolist(),
                scipy_reported_integral_error=reported_quad_error.tolist(),
                integration_cutoff=cutoff, von_mangoldt_terms=terms)


def run():
    probes = KNOWN_ORDINATES[0]+np.array([-1.2, 1.2])
    records = [prime_side(probes, 1.0, cutoff) for cutoff in [64., 128., 256.]]
    reference = np.array(records[-1]["matrix"])
    convergence = [float(np.linalg.norm(np.array(r["matrix"])-reference, 2)) for r in records[:-1]]
    # Independent side of the explicit formula: insert only the seven
    # recorded low zeros and their conjugates. Rounding and omitted zeros
    # mean exact equality is neither expected nor asserted.
    v = window_transform(np.r_[KNOWN_ORDINATES, -KNOWN_ORDINATES][:, None]-probes).real
    zero_side = v.T@v
    discrepancy = float(np.linalg.norm(reference-zero_side, 2))
    if max(convergence) > 2e-9 or discrepancy > 1e-8:
        raise RuntimeError("Prime-side convergence / low-zero cross-check failed")
    if records[-1]["eigenvalues"][0] <= 0:
        raise RuntimeError("Unexpected sign in the low-height reference compression")
    return dict(
        scope="Floating-point prime-side Weil compression; not interval-certified and not a proof of RH",
        window="(4/(3B))*cos^4(pi*u/(2B)) on [-B,B]", bandwidth=1.0,
        fourier_convention="exp(-i*u*z)", probes=probes.tolist(),
        acceptance=dict(status="PASS", cutoff_difference_operator_norms=convergence,
                        seven_zero_pair_discrepancy_operator_norm=discrepancy,
                        convergence_tolerance=2e-9, low_zero_tolerance=1e-8),
        known_ordinates=KNOWN_ORDINATES.tolist(),
        low_zero_matrix=zero_side.tolist(), records=records,
        limitations=["No rigorous enclosure for quadrature roundoff or infinite gamma-integral tail",
                     "Seven rounded zero pairs are a cross-check, not a complete zero sum",
                     "Positivity of one finite compression does not imply RH"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"scope": report["scope"], "acceptance": report["acceptance"],
                      "prime_side_eigenvalues": report["records"][-1]["eigenvalues"]}, indent=2))


if __name__ == "__main__":
    main()
