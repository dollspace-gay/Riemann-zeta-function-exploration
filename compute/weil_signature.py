#!/usr/bin/env python3
"""Finite planted-spectrum experiments for the October 2026 RH synthesis.

This is a diagnostic model of the zero-side Weil form, NOT an evaluation
of the full zeta explicit formula and NOT a numerical test of RH.
Planted pairs, clusters, and masking backgrounds use unit-multiplicity atoms.

Run from the repository root:
  python compute/weil_signature.py --output compute/results/weil_signature.json

Only NumPy and SciPy are required. No zero tables, GPU, or network calls.
The seed and acceptance tolerances below are part of the experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad

SEED = 20261028
ABS_TOL = 2e-11


def window_transform(z, bandwidth=1.0):
    """Transform of (4/(3B))*cos^4(pi*u/(2B)) on [-B,B], exp(-i*u*z).

    The zero extension of the window is C^3, hence an admissible C_c^2
    test window. The entire sinc expression handles removable values:
    F(0)=1, F(+/-pi/B)=2/3, F(+/-2pi/B)=1/6.
    """
    z = np.asarray(z, dtype=np.complex128)
    q = bandwidth * z / math.pi
    return (np.sinc(q) + (2/3)*(np.sinc(q-1)+np.sinc(q+1))
            + (1/6)*(np.sinc(q-2)+np.sinc(q+2)))


def transform_by_quadrature(z, bandwidth):
    """Independent integral, used only as an oracle for the closed form."""
    def integrand(u):
        return (4 / (3 * bandwidth)
                * math.cos(math.pi * u / (2 * bandwidth))**4
                * np.exp(-1j * u * z))
    re = quad(lambda u: integrand(u).real, -bandwidth, bandwidth,
              epsabs=1e-12, epsrel=1e-12)[0]
    im = quad(lambda u: integrand(u).imag, -bandwidth, bandwidth,
              epsabs=1e-12, epsrel=1e-12)[0]
    return re + 1j * im


def pair_forms(center, displacement, probes, bandwidth=1.0):
    """Unit-multiplicity reflected pair z=center +/- i*displacement.

    Correct zero-side contribution: vv^T + conjugate(vv^T).
    PSD control: vv* + conjugate(v)v^T. The latter erases inertia.
    For common multiplicity m, multiply both returned matrices by m.
    """
    v = window_transform(center + 1j * displacement - probes, bandwidth)
    signed = 2 * np.real(np.outer(v, v))
    psd_control = 2 * np.real(np.outer(v, v.conj()))
    return signed, psd_control


def online_form(zeros, probes, bandwidth=1.0):
    """Sum unit-multiplicity on-line atoms; repeated ordinates add weight."""
    v = window_transform(np.asarray(zeros)[:, None] - probes, bandwidth).real
    return v.T @ v


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def divisor_box(primes, exponents):
    """Support, least tensor eigenvalue, and tensor eigenvector."""
    support = np.array([1], dtype=np.int64)
    vector = np.array([1.0])
    predicted = 1.0
    for p, exponent in zip(primes, exponents, strict=True):
        a = np.arange(exponent + 1)
        block = float(p) ** (-np.abs(a[:, None] - a[None, :]) / 2)
        values, vectors = np.linalg.eigh(block)
        predicted *= float(values[0])
        support = (support[:, None] * p ** a[None, :]).ravel()
        vector = np.kron(vector, vectors[:, 0])
    return support, predicted, vector


def arithmetic_controls():
    """Replay the latest in-repository refutation, not its obsolete floor."""
    reports = []
    for primes, exponents in [([2, 3, 5, 7], [1] * 4),
                               ([2, 3, 5, 7, 11, 13, 17], [1] * 7),
                               ([2, 3, 5, 7, 11, 13], [2, 2, 2, 1, 1, 1])]:
        support, predicted, vector = divisor_box(primes, exponents)
        gcd = np.gcd(support[:, None], support[None, :])
        matrix = gcd / np.sqrt(support[:, None].astype(float) * support)
        actual = float(np.linalg.eigvalsh(matrix)[0])
        residual = float(np.linalg.norm(matrix @ vector - predicted * vector))
        require(abs(actual - predicted) < ABS_TOL, "Divisor-box eigenvalue mismatch")
        require(residual < ABS_TOL, "Divisor-box eigenvector mismatch")
        reports.append(dict(max_support=int(support.max()), size=len(support),
                            primes=primes, exponents=exponents,
                            tensor_eigenvalue=predicted, direct_eigenvalue=actual,
                            eigenvector_residual=residual))
    require(reports[2]["direct_eigenvalue"] < reports[1]["direct_eigenvalue"],
            "The recorded box should beat the largest primorial cube below 10^6")
    return reports


def run():
    rng = np.random.default_rng(SEED)
    # Contract: closed transform agrees with independent quadrature;
    # real clusters remain PSD; a resolved reflected pair has inertia (1,1);
    # replacing vv^T by vv* removes the negative direction; both small
    # channels scale quadratically; a positive background can hide a pair.
    max_quad_error = 0.0
    quadrature_cases = []
    for bandwidth in [0.4, 1.0, 3.0]:
        points = [0j, math.pi/bandwidth, -2*math.pi/bandwidth,
                  math.pi/bandwidth + 1e-10j]
        points += list(rng.uniform(-8, 8, 8) + 1j*rng.uniform(-0.375, 0.375, 8))
        for z in points:
            err = abs(window_transform(z, bandwidth) - transform_by_quadrature(z, bandwidth))
            max_quad_error = max(max_quad_error, float(err))
            quadrature_cases.append(dict(bandwidth=bandwidth, real=float(np.real(z)),
                                          imag=float(np.imag(z)), error=float(err)))
    require(max_quad_error < ABS_TOL, "Fourier-transform quadrature mismatch")

    center, halfspacing, bandwidth = 14.134725141734693, 1.2, 1.0
    probes = center + np.array([-halfspacing, halfspacing])
    displacements = np.geomspace(1e-3, 0.1, 13)
    rows = []
    for delta in displacements:
        real_cluster = online_form([center-delta, center+delta], probes, bandwidth)
        signed, psd = pair_forms(center, delta, probes, bandwidth)
        cluster_eigs = np.linalg.eigvalsh(real_cluster)
        pair_eigs = np.linalg.eigvalsh(signed)
        psd_eigs = np.linalg.eigvalsh(psd)
        # Independent two-probe exact formulas derived in the companion note.
        fp = float(window_transform(halfspacing+delta, bandwidth).real)
        fm = float(window_transform(halfspacing-delta, bandwidth).real)
        complex_value = window_transform(halfspacing+1j*delta, bandwidth)
        exact_cluster = (fp-fm)**2
        exact_pair = -4*float(complex_value.imag)**2
        require(cluster_eigs[0] >= -ABS_TOL, "An on-line cluster became indefinite")
        require(pair_eigs[0] < 0 < pair_eigs[1], "Reflected pair lost its resolved signature")
        require(psd_eigs[0] >= -ABS_TOL, "Hermitian outer-product control became indefinite")
        require(abs(cluster_eigs[0]-exact_cluster) < ABS_TOL, "Cluster closed form mismatch")
        require(abs(pair_eigs[0]-exact_pair) < ABS_TOL, "Signed closed form mismatch")
        rows.append(dict(displacement=float(delta),
                         cluster_min=float(cluster_eigs[0]),
                         signed_pair_min=float(pair_eigs[0]),
                         psd_control_min=float(psd_eigs[0]),
                         exact_cluster_min=exact_cluster, exact_signed_min=exact_pair))

    # Declare the small-displacement fit range before fitting; use exact
    # channel formulas to avoid subtracting nearly equal eigenvalues.
    fit = rows[:7]
    lx = np.log([r["displacement"] for r in fit])
    cluster_exponent = float(np.polyfit(lx, np.log([r["exact_cluster_min"] for r in fit]), 1)[0])
    signed_exponent = float(np.polyfit(lx, np.log([-r["exact_signed_min"] for r in fit]), 1)[0])
    require(abs(cluster_exponent-2) < 0.002 and abs(signed_exponent-2) < 0.002,
            "The derivative-channel quadratic prediction failed")

    # Generated, distinct geometries: avoid symmetric-only acceptance.
    generated = []
    for _ in range(160):
        b = float(rng.uniform(0.3, 3.0))
        gamma = float(rng.uniform(-200, 200))
        ts = gamma + rng.uniform(-3/b, 3/b, 2)
        delta = float(rng.uniform(0.005, 0.375))
        signed, psd = pair_forms(gamma, delta, ts, b)
        v = window_transform(gamma+1j*delta-ts, b)
        # det(2aa^T-2bb^T) = -4(a_0 b_1-a_1 b_0)^2.
        wedge = float(v.real[0]*v.imag[1]-v.real[1]*v.imag[0])
        expected_det = -4*wedge*wedge
        require(abs(np.linalg.det(signed)-expected_det) < ABS_TOL,
                "General pair determinant mismatch")
        require(np.linalg.eigvalsh(psd)[0] >= -ABS_TOL, "PSD control failed")
        real_zeros = gamma+rng.uniform(-2/b, 2/b, 5)
        require(np.linalg.eigvalsh(online_form(real_zeros, ts, b))[0] >= -ABS_TOL,
                "Generated real cluster failed positivity")
        generated.append(dict(bandwidth=b, center=gamma, displacement=delta,
                              probes=ts.tolist(), signed_determinant=float(np.linalg.det(signed)),
                              determinant_prediction=expected_det))

    signed, _ = pair_forms(center, 0.03, probes)
    background = online_form([center-0.5, center+0.5], probes)
    masked = signed+background
    require(np.linalg.eigvalsh(signed)[0] < 0 < np.linalg.eigvalsh(masked)[0],
            "The positive-background masking control did not exercise masking")
    margin = float(-np.linalg.eigvalsh(signed)[0])
    perturbation = 0.4*margin*np.eye(2)
    observed = float(np.linalg.eigvalsh(signed+perturbation)[0])
    require(observed < -np.linalg.norm(perturbation, 2),
            "The synthetic Weyl-margin control failed")

    # Exact integer spectra: identical first two moments do not decide inertia.
    positive_spectrum = np.array([1, 15, 29], dtype=np.int64)
    indefinite_spectrum = np.array([-1, 21, 25], dtype=np.int64)
    require(positive_spectrum.sum() == indefinite_spectrum.sum()
            and positive_spectrum @ positive_spectrum == indefinite_spectrum @ indefinite_spectrum,
            "The same-moments obstruction failed")

    return dict(
        scope="Finite planted-spectrum diagnostics only; not the full zeta Weil form or an RH test",
        seed=SEED, absolute_tolerance=ABS_TOL,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        acceptance=dict(status="PASS", quadrature_cases=len(quadrature_cases),
                        max_quadrature_error=max_quad_error,
                        generated_geometries=len(generated),
                        cluster_exponent=cluster_exponent, signed_exponent=signed_exponent),
        two_probe=dict(center=center, halfspacing=halfspacing, bandwidth=bandwidth, rows=rows),
        masking=dict(pair_min=float(np.linalg.eigvalsh(signed)[0]),
                     pair_plus_online_background_min=float(np.linalg.eigvalsh(masked)[0])),
        synthetic_perturbation=dict(observed_min=observed, supplied_operator_bound=float(np.linalg.norm(perturbation, 2))),
        arithmetic_controls=arithmetic_controls(),
        moment_obstruction=dict(positive_spectrum=positive_spectrum.tolist(),
                                indefinite_spectrum=indefinite_spectrum.tolist(),
                                trace=int(positive_spectrum.sum()),
                                hilbert_schmidt_squared=int(positive_spectrum @ positive_spectrum)),
        quadrature_cases=quadrature_cases, generated_geometries=generated,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a reproducible JSON record")
    args = parser.parse_args()
    report = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"scope": report["scope"], "acceptance": report["acceptance"],
                      "masking": report["masking"], "arithmetic_controls": report["arithmetic_controls"]}, indent=2))


if __name__ == "__main__":
    main()
