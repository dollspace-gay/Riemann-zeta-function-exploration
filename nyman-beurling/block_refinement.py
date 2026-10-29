"""Finite Nyman--Beurling refinement identities, with explicit numerical checks.

All energies are squared L2 errors. Floating-point output is diagnostic, not
an interval certificate or a bound uniform in N. No matrix inverse is formed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh, solve_triangular


@dataclass(frozen=True)
class Projection:
    coefficients: np.ndarray
    error: float


def validate_problem(gram, target, target_norm_squared=1.0):
    gram = np.asarray(gram, dtype=float)
    target = np.asarray(target, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram matrix must be square")
    if target.shape != (len(gram),) or not len(gram):
        raise ValueError("Target correlations must match a nonempty Gram matrix")
    if not np.isfinite(gram).all() or not np.isfinite(target).all():
        raise ValueError("Inputs must be finite")
    if not np.isfinite(target_norm_squared) or target_norm_squared <= 0:
        raise ValueError("Target squared norm must be positive and finite")
    if not np.allclose(gram, gram.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Gram matrix must be symmetric")
    return gram, target


def project(gram, target, target_norm_squared=1.0):
    """Project the target onto the given family; reject an invalid L2 problem."""
    gram, target = validate_problem(gram, target, target_norm_squared)
    factor = cholesky(gram, lower=True)
    coordinates = solve_triangular(factor, target, lower=True)
    coefficients = solve_triangular(factor.T, coordinates, lower=False)
    error = float(target_norm_squared - coordinates @ coordinates)
    if error <= 0:
        raise ValueError("Nonpositive residual energy; exact fit or numerical failure")
    return Projection(coefficients, error)


def quadratic_gain(gram, correlations):
    """Maximum decrease from fitting corrections with this Gram matrix."""
    coefficients = cho_solve(cho_factor(gram, lower=True), correlations)
    gain = float(correlations @ coefficients)
    if gain < 0 or not np.isfinite(gain):
        raise ValueError("Invalid correction energy")
    return gain, coefficients


def residual_head_energy(coefficients, stop):
    """Integrate the optimal residual exactly in form on 0 < u < integer stop.

    On each unit interval R(u) = 1 + sum c_k floor(u/k) - u sum c_k/k.
    Long-double evaluation reduces cancellation in the elementary integral;
    coefficients are still floating-point inputs, so this is not an enclosure.
    """
    if not isinstance(stop, (int, np.integer)) or stop < 1:
        raise ValueError("Head endpoint must be a positive integer")
    c = np.asarray(coefficients, dtype=np.longdouble)
    k = np.arange(1, len(c)+1, dtype=np.longdouble)
    slope = np.sum(c / k)
    increments = np.zeros(stop, dtype=np.longdouble)
    for index, coefficient in enumerate(c, 1):
        increments[index::index] += coefficient
    levels = 1 + np.cumsum(increments)[1:]
    n = np.arange(1, stop, dtype=np.longdouble)
    pieces = (levels**2 / (n*(n+1))
              - 2*levels*slope*np.log1p(1/n) + slope**2)
    if np.any(pieces < -1e-12):
        raise ValueError("Negative interval energy beyond roundoff tolerance")
    return float(slope**2 + pieces.sum())


def refinement(gram, target, n, target_norm_squared=1.0):
    """Exact block-projection identity, evaluated numerically at N -> 2N."""
    gram, target = validate_problem(gram, target, target_norm_squared)
    if not isinstance(n, (int, np.integer)) or n < 2 or 2 * n > len(gram):
        raise ValueError("Require integer N >= 2 and a Gram matrix of size >= 2N")
    old = gram[:n, :n]
    cross = gram[:n, n:2*n]
    new = gram[n:2*n, n:2*n]
    old_projection = project(old, target[:n], target_norm_squared)
    full_projection = project(gram[:2*n, :2*n], target[:2*n], target_norm_squared)
    solved_cross = cho_solve(cho_factor(old, lower=True), cross)
    schur = new - cross.T @ solved_cross
    # Removing roundoff antisymmetry does not alter the input Gram matrix.
    schur = (schur + schur.T) / 2
    residual_correlations = target[n:2*n] - cross.T @ old_projection.coefficients
    gain, new_coefficients = quadratic_gain(schur, residual_correlations)
    reconstructed = np.concatenate((
        old_projection.coefficients - solved_cross @ new_coefficients,
        new_coefficients,
    ))
    np.testing.assert_allclose(
        gain, old_projection.error - full_projection.error, rtol=1e-7, atol=5e-11,
        err_msg="Schur gain disagrees with independent full projection",
    )
    np.testing.assert_allclose(
        reconstructed, full_projection.coefficients, rtol=1e-7, atol=5e-9,
        err_msg="Block elimination does not recover the full coefficients",
    )
    eigenvalues, eigenvectors = eigh(schur)
    if eigenvalues[0] <= 0:
        raise ValueError("Nonpositive Schur eigenvalue")
    mode_energies = (eigenvectors.T @ residual_correlations)**2 / eigenvalues
    np.testing.assert_allclose(mode_energies.sum(), gain, rtol=1e-7, atol=5e-11)
    return {
        "old_projection": old_projection,
        "full_projection": full_projection,
        "schur": schur,
        "correlations": residual_correlations,
        "gain": gain,
        "mode_energies": mode_energies,
        "schur_eigenvalues": eigenvalues,
    }


def chain_refinement(gram, target, n):
    """Compare all new dilations, projected doubling chains, and direct chains.

    Parents are floor(N/2)+1,...,N, so every child 2k is new. For odd N
    this family has ceil(N/2) members. Direct corrections hold the old
    approximant fixed; projected corrections also refit its coefficients.
    """
    result = refinement(gram, target, n)
    old_projection = result["old_projection"]
    parents = np.arange(n // 2 + 1, n + 1)
    p = parents - 1
    q = 2 * parents - 1
    new_positions = q - n
    direct = (gram[np.ix_(q, q)] - gram[np.ix_(q, p)] / 2
              - gram[np.ix_(p, q)] / 2 + gram[np.ix_(p, p)] / 4)
    cross = gram[:n, q] - gram[:n, p] / 2
    chain_target = target[q] - target[p] / 2
    correlations = chain_target - cross.T @ old_projection.coefficients
    projected = direct - cross.T @ cho_solve(
        cho_factor(gram[:n, :n], lower=True), cross)
    projected = (projected + projected.T) / 2
    np.testing.assert_allclose(
        projected, result["schur"][np.ix_(new_positions, new_positions)],
        rtol=1e-7, atol=5e-12,
    )
    np.testing.assert_allclose(correlations, result["correlations"][new_positions],
                               rtol=1e-7, atol=5e-12)
    direct_gain, alpha = quadratic_gain(direct, correlations)
    projected_gain, _ = quadratic_gain(projected, correlations)
    # Independent check: use the direct correction as an actual 2N-vector.
    corrected = np.zeros(2*n)
    corrected[:n] = old_projection.coefficients
    corrected[q] += alpha
    corrected[p] -= alpha / 2
    corrected_error = float(1 - 2 * target[:2*n] @ corrected
                            + corrected @ gram[:2*n, :2*n] @ corrected)
    np.testing.assert_allclose(corrected_error, old_projection.error - direct_gain,
                               rtol=1e-7, atol=5e-11)
    # Independent subset solve: old functions together with the new evens.
    selected = np.concatenate((np.arange(n), q))
    subset_error = project(gram[np.ix_(selected, selected)], target[selected]).error
    np.testing.assert_allclose(projected_gain, old_projection.error - subset_error,
                               rtol=1e-7, atol=5e-11)
    if not (-5e-11 <= direct_gain <= projected_gain + 5e-11
            and projected_gain <= result["gain"] + 5e-11):
        raise ValueError("Correction-space energy ordering failed")
    error = old_projection.error
    log_n = float(np.log(n))
    tail = error - residual_head_energy(old_projection.coefficients, int(parents[0]))
    if tail <= 0 or direct_gain > tail + 5e-11:
        raise ValueError("Direct-chain gain exceeds the residual energy on its support")
    return {
        "N": int(n),
        "chain_count": len(parents),
        "error_N": error,
        "error_2N": result["full_projection"].error,
        "full_gain": result["gain"],
        "projected_chain_gain": projected_gain,
        "direct_chain_gain": direct_gain,
        "full_eta_logN": result["gain"] / error * log_n,
        "projected_chain_eta_logN": projected_gain / error * log_n,
        "direct_chain_eta_logN": direct_gain / error * log_n,
        "projected_chain_share": projected_gain / result["gain"],
        "direct_chain_share": direct_gain / result["gain"],
        "chain_support_starts_at_u": int(parents[0]),
        "residual_tail_energy_on_chain_support": tail,
        "tail_fraction_times_logN": tail / error * log_n,
        "direct_chain_tail_capture": direct_gain / tail,
        "bottom_tenth_schur_mode_share": float(
            result["mode_energies"][:max(1, n // 10)].sum() / result["gain"]),
        "schur_lambda_min": float(result["schur_eigenvalues"][0]),
        "chain_target_identity_error": float(np.max(np.abs(
            chain_target - np.log(2) / (2 * parents)))),
        "old_normal_equation_residual": float(np.linalg.norm(
            gram[:n, :n] @ old_projection.coefficients - target[:n])),
    }


def polynomial_control(size, a):
    """H2 Gram problem for (1-a*z)*z^j, j=0,...,size-1, target 1.

    For a=1 the exact squared error is 1/(size+1). For a>1 it tends
    to 1-a^-2 despite a uniform spectral lower bound (a-1)^2.
    """
    if size < 1 or a < 1:
        raise ValueError("Require size >= 1 and a >= 1")
    gram = np.diag(np.full(size, 1 + a*a))
    gram += np.diag(np.full(size - 1, -a), 1)
    gram += np.diag(np.full(size - 1, -a), -1)
    target = np.zeros(size)
    target[0] = 1
    exact_error = 1 / (size + 1) if a == 1 else (a*a - 1) / (a*a - a**(-2*size))
    return gram, target, exact_error
