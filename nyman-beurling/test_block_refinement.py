"""Independent geometric and analytic checks of the refinement calculation."""

import unittest

import numpy as np

from block_refinement import (chain_refinement, polynomial_control, project,
                              refinement, residual_head_energy)
from nb_gram_np import b_vector, build_gram


class RefinementTests(unittest.TestCase):
    def test_block_identity_against_ambient_qr(self):
        rng = np.random.default_rng(881)
        vectors = rng.normal(size=(29, 18))
        target = rng.normal(size=29)
        gram = vectors.T @ vectors
        rhs = vectors.T @ target
        norm_sq = float(target @ target)
        result = refinement(gram, rhs, 9, norm_sq)
        old_q, _ = np.linalg.qr(vectors[:, :9], mode="reduced")
        full_q, _ = np.linalg.qr(vectors, mode="reduced")
        old_error = np.linalg.norm(target - old_q @ (old_q.T @ target))**2
        full_error = np.linalg.norm(target - full_q @ (full_q.T @ target))**2
        self.assertAlmostEqual(result["gain"], old_error - full_error, places=11)

    def test_polynomial_convergent_control(self):
        for n in (4, 16, 64):
            gram, rhs, exact = polynomial_control(n, 1.0)
            self.assertAlmostEqual(project(gram, rhs).error, exact, places=13)

    def test_positive_spectral_floor_does_not_force_convergence(self):
        gram, rhs, exact = polynomial_control(64, 2.0)
        self.assertGreaterEqual(np.linalg.eigvalsh(gram)[0], 1.0)
        self.assertAlmostEqual(project(gram, rhs).error, 0.75, places=13)
        self.assertAlmostEqual(exact, 0.75, places=13)
        early = refinement(gram, rhs, 4)
        late = refinement(gram, rhs, 16)
        self.assertGreater(early["gain"], 0)
        self.assertLess(late["gain"], early["gain"] * 1e-6)

    def test_chain_projection_and_direct_witness_for_odd_and_even_sizes(self):
        gram = build_gram(30, verbose=False)
        target = b_vector(30)
        for n in (7, 8, 15):
            row = chain_refinement(gram, target, n)
            self.assertEqual(row["chain_count"], (n+1)//2)
            self.assertGreater(row["direct_chain_gain"], 0)
            self.assertLessEqual(row["direct_chain_gain"], row["projected_chain_gain"] + 1e-12)
            self.assertLessEqual(row["projected_chain_share"], 1 + 1e-10)
            self.assertLess(row["chain_target_identity_error"], 1e-14)
            self.assertLessEqual(row["direct_chain_tail_capture"], 1 + 1e-10)
            self.assertAlmostEqual(row["direct_chain_eta_logN"],
                row["tail_fraction_times_logN"] * row["direct_chain_tail_capture"], places=12)

    def test_head_integral_against_piecewise_quadrature(self):
        from scipy.integrate import quad
        c = np.array([-0.8, 0.7, 0.4, -0.1])
        k = np.arange(1, len(c)+1)
        def integrand(u):
            if u == 0:
                return float(np.sum(c/k)**2)
            residual = float(u >= 1) - np.sum(c*np.mod(u/k, 1))
            return float((residual/u)**2)
        numerical = sum(quad(integrand, j, j+1, epsabs=1e-12)[0] for j in range(9))
        self.assertAlmostEqual(residual_head_energy(c, 9), numerical, places=12)

    def test_invalid_inputs_fail_instead_of_clipping(self):
        with self.assertRaises(ValueError):
            project(np.eye(3), np.zeros(2))
        with self.assertRaises(ValueError):
            project(np.eye(3), np.ones(3))
        with self.assertRaises(ValueError):
            refinement(np.eye(3), np.zeros(3), 2)
        with self.assertRaises(ValueError):
            project(np.array([[1., np.nan], [np.nan, 1.]]), np.zeros(2))
        with self.assertRaises(np.linalg.LinAlgError):
            project(np.ones((2, 2)), np.zeros(2))


if __name__ == "__main__":
    unittest.main()
