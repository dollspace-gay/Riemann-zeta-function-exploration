"""Independent checks for the signed-correlation investigation."""
from math import lcm
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import polygamma

from mobius_correlation_probe import investigate
from nb_gram_np import b_vector, build_gram, mobius


class MobiusCorrelationTests(unittest.TestCase):
    def test_projected_correlation_against_periodic_integrals(self):
        n, m = 3, 6
        gram = build_gram(m, verbose=False)
        b = b_vector(m)
        c = np.linalg.solve(gram[:n, :n], b[:n])
        mu = mobius(m).astype(float)
        g = np.r_[-np.linalg.solve(gram[:n, :n], gram[:n, n:] @ mu[n:]), mu[n:]]
        period = lcm(*range(1, m+1))
        indices = np.arange(1, m+1)
        a = c @ (1 / indices[:n])
        g_slope = g @ (1 / indices)

        def integrand(u, cost):
            residual = 1 - c @ np.mod(u/indices[:n], 1)
            direction = g @ np.mod(u/indices, 1)
            weight = polygamma(1, u/period) / period**2
            return (direction**2 if cost else residual*direction) * weight

        correlation = -a*g_slope + sum(quad(lambda u: integrand(u, False), j, j+1,
                                                epsabs=1e-13)[0] for j in range(1, period+1))
        cost = g_slope**2 + sum(quad(lambda u: integrand(u, True), j, j+1,
                                          epsabs=1e-13)[0] for j in range(1, period+1))
        row = investigate(gram, [n])['rows'][0]['blocks'][0]
        self.assertAlmostEqual(row['correlation'], correlation, places=11)
        self.assertAlmostEqual(row['projected_cost'], cost, places=11)

    def test_chain_limit_formula_against_target_integrals(self):
        n = 6
        gram = build_gram(2*n, verbose=False)
        b = b_vector(n)
        c = np.linalg.solve(gram[:n, :n], b)
        # Integrate r_N against chi(2t). In u-coordinates this is u >= 2.
        period = lcm(*range(1, n+1))
        indices = np.arange(1, n+1)
        def integrand(u):
            return (1-c @ np.mod(u/indices, 1)) * polygamma(1, u/period) / period**2
        pairing = sum(quad(integrand, j, j+1, epsabs=1e-13)[0]
                      for j in range(2, period+2))
        row = investigate(gram, [n])['rows'][0]
        self.assertAlmostEqual(row['fixed_residual_chain_limit'],
                               row['energy']/2-pairing, places=11)

    def test_fixed_residual_capture_does_not_force_convergence(self):
        # Orthonormal coordinates: one permanently invisible component, M
        # visible components, and an aggregated geometric tail after M.
        for n, m in [(2, 4), (2, 8), (4, 8), (4, 16)]:
            target = np.r_[1., 2.**-np.arange(1, m+1), np.sqrt(4.**-m/3)]
            residual = target.copy()
            residual[1:n+1] = 0
            correction = np.zeros(m+2)
            correction[n+1:m+1] = -target[n+1:m+1]
            correction[m] -= 2.**m
            mass = (4.**-n - 4.**-m)/3
            self.assertAlmostEqual(residual@residual, 1+4.**-n/3)
            self.assertAlmostEqual(residual@correction, -(1+mass))
            self.assertAlmostEqual(correction@correction, 4.**m+2+mass)
            gain = (residual@correction)**2 / (correction@correction)
            self.assertLessEqual(gain, 16/(9*4.**m))


if __name__ == '__main__':
    unittest.main()
