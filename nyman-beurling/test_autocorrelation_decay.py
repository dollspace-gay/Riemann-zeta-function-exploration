"""Independent quadrature and arithmetic checks of the decay obstruction."""
import unittest

import mpmath as mp
import numpy as np

from autocorrelation_decay import (arithmetic_defect, control, log_kernel,
                                   quadrature_checks, residue_checks, truncated_pairing)


class AutocorrelationDecayTests(unittest.TestCase):
    def check_quadrature(self, real_pair):
        with mp.workdps(40):
            for s in [mp.mpc('.75', '1.3'), mp.mpc('.75', '14.13472514173469')]:
                model = control(s, real_pair)
                self.assertGreater(float(model['energy']), 0)
                errors = quadrature_checks(model)
                self.assertLess(errors['norm_error'], 3e-11)
                self.assertLess(errors['target_error'], 3e-11)
                for row in errors['scales']:
                    self.assertLess(row['covariance_error'], 3e-11)
                    self.assertLess(row['primitive_error'], 3e-11)

    def test_complex_self_correlation_by_direct_quadrature(self):
        self.check_quadrature(False)

    def test_real_self_correlation_by_direct_quadrature(self):
        self.check_quadrature(True)

    def test_arithmetic_defect_against_cells_with_retained_tail(self):
        with mp.workdps(40):
            for real_pair in [False, True]:
                model = control(mp.mpc('.75', '1.3'), real_pair)
                for k in [1, 2, 5]:
                    partial, bound = truncated_pairing(model, k, stop=256)
                    exact = arithmetic_defect(model, k)
                    self.assertLess(abs(partial-exact), bound)
                    # The point is not a zeta zero; the arithmetic equations
                    # really fail, by more than the omitted-cell bound.
                    self.assertGreater(abs(exact), 2*bound)

    def test_residue_defect_by_independent_hurwitz_grouping(self):
        with mp.workdps(40):
            for real_pair in [False, True]:
                model = control(mp.mpc('.75', '14.13472514173469'), real_pair)
                rows = residue_checks(model)
                self.assertLess(max(r['hurwitz_discrepancy'] for r in rows), 1e-35)
                self.assertGreater(max(r['abs_defect'] for r in rows), 1e-5)

    def test_hermitian_positive_kernel_at_irregular_points(self):
        with mp.workdps(40):
            points = [0., .11, .39, 1.7, 4.2, 9.1]
            for real_pair in [False, True]:
                model = control(mp.mpc('.75', '1.3'), real_pair)
                kernel = np.array([[complex(log_kernel(model, x-y)) for y in points]
                                   for x in points])
                np.testing.assert_allclose(kernel, kernel.conj().T, atol=1e-14, rtol=0)
                self.assertGreater(np.linalg.eigvalsh(kernel)[0], 0)

    def test_domain_rejects_non_l2_and_degenerate_pair(self):
        for s in [mp.mpc('.5', 14), mp.mpc('.4', 14), mp.mpc(1, 14)]:
            with self.assertRaises(ValueError):
                control(s)
        with self.assertRaises(ValueError):
            control(mp.mpf('.75'), real_pair=True)


if __name__ == '__main__':
    unittest.main()
