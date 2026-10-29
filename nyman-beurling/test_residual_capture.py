"""Independent integral, arithmetic, and projection checks for residual capture."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import mpmath as mp
import numpy as np
from scipy.integrate import quad

from gain_attempt import run as legacy_screen
from nb_gram_np import build_gram as integral_gram, mobius
from nb_vasyunin import build_gram
from residual_capture import cell_arrays, direction_stats, investigate


class ResidualCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which('g++') is None:
            raise unittest.SkipTest('C++ compiler required for the Vasyunin cross-check')
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.library = Path(cls.directory.name) / 'gram.so'
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC',
                        str(Path(__file__).with_name('nb_vasyunin.cpp')), '-o', str(cls.library)], check=True)
        cls.gram = build_gram(64, cls.library)

    def test_vasyunin_against_independent_integral_builder(self):
        np.testing.assert_allclose(self.gram[:24, :24], integral_gram(24, verbose=False),
                                   atol=8e-12, rtol=0)

    def test_vasyunin_against_high_precision_cotangent_sums(self):
        with mp.workdps(65):
            def v(a, b):
                return mp.fsum(mp.mpf((k*a) % b) / b * mp.cot(mp.pi * k / b)
                               for k in range(1, b))
            for m, n in [(1, 1), (1, 2), (2, 5), (6, 15), (17, 31)]:
                divisor = int(np.gcd(m, n))
                a, b = m // divisor, n // divisor
                exact = ((a+b) * (mp.log(2*mp.pi)-mp.euler)
                         - (a-b)*mp.log(mp.mpf(a)/b) - mp.pi*(v(a,b)+v(b,a))) / (2*a*b*divisor)
                self.assertLess(abs(mp.mpf(self.gram[m-1, n-1])-exact), mp.mpf('3e-16'))

    def test_cells_against_direct_piecewise_quadrature(self):
        c = np.array([-.8, .7, .4, -.1])
        k = np.arange(1, len(c)+1)
        parents = np.array([3, 4])
        weights = np.array([-1., 1.])
        cells = cell_arrays(c, parents, weights, 20)
        def values(u):
            r = float(u >= 1) - np.sum(c * np.mod(u/k, 1))
            h = np.sum(weights * (np.mod(u/(2*parents), 1) - .5*np.mod(u/parents, 1)))
            return r, h
        for name, operation in [('energy', lambda r,h:r*r), ('cost', lambda r,h:h*h),
                                ('correlation', lambda r,h:r*h)]:
            direct = [quad(lambda u: operation(*values(u))/u**2, j, j+1, epsabs=1e-13)[0]
                      for j in range(20)]
            np.testing.assert_allclose(cells[name], direct, atol=2e-14, rtol=1e-11)

    def test_mertens_shell_identity(self):
        n = 30
        mu = mobius(n)
        mertens = np.r_[0, np.cumsum(mu)]
        parents = np.arange(n//2+1, n+1)
        cells = cell_arrays(np.zeros(n), parents, mu[parents-1], 4*n)
        for j in range(4*n):
            def shell_divisor_sum(x):
                return sum(mertens[min(n, x//q)] - mertens[min(n//2, x//q)]
                           for q in range(1, x//parents[0]+1))
            predicted = shell_divisor_sum(j)/2 - shell_divisor_sum(j//2)
            self.assertEqual(cells['correction_levels'][j], predicted)

    def test_cleanup_gain_against_ambient_least_squares(self):
        rng = np.random.default_rng(881)
        old = rng.normal(size=(29, 9))
        target = rng.normal(size=29)
        h = rng.normal(size=29)
        q, _ = np.linalg.qr(old, mode='reduced')
        r = target - q @ (q.T @ target)
        old_error = r @ r
        combined = np.column_stack([old, h])
        fitted = combined @ np.linalg.lstsq(combined, target, rcond=None)[0]
        full_gain = old_error - np.sum((target-fitted)**2)
        result = direction_stats(float(r@h), float(h@h), q.T@h, old_error, full_gain, 9)
        self.assertAlmostEqual(result['projected_gain'], full_gain, places=12)
        self.assertLessEqual(result['raw_gain'], result['projected_gain'])

    def test_prefix_solve_reproduces_old_screen(self):
        old = legacy_screen(self.gram, 16, 32)
        new = investigate(self.gram, [16, 24, 32], [32], multiplier=8)
        for row in new['rows']:
            previous = old['rows'][row['N']-16]
            self.assertAlmostEqual(row['error_N'], previous['error_N'], places=12)
            self.assertAlmostEqual(row['directions']['mobius_chain']['raw_eta_logN'],
                                   previous['directions']['mobius']['eta_logN'], places=9)


if __name__ == '__main__':
    unittest.main()
