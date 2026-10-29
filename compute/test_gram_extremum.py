"""Failure controls for the computer-assisted finite Gram certificate."""
import unittest

import mpmath as mp

from certify_gram_extremum import bounds, certify, root_checks, spectral_enclosures


class CertificateControls(unittest.TestCase):
    def setUp(self):
        mp.mp.dps = 100
        mp.iv.dps = 65

    def test_rejects_bracket_without_zero(self):
        with self.assertRaisesRegex(ArithmeticError, "sign change"):
            root_checks([["14", "14.01"]])

    def test_rejects_singular_gram_matrix(self):
        with self.assertRaisesRegex(ArithmeticError, "Positive definiteness"):
            spectral_enclosures(mp.iv.matrix([[1, 1], [1, 1]]))

    def test_rejects_actual_two_zero_five_prime_control(self):
        enclosures = []
        for index in [1, 2]:
            zero = mp.im(mp.zetazero(index))
            enclosures.append([mp.nstr(zero - mp.mpf("1e-20"), 80),
                               mp.nstr(zero + mp.mpf("1e-20"), 80)])
        with self.assertRaisesRegex(ArithmeticError, "hinge inequalities"):
            certify(enclosures, prime_count=5)

    def test_serialized_intervals_contain_original_intervals(self):
        for endpoints in [("-1e100", "2e100"), ("-1e-100", "-2e-101"),
                          ("1.23456789", "1.23456789000000001"), ("0", "0")]:
            interval = mp.iv.mpf(endpoints)
            lo, hi = bounds(interval)
            self.assertTrue(mp.iv.mpf(lo).b <= interval.a)
            self.assertTrue(mp.iv.mpf(hi).a >= interval.b)


if __name__ == "__main__":
    unittest.main()
