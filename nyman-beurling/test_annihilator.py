"""Independent arithmetic, quadrature, and boundary checks for annihilators."""
from fractions import Fraction
from math import gcd, lcm
import unittest

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.special import polygamma

from annihilator_probe import cell_coordinates, power_mode
from nb_gram_np import mobius, b_vector, build_gram


class AnnihilatorTests(unittest.TestCase):
    def test_residue_identity_including_cell_endpoints(self):
        for n in range(1,17):
            for j in range(0,33):
                for frac in [Fraction(0),Fraction(1,3),Fraction(7,8)]:
                    u = j+frac
                    fract = lambda x: x-x.numerator//x.denominator
                    self.assertEqual(fract(u/n)-fract(u)/n, Fraction(j % n,n))

    def test_cell_integrals_against_direct_quadrature(self):
        c=np.array([-.8,.7,.4,-.1])
        cells=cell_coordinates(c,30)
        indices=np.arange(1,len(c)+1)
        for j in range(1,31):
            direct=quad(lambda t: 1-c @ np.mod(1/(indices*t),1),1/(j+1),1/j,
                        epsabs=1e-13)[0]
            self.assertAlmostEqual(float(cells['d'][j]),direct,places=13)

    def test_abel_identity_retains_boundary(self):
        F=[Fraction(0)]+[Fraction(1,j*j+1) for j in range(1,52)]
        for n in [2,3,5,11]:
            for stop in [1,7,19,50]:
                direct=sum((j % n)*(F[j]-F[j+1]) for j in range(1,stop+1))
                abel=sum(F[1:stop+1])-n*sum(F[n:stop+1:n])-(stop % n)*F[stop+1]
                self.assertEqual(direct,abel)

    def test_finite_prime_sieve_identity(self):
        F=[Fraction(0)]+[Fraction((-1)**j,j*j+1) for j in range(1,101)]
        mu=mobius(210)
        for primorial in [2,6,30,210]:
            for n in [1,2,5]:
                inverted=sum(int(mu[d-1])*sum(F[n*d::n*d])
                             for d in range(1,primorial+1) if primorial % d==0)
                surviving=sum(F[n*m] for m in range(1,100//n+1) if gcd(m,primorial)==1)
                self.assertEqual(inverted,surviving)

    def test_adjoint_decimates_primitive(self):
        F=np.r_[0.,1/np.arange(1,102,dtype=float)**2]
        d=F[1:-1]-F[2:]
        for n in [2,3,5]:
            for k in range(1,10):
                grouped=d[n*k-1:n*(k+1)-1].sum()
                self.assertAlmostEqual(grouped,F[n*k]-F[n*(k+1)],places=15)

    def test_dilation_covariance_against_periodic_quadrature(self):
        n, scale=3,2
        gram=build_gram(n*scale,verbose=False)
        b=b_vector(n*scale)
        c=np.linalg.solve(gram[:n,:n],b[:n])
        cells=cell_coordinates(c,10)
        scaled=scale*np.arange(1,n+1)-1
        cross=c @ (b[scaled]-c @ gram[:n,scaled])
        expected=float(cells['F'][scale])-cross
        k=np.arange(1,n+1)
        period=lcm(*range(1,n*scale+1))
        def product(u):
            r=float(u>=1)-c @ np.mod(u/k,1)
            dilated=float(u>=scale)-c @ np.mod(u/(scale*k),1)
            return r*dilated
        head=sum(quad(lambda u:product(u)/u**2,j,j+1,epsabs=1e-13)[0]
                 for j in range(scale))
        tail=sum(quad(lambda u:product(u)*polygamma(1,u/period)/period**2,
                      j,j+1,epsabs=1e-13)[0] for j in range(scale,scale+period))
        self.assertAlmostEqual(expected,head+tail,places=11)

    def test_power_mode_formula_and_critical_energy(self):
        with mp.workdps(40):
            mode=power_mode(mp.mpc('.5','14.134725141734693790'),bands=(1024,8192))
            self.assertLess(max(row['formula_discrepancy'] for row in mode['residues']),1e-35)
            # A critical-line power has a nonzero limiting energy in each
            # dyadic band, so it must not be accepted as a weighted-L2 vector.
            self.assertAlmostEqual(mode['bands'][-1]['weighted_energy'],float(mp.log(2)),places=6)
            off=power_mode(mp.mpc('.75','14.134725141734693790'),bands=(1024,8192))
            self.assertGreater(min(row['abs_residue'] for row in off['residues']),.02)


if __name__=='__main__':
    unittest.main()
