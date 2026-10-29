"""Check exact autocorrelation controls with a known, nonzero arithmetic defect.

These controls are not the remainder for the full integer-dilation span.
All numerical outputs are diagnostics, not interval certificates or RH proofs.
See autocorrelation_decay.md for the written proofs and conditional statements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import mpmath as mp
import numpy as np
from scipy.integrate import quad


def control(s, real_pair=False):
    """A normalized complex Mellin witness, or a real conjugate-pair projection.

    Components (z, a) mean a*t**(z-1) on (0,1]. The tail is B/t.
    Use a surrounding mp.workdps context when more precision is needed.
    """
    s = mp.mpc(s)
    if not mp.mpf('.5') < s.real < 1:
        raise ValueError('Require 1/2 < Re(s) < 1')
    K = 1/(2*s.real-1) + 1/abs(1-s)**2
    if real_pair:
        if s.imag == 0:
            raise ValueError('The real pair requires Im(s) != 0')
        J = 1/(2*s-1) + 1/(1-s)**2
        gram = mp.matrix([[(K+J.real)/2, J.imag/2],
                          [J.imag/2, (K-J.real)/2]])
        target = mp.matrix([(1/s).real, (1/s).imag])
        coefficients = mp.lu_solve(gram, target)
        A = (coefficients[0]-1j*coefficients[1])/2
        components = [(s, A), (mp.conj(s), mp.conj(A))]
        energy = (target.T*coefficients)[0]
    else:
        A = 1/(mp.conj(s)*K)
        components = [(s, A)]
        energy = 1/(abs(s)**2*K)
    B = mp.fsum(a/(1-z) for z, a in components)
    return {'s': s, 'components': components, 'B': B,
            'energy': energy, 'real_pair': real_pair}


def primitive(model, x):
    if x < 1:
        raise ValueError('Require x >= 1')
    return mp.fsum(a/z*mp.power(x, -z) for z, a in model['components'])


def arithmetic_defect(model, k):
    return -mp.fsum(a/z*mp.zeta(z)*mp.power(k, -z)
                   for z, a in model['components'])


def log_kernel(model, x):
    """Predicted actual autocorrelation; complex convention linear in first slot."""
    if x < 0:
        return mp.conj(log_kernel(model, -x))
    return mp.exp(x/2)*primitive(model, mp.exp(x))


def quadrature_checks(model, scales=(1, 2, 3, np.pi)):
    """Independent integration in log t, including all infinite intervals."""
    components = [(complex(z), complex(a)) for z, a in model['components']]
    B = complex(model['B'])

    def profile(y):
        if y <= 0:
            return sum(a*np.exp((z-.5)*y) for z, a in components)
        return B*np.exp(-y/2)

    def integrate(f, breaks):
        def part(left, right):
            re = quad(lambda y: f(y).real, left, right,
                      epsabs=2e-13, epsrel=2e-12, limit=600)[0]
            im = quad(lambda y: f(y).imag, left, right,
                      epsabs=2e-13, epsrel=2e-12, limit=600)[0]
            return re+1j*im
        return sum(part(left, right) for left, right in zip(breaks, breaks[1:])
                   if left < right)

    norm = integrate(lambda y: complex(abs(profile(y))**2), [-np.inf, 0, np.inf])
    target = integrate(lambda y: profile(y)*np.exp(y/2), [-np.inf, 0])
    rows = []
    for scale in scales:
        shift = np.log(scale)
        covariance = integrate(lambda y: profile(y)*np.conj(profile(y+shift)),
                               [-np.inf, -shift, 0, np.inf])/np.sqrt(scale)
        direct_primitive = integrate(lambda y: profile(y)*np.exp(y/2),
                                     [-np.inf, -shift])
        predicted = complex(primitive(model, scale))
        rows.append({'scale': float(scale),
                     'covariance_error': abs(covariance-predicted),
                     'primitive_error': abs(direct_primitive-predicted)})
    return {'norm_error': abs(norm-float(model['energy'])),
            'target_error': abs(target-float(model['energy'])), 'scales': rows}


def truncated_pairing(model, k, stop=2048):
    """Integrate h*e_k from 1/(k*(stop+1)) to infinity cell by cell.

    The omitted integral has the stated analytic absolute bound. The evaluated
    values and bound are floating point, so this is not an enclosure routine.
    """
    if k < 1 or stop < 1:
        raise ValueError('Require positive k and stop')
    value = model['B']/k
    bound = mp.mpf(0)
    for z, a in model['components']:
        cells = mp.fsum(
            (mp.power(j, 1-z)-mp.power(j+1, 1-z))/(z-1)
            -j*(mp.power(j, -z)-mp.power(j+1, -z))/z
            for j in range(1, stop+1))
        value += a*(mp.power(k, -z)*cells
                    +(1-mp.power(k, 1-z))/(k*(z-1)))
        bound += abs(a)/z.real*mp.power(k*(stop+1), -z.real)
    return value, bound


def residue_checks(model, moduli=(2, 3, 5)):
    rows = []
    for n in moduli:
        grouped = mp.mpc(0)
        formula = mp.mpc(0)
        for z, a in model['components']:
            grouped += a*mp.power(n, -z)/z*mp.fsum(
                r*(mp.zeta(z, mp.mpf(r)/n)-mp.zeta(z, mp.mpf(r+1)/n))
                for r in range(1, n))
            formula += a/z*mp.zeta(z)*(1-mp.power(n, 1-z))
        rows.append({'modulus': n, 'abs_defect': float(abs(formula)),
                     'hurwitz_discrepancy': float(abs(grouped-formula))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with mp.workdps(40):
        # This height comes from a critical-line zero; the point at sigma=.75
        # is a separate nonzero-zeta control, never an alleged off-line zero.
        height = mp.zetazero(1).imag
        for s in [mp.mpc('.75', '1.3'), mp.mpc('.75', height)]:
            for real_pair in [False, True]:
                model = control(s, real_pair)
                pairings = []
                for k in [1, 2, 5]:
                    partial, bound = truncated_pairing(model, k)
                    exact = arithmetic_defect(model, k)
                    pairings.append({'k': k, 'abs_full_defect': float(abs(exact)),
                                     'abs_truncation_error': float(abs(partial-exact)),
                                     'tail_bound': float(bound)})
                points = np.array([0., .17, .9, 1.8, 3.1])
                kernel = np.array([[complex(log_kernel(model, x-y)) for y in points]
                                   for x in points])
                rows.append({'sigma': float(s.real), 'height': str(s.imag),
                             'real_pair': real_pair, 'energy': float(model['energy']),
                             'abs_zeta': str(abs(mp.zeta(s))),
                             'quadrature': quadrature_checks(model),
                             'pairings': pairings, 'residues': residue_checks(model),
                             'sample_kernel_min_eigenvalue': float(np.linalg.eigvalsh(kernel)[0])})
    result = {'status': 'Controls only; zeta defects are nonzero. No RH proof or decay bound.',
              'mpmath_decimal_precision': 40, 'checks': rows,
              'python': platform.python_version(), 'mpmath': mp.__version__,
              'numpy': np.__version__,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Checked {len(rows)} autocorrelation controls; saved {args.output}')


if __name__ == '__main__':
    main()
