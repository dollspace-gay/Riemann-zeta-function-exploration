"""Finite checks of reciprocal-cell annihilator constraints.

Reports concern finite optimal residuals, not a computed infinite remainder.
Power-mode tests distinguish exact written identities from numerical zeta data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import mpmath as mp
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from nb_gram_np import b_vector


def cell_coordinates(c, stop):
    """Cell integrals d_j and primitives F_j for chi - sum c_k e_k.

    Index zero is unused. F_j means integral over (0,1/j). No claim about
    coefficient convergence or decay of the infinite remainder is made.
    """
    if stop < 1:
        raise ValueError('stop must be positive')
    c = np.asarray(c, dtype=np.longdouble)
    j = np.arange(1, stop+1, dtype=np.longdouble)
    slope = np.sum(c / np.arange(1, len(c)+1, dtype=np.longdouble))
    jumps = np.zeros(stop+1, dtype=np.longdouble)
    for k, value in enumerate(c, 1):
        jumps[k::k] += value
    intercepts = 1 + np.cumsum(jumps)[1:]
    log = np.log1p(1/j)
    weight = 1/(j*(j+1))
    d = np.r_[np.longdouble(0), intercepts*weight-slope*log]
    primitive_one = 1 - c @ b_vector(len(c)).astype(np.longdouble)
    primitive = np.r_[np.longdouble(0), primitive_one, primitive_one-np.cumsum(d[1:])]
    energy = ((intercepts*log-slope)**2 + intercepts**2*(weight-log**2))
    return {'d': d, 'F': primitive, 'cell_energy': energy,
            'slope': slope, 'means': d[1:]/weight}


def power_mode(s, moduli=(2, 3, 5), bands=(128, 1024, 8192)):
    """Closed formula vs independently grouped Hurwitz-zeta residue sums."""
    s = mp.mpc(s)
    residues = []
    for n in moduli:
        grouped = mp.power(n, -s)/s * mp.fsum(
            r*(mp.zeta(s, mp.mpf(r)/n)-mp.zeta(s, mp.mpf(r+1)/n))
            for r in range(1, n))
        formula = mp.zeta(s)*(1-mp.power(n, 1-s))/s
        residues.append({'modulus': n, 'abs_residue': float(abs(formula)),
                         'formula_discrepancy': float(abs(grouped-formula))})
    energy = []
    for start in bands:
        mass = mp.fsum(j*(j+1)*abs((mp.power(j,-s)-mp.power(j+1,-s))/s)**2
                       for j in range(start, 2*start))
        energy.append({'from': start, 'to_exclusive': 2*start, 'weighted_energy': float(mass)})
    return {'sigma': float(s.real), 'height': str(s.imag),
            'abs_zeta': str(abs(mp.zeta(s))), 'residues': residues, 'bands': energy}


def investigate(gram, sizes):
    limit = max(sizes)
    if min(sizes) < 2 or limit > len(gram):
        raise ValueError('Require 2 <= N <= matrix size')
    factor = cholesky(gram[:limit, :limit], lower=True)
    b = b_vector(limit)
    coordinates = solve_triangular(factor, b, lower=True)
    all_targets = b_vector(len(gram))
    rows = []
    for n in sizes:
        c = solve_triangular(factor[:n,:n].T, coordinates[:n], lower=False)
        error = float(1-coordinates[:n] @ coordinates[:n])
        stop = 64*n
        cells = cell_coordinates(c, stop)
        tail_energy = float(error-cells['slope']**2-cells['cell_energy'].sum())
        if tail_energy <= 0:
            raise ArithmeticError('Nonpositive tail energy')
        normal = b[:n] - gram[:n,:n] @ c
        if abs(float(cells['F'][1])-error) > 1e-10:
            raise ArithmeticError('Residual target pairing differs from squared norm')
        constraints = []
        indices = np.arange(1, stop+1)
        for modulus in [2,3,5,7,11,13]:
            if modulus > n:
                continue
            partial = float((indices % modulus) @ cells['d'][1:])
            full = float(modulus*normal[modulus-1]-normal[0])
            bound = (modulus-1)*np.sqrt(tail_energy/(stop+1))
            if abs(full-partial) > bound+1e-10:
                raise ArithmeticError('Residue tail exceeds its Cauchy--Schwarz bound')
            abel = (cells['F'][1:stop+1].sum()
                    -modulus*cells['F'][modulus:stop+1:modulus].sum()
                    -(stop % modulus)*cells['F'][stop+1])
            if abs(float(abel)-partial) > 1e-10:
                raise ArithmeticError('Abel boundary identity failed')
            constraints.append({'modulus': modulus, 'full_pairing': full,
                                'partial_cell_sum': partial, 'tail_bound': float(bound),
                                'abel_discrepancy': abs(float(abel)-partial)})
        covariances = []
        for scale in [2,3,5]:
            if scale*n > len(gram):
                continue
            scaled = scale*np.arange(1,n+1)-1
            cross = float(c @ (all_targets[scaled]-c @ gram[:n,scaled]))
            primitive = float(cells['F'][scale])
            covariance = primitive-cross
            if abs(covariance) > error/np.sqrt(scale)+1e-10:
                raise ArithmeticError('Dilation covariance violates Cauchy--Schwarz')
            covariances.append({'scale':scale, 'primitive':primitive,
                                'residual_covariance':covariance,
                                'finite_projection_defect':cross,
                                'covariance_bound':float(error/np.sqrt(scale))})
        rows.append({'N':n, 'stop':stop, 'energy':error,
                     'primitive_at_one':float(cells['F'][1]),
                     'slope':float(cells['slope']), 'constraints':constraints,
                     'dilation_covariances':covariances,
                     'primitive_abs_sum_through_stop':float(abs(cells['F'][1:stop+1]).sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gram', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrix = np.load(args.gram, allow_pickle=False)
    rows = investigate(matrix, [32,128,512,1024,4096])
    with mp.workdps(40):
        first = mp.zetazero(1)
        modes = [power_mode(first), power_mode(mp.mpc('0.75', first.imag))]
    with args.gram.open('rb') as stream:
        gram_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {'status':'Finite checks only; no infinite remainder or RH proof computed',
              'finite_residuals':rows, 'power_modes':modes,
              'python':platform.python_version(), 'numpy':np.__version__, 'mpmath':mp.__version__,
              'gram_sha256':gram_hash,
              'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ['annihilator_probe.py','nb_gram_np.py']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Checked 5 finite residuals, 30 residue constraints, and 2 power modes')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
