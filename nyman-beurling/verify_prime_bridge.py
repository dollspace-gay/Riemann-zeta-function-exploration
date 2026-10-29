"""Check the exact Chebyshev identities in uniform_gain_attempt.md.

Floating-point checks of written identities; no RH or positivity proof.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from block_refinement import residual_head_energy
from nb_gram_np import mobius


def von_mangoldt(limit):
    prime = np.ones(limit+1, dtype=bool)
    prime[:2] = False
    for p in range(2, int(np.sqrt(limit))+1):
        if prime[p]:
            prime[p*p::p] = False
    values = np.zeros(limit+1, dtype=np.longdouble)
    for p in np.flatnonzero(prime):
        p = int(p)
        power = p
        while power <= limit:
            values[power] = np.log(np.longdouble(p))
            power *= p
    return values


def run():
    limit = int(np.ceil(np.exp(12)))
    mangoldt = von_mangoldt(limit)
    chebyshev = np.cumsum(mangoldt)
    mu = mobius(1024).astype(np.longdouble)
    trials = []
    for n in (16, 32, 64, 128, 256, 512, 1024):
        k = np.arange(1, n+1, dtype=np.longdouble)
        log_n = np.log(np.longdouble(n))
        c = -mu[:n]*np.log(n/k)/log_n
        theta = np.sum(mu[:n]*np.log(n/k)/k)
        u = np.r_[np.arange(1, n+1, dtype=np.longdouble),
                  np.arange(1, n, dtype=np.longdouble)+np.longdouble('.375')]
        direct = 1-np.mod(u[:, None]/k, 1)@c
        predicted = (theta*u-chebyshev[np.floor(u).astype(int)])/log_n
        max_error = float(np.max(np.abs(direct-predicted)))
        np.testing.assert_allclose(direct, predicted, atol=1e-11, rtol=0)
        # Independent prime-sieve evaluation of the trial head integral.
        j = np.arange(1, n, dtype=np.longdouble)
        psi = chebyshev[1:n]
        prime_head = np.sum(theta**2 - 2*theta*psi*np.log1p(1/j)
                            + psi**2/(j*(j+1)))/log_n**2
        synthesis_head = residual_head_energy(c, n)-(theta/log_n)**2
        np.testing.assert_allclose(prime_head, synthesis_head, atol=1e-11, rtol=0)
        trials.append({"N": n, "theta_N": float(theta),
                       "maximum_pointwise_identity_error": max_error,
                       "head_energy_identity_error": float(abs(prime_head-synthesis_head))})
    smoothing = []
    for time in (.25, .5, 1., 2., 4., 8., 12.):
        t = np.longdouble(time)
        end = np.exp(t)
        indices = np.arange(1, int(np.floor(end))+1)
        left = indices.astype(np.longdouble)
        right = np.minimum(left+1, end)
        weight = lambda x: (t-np.log(x))/np.sqrt(x)
        prime_sum = np.sum(mangoldt[indices]*weight(left))
        by_parts = np.sum(chebyshev[indices]*(weight(left)-weight(right)))
        np.testing.assert_allclose(prime_sum, by_parts, atol=1e-11, rtol=0)
        smoothing.append({"t": time, "prime_sum": float(prime_sum),
                          "integration_by_parts_error": float(abs(prime_sum-by_parts))})
    return {"status": "Written identities numerically checked; no uniform bound or RH proof",
            "trial_residual_identity": trials, "suzuki_prime_sum_identity": smoothing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    result["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("verify_prime_bridge.py", "block_refinement.py", "nb_gram_np.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
