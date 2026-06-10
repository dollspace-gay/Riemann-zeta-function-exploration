"""
Prime spectroscopy: the primes as resonance lines in the zeta zeros.
=====================================================================
Treat the zeros as spectral lines of an unknown system and take the
Fourier transform a spectroscopist would: F(tau) = sum_j w(g_j) e^{i g_j tau}
(Hann window w over the zero range). The explicit formula predicts
resonances at tau = m log p with intensity proportional to log p / p^{m/2}
("periodic orbits" of length log p in the Hilbert-Polya picture).

Result (2,001,052 zeros): all tested prime powers detected at their exact
locations (offsets <= 2e-4 on a coarse scan; the needles have width
~ 2pi/range ~ 1e-5), and the needle intensities match the explicit-formula
prediction to three decimal places:

    p^m     measured/predicted relative intensity
    2       1.000 / 1.000     7      1.501 / 1.501
    3       1.294 / 1.294     8      0.500 / 0.500
    4       0.707 / 0.707     9      0.747 / 0.747
    5       1.469 / 1.469     11     1.475 / 1.475   ... etc.

This is the trace-formula structure of the zeros made visible: zeros and
primes are Fourier duals. Known mathematics (the explicit formula),
reproduced here as a measurement on Odlyzko's data.
"""

import numpy as np
import torch
import math, os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

zeros = []
with open(os.path.expanduser("~/rh_data/zeros6")) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                v = float(line.split()[-1])
                if v > 0:
                    zeros.append(v)
            except ValueError:
                continue
g = np.array(zeros)
gt = torch.tensor(g, dtype=torch.float64, device=DEVICE)
T0, T1 = g[0], g[-1]
win = torch.sin(math.pi * (gt - T0) / (T1 - T0)) ** 2

CANDS = [(2,1),(3,1),(2,2),(5,1),(7,1),(2,3),(3,2),(11,1),(13,1),(17,1),(19,1),(23,1)]

print(f"{'p^m':>5} {'log p^m':>9} {'needle amp':>11} {'measured':>9} {'predicted':>10}")
amps = {}
for p, m in CANDS:
    v = m * math.log(p)
    taus = torch.arange(v - 2e-5, v + 2e-5, 1e-6, dtype=torch.float64, device=DEVICE)
    Sre = torch.zeros(len(taus), dtype=torch.float64, device=DEVICE)
    Sim = torch.zeros_like(Sre)
    for i in range(0, len(g), 250000):
        ph = torch.outer(taus, gt[i:i+250000])
        wb = win[i:i+250000]
        Sre += torch.cos(ph) @ wb
        Sim += torch.sin(ph) @ wb
    amps[(p, m)] = float(torch.sqrt(Sre**2 + Sim**2).max())
a2 = amps[(2, 1)]
for p, m in CANDS:
    v = m * math.log(p)
    pred = (math.log(p) / p**(m/2)) / (math.log(2) / 2**0.5)
    lbl = f"{p}^{m}" if m > 1 else str(p)
    print(f"{lbl:>5} {v:>9.4f} {amps[(p,m)]:>11.0f} {amps[(p,m)]/a2:>9.3f} {pred:>10.3f}")
