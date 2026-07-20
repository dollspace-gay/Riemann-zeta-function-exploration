# Literature Check — the Chains Result (June 2026)

Targeted web search (not exhaustive; no library access; some key sources
paywalled or in French). Purpose: calibrate the novelty claims before
outside contact.

## Clearly classical / known

- The NB criterion, Báez-Duarte's integer-dilation strengthening, Burnol's
  lower bound, and the BDBLS d_N² ~ C/log N conjecture: all standard.
- **Vasyunin's closed-form Gram entries** (St. Petersburg Math. J. 7,
  1996): explicit formulas for ⟨e_m, e_n⟩ via cotangent sums. Our closed
  form A(1,2) = (3/4)(log 2π − γ) − (log 2)/4 should be presented as
  *rederived* — **VERIFIED (July 2026)** as a special case: the formula
  (obtained verbatim from Darses–Hillion arXiv:2004.10086, p. 2, citing
  [Vas95] and [BDBLS00, p. 141]) at (m,n) = (2,1) gives exactly our
  closed form (cot(π/2) = 0 kills the sums). Implemented and
  cross-checked against our exact scheme to 1e−12–1e−15 on 26 pairs
  (`vasyunin_check.py`); the parity-disagreement integral P now has
  three independent computations agreeing (enumeration, digamma-period,
  Vasyunin combinations). Context worth knowing: Vasyunin sums connect
  to the Bettin–Conrey cotangent-sum reciprocity (arXiv:1303.2050
  circle), which is the deep structure behind these closed forms.
- Qualitative ill-conditioning of the NB system: folklore; Landreau–
  Richard (Exp. Math. 11, 2002) note the computational cost/difficulty of
  the Gram determinants. (French text; we could not verify whether they
  make spectral statements — flagged as the most likely place for
  overlap.)
- The Mellin-transform description ê_k = −k^{−s}ζ(s)/s and the resulting
  |ζ(½+it)|²-weighted reformulations: standard in this literature (e.g.
  Báez-Duarte's papers; the "weighted zeta-square measure" appears in
  recent work, arXiv:2209.10990).

## Adjacent recent work (checked, does not contain our statements)

- arXiv:2510.18132 (2025), "Beurling Nyman Geometry and Gram Matrix
  Structure...": polynomial decay envelopes for Gram entries of *smoothed*
  generalized families; no smallest-eigenvalue analysis, no N-scaling of
  the spectral floor, no chain structure (confirmed via abstract).
- arXiv:2405.06349 (2024), "On certain Gram matrices and their associated
  series": derives Gram formulas in the NB setting, studies reciprocity of
  series Σ R(nx); abstract makes no spectral claims. **RESOLVED (July
  2026, full text examined):** Ehm's paper concerns the *Mellin-weighted*
  kernels G^(q)_{u,v} (Báez-Duarte Dirichlet-polynomial side), not the
  raw dilation Gram matrix; zero content on eigenvalues, condition
  numbers, chains, parity integrals, or harmonic-number bounds (searched
  full text). No overlap with our claims. Bonus: his Theorem 2.1 (q = 1
  kernel, equivalent to BDBLS Prop. 90) is a published closed form
  anchoring our Mellin identity — cite it in the lower-bound program.
- Probabilistic NB generalizations (Darses–Hillion etc.): different
  questions.
- Nikolski's Hardy-multidisc program: completeness/cyclicity of dilation
  systems, not finite-N spectral floors.

## Not found anywhere in our search

1. The empirical law λ_min(A_N)·N² ≈ 0.50 + 0.134·log N, or any stated
   λ_min ≍ N⁻²·log N result for the classical (unsmoothed) NB Gram matrix.
2. The doubling-chain mechanism: square-wave identity used spectrally;
   near-null vectors supported on (k, 2k) pairs near k = N/2.
3. An explicit upper bound of the form λ_min ≤ (H_{K−1}+1)/(10·K(K−1)).
4. The chain-interaction kernel law (parity-disagreement integral
   asymptotics).

## Calibrated claim level for outside contact

"We have not found these statements in a targeted literature search; the
ingredients are classical, the experts may know them or they may follow
easily from known formulas (Vasyunin), and asking whether they are known
is part of the question." Do NOT claim novelty outright; DO claim the
machine-verified theorem and the measurements as solid regardless of
novelty status.

Sources consulted (June 9, 2026): arXiv 2510.18132, 2405.06349,
2209.10990, 2006.02953, 1805.06733, math/0607733, 1705.09921; Landreau–
Richard Exp. Math. 11 (2002) (abstract only); EUDML/Numdam records for
Vasyunin-related and Nikolski works.
