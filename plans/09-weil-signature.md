# Plan 9 — Cluster-adapted Weil compressions

October 28, 2026. Companion: [the synthesis and execution record](../papers/rh_synthesis_2026_10.md).

## Goal

Use the project's cluster-resolution tools to study signed compressions
of Weil's form, incorporating the recent proportion theorem's operator
and the quasi-RH strip as a separately named error-bound input.
The mathematical target is a controlled spectral statement beyond
the trace/second-moment certificate.

## Gates

| Gate | Observable deliverable | Status |
|---|---|---|
| 1. Preserve signature | Exact pair formulas and an admissible test window; varied planted controls; arithmetic refutation replay | **Done** in `compute/weil_signature.py` |
| 2. Arithmetic reconstruction | Prime powers, pole terms, and gamma factor assemble a low-height compression independently of its validation zero sum | **Done numerically** in `compute/weil_prime_side.py` |
| 3. Certify numerical error | Interval enclosure for convolution and gamma integrals, analytic infinite-tail bound, and roundoff budget | **Open**; cutoff agreement is insufficient |
| 4. Cluster-aware probes | Adapted divided-difference probes tested against on-line clusters, reflected pairs, positive-background masking, and ill-conditioned basis changes | **Open**; choosing probes from known zeros must remain disclosed |
| 5. Additional spectral control | An arithmetic-side estimate beyond the first two moments, with explicit dependence on dimension, height, support, and strip input | **Open research target** |
| 6. Quantifiers | Show exactly which test spaces/heights are controlled; identify what is required to extend to all admissible functions | **Open**; a sequence of positive finite matrices is not enough by itself |

## The first new work item

Upgrade the reference arithmetic compression to an interval-certified
matrix with a total operator-norm enclosure. Begin at the existing
two-probe example, where a prime-side/zero-side check is available.
Separate convolution quadrature, gamma integral truncation, numerical
roundoff, and (when comparing with a zero sum) the zero-sum tail.
They are different error sources.

For a chosen published quasi-RH input the synthesis derives a
zero-sum-tail envelope with an `exp(-B/4)` (7/8 input) or `exp(-B/6)`
(11/12 input) improvement over the classical-strip envelope, holding
the rest fixed. That is not an interval estimate for the gamma
integral. Do not substitute one for the other.

Acceptance of an alleged negative direction requires
`lambda_min(W_hat) < -eta` with a proved operator error `eta`, and a
statement of whether `W` is the full form on the selected probes or
only a truncated zero contribution. The current scripts claim neither.

## What remains a separate program

- Nyman–Beurling distance decay needs coefficient/projection control.
  A small or large Gram eigenvalue alone does not establish it.
- The surviving GCD question is the post-Session-27 arithmetic one.
  Neither the false polylogarithmic floor nor the old K-floor is a premise.
- Trial-distance spectroscopy retains its RH/simple-zero assumptions
  and its explicitly unproved contour/convergence steps.

No external contact is part of this plan.
