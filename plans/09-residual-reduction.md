# Plan 9: prove persistent reduction of the actual residual

Current research direction, September 24: [annihilator rigidity](../nyman-beurling/annihilator_rigidity.md).
The surviving remainder is characterized by residue equations, an explicit
projection constraint, and dilation autocorrelations. A rigidity theorem is
proved with the extra hypothesis Σ|F_n|<∞. Deriving sufficient regularity
from the canonical constraints, or finding a rigidity proof without it, is
open. This shifts the current proof attempt while retaining the gain target
below as another sufficient route. Neither route has proved RH.

The [decay follow-up](../nyman-beurling/autocorrelation_decay.md) proves that
canonical autocorrelation is automatic for an invariant-space projection.
Normalized real and complex controls satisfy it with nonsummable primitives;
their arithmetic defects are proportional to ζ(s), or real combinations of
that factor. A hypothetical off-line zero survives these tests. The controls
are not identified with the actual canonical remainder. The proof still needs
an arithmetic estimate that excludes the slow modes without assuming the
zero-free region it is intended to prove.

Status, September 10, 2026: the numerical instrument and general convergence
implication are implemented. The uniform arithmetic estimate remains open.
The [first direct proof attempt](../nyman-beurling/uniform_gain_attempt.md)
did not close it: it establishes a positive error floor for using chains alone,
screens explicit directions, and derives a Chebyshev-function connection for
the tapered trial function. That obstruction does not refute the gain target
for fully optimized residuals.
See [the proof note](../nyman-beurling/refinement_theory.md) for definitions,
derivations, verification, and the boundary of the Lean formalization.

The immediate target is an explicit correction in the new doubling chains
whose gain is at least c E_N/log N, for a fixed positive c at all sufficiently
large dyadic N. A weaker bound with nonsummable fractional gains also suffices.

| Deliverable | Evidence | Status |
|---|---|---|
| Exact full and chain refinement formulas | Block elimination, direct witness, subset solve, independent ambient QR | Implemented and checked |
| Tail availability and capture measured separately | Closed-form head energy, independent quadrature, gain ≤ tail check | Implemented through N=512 → 1024 |
| A control with positive spectral floor but nonzero limiting error | (1−2z) polynomial family; exact limit 3/4 | Reproduced |
| Persistent fractional gains imply error → 0 | `Refinement.tendsto_zero_of_divergent_gains` | Lean verified, gain hypotheses explicit |
| Unconditional real jump lower bound | `Jump.synthesis_integral_lower_bound`: actual synthesis integral ≥ Σc²/(36N²H_N) | Lean verified end to end |
| Uniform lower bound on residual correlations | Needs arithmetic cancellation, not another matrix eigenvalue bound | OPEN |

The September 23 [Möbius capture investigation](../nyman-beurling/residual_capture_theory.md)
now expresses these signed integrals on unit cells and identifies their
Mertens-sum weights. It extends finite checks through N=4096, proves the gain
improvement from removing old-span components, and finds that the current
tapered-trial transfer bound is too weak. No uniform bound is established.

The September 24 [noncancellation attempt](../nyman-beurling/mobius_correlation_identity.md)
proves a full-block correlation limit with N fixed and derives the separate
chain limit. It does not justify setting the summation cutoff proportional to
N, or control the cost of the resulting correction. An explicit counterexample
shows why fixed-residual correlation alone cannot force convergence. For the
full-block recipe, equation (5) there separates the two missing uniform bounds.


For the direct-gain route, the mathematical task remains to bound the resulting
signed sum against the actual optimal residual. One explicit sufficient target
is equation (8) of the
[residual-capture investigation](../nyman-beurling/residual_capture_theory.md):
a finite correlation minus its tail bound must have square at
least (c/log N) times E_N times the projected correction cost. More generally,
seek an explicit coefficient vector v_N satisfying

    (q_Nᵀv_N)² ≥ (c/log N) E_N (v_Nᵀ M_N v_N).

Before promoting a candidate, test whether it fails through insufficient
residual energy on u ≥ floor(N/2)+1 or insufficient capture of that energy.
Do not demand the conjectured optimal error constant; mere convergence is enough.

The new prime–zero bundle is verified locally and supplies a separate target:
pointwise nonnegativity of Suzuki's Ψ for every real argument. Neither the
finite positive kernel samples nor the Nyman–Beurling coefficient bound prove
that target. A relation between the two forms is an additional theorem to prove.

Acceptance for a claimed RH advance: the gain bound must be uniform, derived
without RH-equivalent cancellation assumptions, and connected to the actual
residual. Finite experiments and conditional Lean theorems must stay labeled.

September 23 update: the [shared-conversation review](../papers/gram_extremum_review.md)
adds a sufficient global-extremum theorem and a reproduced interval certificate
for a finite prime–zero Gram family. It also certifies a counterexample to an
unrestricted minimum claim. No identity connects its mode distributions to
the signed residual correlations above; the uniform-gain target remains open.
