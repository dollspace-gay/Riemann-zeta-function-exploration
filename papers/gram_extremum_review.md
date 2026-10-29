# Finite Gram extrema and geometry: verification review

Reviewed September 23, 2026. Sources: the supplied mathematical report `REPORT.md`, `certificate.json`, and supporting mathematical notes.

**Result:** the sufficient extremum theorem is correct, and a new implementation reproduces the interval proof for the supplied 20 ordinate brackets and first 120 primes. The universal minimum assertion is false. These results concern the finite prime–zero Gram construction; they do not establish the uniform Nyman–Beurling residual-gain estimate or RH.

## 1. What was actually checked

| Claim | Review status |
|---|---|
| Raw Gram matrix has a positive-semidefinite minimum at the center | Written proof checked below |
| Integrated-tail dominance implies a strict global condition-number minimum | Written proof checked below; sufficient, not necessary |
| Supplied 20-bracket, 120-prime family satisfies that dominance | Reconstructed interval verification passed for every ordinate choice in the brackets |
| These brackets contain critical-line zeros | All 20 Hardy Z sign changes independently verified with an explicit remainder |
| They are exactly the first 20 zeros, with none omitted | Indexed by `mpmath.zetazero`; no independent argument-principle zero count performed |
| Every zero count / prime cutoff has a minimum at the center | False; a two-zero/five-prime counterexample was interval verified |
| Centered 20-point geometry has rank 19 and effective linear dimension about 18 | Reproduced numerically; does not measure a manifold's intrinsic dimension |
| Completed-xi vertical squared energy is strictly convex | Written proof checked; quadrature corroborates it |
| These extrema imply RH or persistent residual gain | No such implication established |

The supplied material included an earlier numerical verifier, but the final certificate-generating program was not available. The report and JSON supply the description and results. We reconstructed the algorithm from the report instead of trusting its `PASS` field. This is an independent implementation using the **same mpmath interval library**, not an independent arithmetic backend or a Lean formalization.

Raw mathematical transcript, supplied files, recovered earlier verifier, provenance hashes, and generated JSON live under the already ignored `Riemann_Continuation_Verification/shared-conversation-20260923/`. The scripts and this reviewed note are the maintained research artifacts.

## 2. Precise finite matrix and the automatic inequality

Let \(t=\sigma-1/2\), \(\ell_p=\log p\), and \(a_p=\ell_p/\sqrt p\). For a finite prime set and real ordinates define

\[
 (C_p)_{ij}=\cos((\gamma_i-\gamma_j)\ell_p),\qquad
 G(t)=\sum_p a_p\cosh(t\ell_p)C_p.
\]

This is the **real cosine matrix** computed in `compute/rh_gpu.py`, not the different complex exponential Gram matrix. Since

\[
 C_p=c_pc_p^T+s_ps_p^T\succeq0,
\]

with \((c_p)_i=\cos(\gamma_i\ell_p)\) and \((s_p)_i=\sin(\gamma_i\ell_p)\), termwise comparison gives

\[
 \cosh(|t|\ell_{\min})G(0)\preceq G(t)
 \preceq\cosh(|t|\ell_{\max})G(0).
\]

If \(G(0)\succ0\), every ordered eigenvalue is strictly larger away from zero. Also

\[
 G(t)-G(0)\succeq \tfrac12 t^2\ell_{\min}^2G(0),\qquad G''(t)\succeq0.
\]

None of these inequalities orders the ratio of the extreme eigenvalues. Positive scalar normalization preserves condition number, but generally destroys the raw matrix ordering: a positive-semidefinite difference between matrices of equal trace must vanish.

## 3. A sufficient theorem for the condition number

Assume \(G(0)\succ0\). Choose unit extreme eigenvectors \(u_\pm\), with eigenvalues \(\lambda_\pm\), and set

\[
 \mu_\pm(p)=\frac{a_pu_\pm^TC_pu_\pm}{\lambda_\pm},\qquad
 d_p=\mu_+(p)-\mu_-(p),\qquad
 H(u)=\sum_p d_p(\ell_p-u)_+.
\]

Both \(\mu\)'s are probability distributions, so \(\sum d_p=0\).

**Theorem.** If \(H\ge0\) on \([0,\infty)\), then \(\kappa(G(t))\ge\kappa(G(0))\) for every real \(t\). If \(H\) is positive somewhere, the inequality is strict for every \(t\ne0\).

**Proof.** Put \(f_\pm(t)=\sum_p\mu_\pm(p)\cosh(t\ell_p)>0\). Rayleigh bounds using the fixed center vectors give

\[
 \kappa(G(t))\ge\frac{\lambda_+ f_+(t)}{\lambda_- f_-(t)}.
\]

The elementary identity

\[
 \cosh(tx)=1+t^2\int_0^x(x-u)\cosh(tu)\,du
\]

and equality of the masses imply

\[
 f_+(t)-f_-(t)
 =t^2\int_0^{\ell_{\max}}\cosh(tu)H(u)\,du.
\]

Nonnegativity gives the weak conclusion. If the continuous function \(H\) is positive somewhere, it is positive on a nonempty interval, making this integral positive for \(t\ne0\). This proves the strict conclusion. No eigenvector continuation or simple-eigenvalue hypothesis is needed for this theorem. ∎

The hypothesis is increasing convex order for the two distributions of \(\ell_p\). The proof above establishes the implication directly, without needing a stochastic-order citation. It is stronger than merely comparing mean frequencies or second moments.

For increasing primes \(p_1,\ldots,p_P\), \(H\) is affine between successive logarithms, constant on \([0,\log p_1]\), and zero past \(\log p_P\). Thus it suffices to check the \(P-1\) values

\[
 H(\log p_j)=\sum_{k>j}d_{p_k}\log(p_k/p_j).
\]

For simple extreme eigenvalues, the familiar local coefficient is

\[
 \log\kappa(G(t))=\log\kappa(G(0))+rt^2+O(t^4),\qquad
 r=\tfrac12\sum_pd_p\ell_p^2.
\]

A positive local coefficient alone does not give the global theorem.

## 4. Reconstructed interval certificate

The new [interval verifier](../compute/certify_gram_extremum.py) uses only the supplied JSON's 20 bracket endpoints. It recomputes the signs, matrix, spectral enclosures, and all hinge inequalities. Its steps are:

1. Evaluate zeta by Euler–Maclaurin with \(N=128,m=32\), exact rational Bernoulli numbers, and the explicit remainder bound below. Combine it with interval log-gamma to verify opposite Hardy Z signs at each bracket's endpoints.
2. Enclose the real Gram entries for **every** choice of ordinates in the brackets.
3. Use NumPy only to propose a basis. Interpret its decimal entries as exact rationals and invert that basis by exact rational Gaussian elimination. Interval multiplication encloses the similar matrix. Disjoint Gershgorin intervals identify all 20 positive eigenvalues.
4. For each normalized extreme candidate, enclose its residual \(\epsilon\), separation \(\delta\) from the other eigenvalues, and vector error \(2\epsilon/\delta\). This follows by orthogonal spectral decomposition and choice of the true eigenvector's sign. Enclose the two mode distributions and every hinge value.

For \(s=1/2+it\), the remainder bound is

\[
 |R|\le\frac{4|(s)_{2m}|}{(2\pi)^{2m}}
 \frac{N^{1/2-2m}}{2m-1/2}.
\]

It follows from the periodic Bernoulli bound \(2\zeta(2m)<4\) and integrating \(x^{-1/2-2m}\). The Euler–Maclaurin representation and functional equation are documented by [NIST DLMF §25.2](https://dlmf.nist.gov/25.2) and [§25.4](https://dlmf.nist.gov/25.4). The sign checks prove existence in each bracket; they do not count every intervening zero.

**Observed execution:** all 20 root brackets, all 20 positive eigenvalue enclosures, and all 119 strict hinge inequalities pass at 65 decimal interval digits. The reconstructed minimum lower bound exceeds \(8.32217\times10^{-5}\). The supplied certificate's slightly different enclosures need not match bit for bit because the candidate basis and enclosure procedure can differ.

An entrywise high-precision calculation, independent of the interval spectral pipeline, was run at 45 and 75 digits:

\[
 \kappa(G(0))\approx2.54932054632960978259570352506,
 \qquad \min_jH(\ell_j)\approx0.00008322177968873579384076407187.
\]

The two precision runs differ by less than \(1.5\times10^{-44}\) in all hinge values and \(6.1\times10^{-44}\) in the tabulated condition numbers. Numerical agreement corroborates the interval result; it is not itself certification.

A further 90-digit audit places all 20 numerical zeros and 40 endpoint Z values inside the supplied bounds, all 20 eigenvalues and 119 hinges inside both certificates, and both counterexample condition numbers inside the supplied intervals. Generating fresh zero brackets without the supplied JSON also passes the interval verifier.

| Zeros | Primes | \(\kappa(0.5)\) | \(\kappa(0.6)\) | \(\kappa(0.9)\) |
|---:|---:|---:|---:|---:|
| 2 | 5 | 1.176113076249 | 1.171131628612 | 1.106441535104 |
| 20 | 30 | 6.162125255510 | 6.151503781714 | 7.095660804247 |
| 20 | 120 | 2.549320546330 | 2.679129630429 | 4.049775989301 |

For the two-zero/five-prime case, the eigenvalues are \(A\pm|B|\); interval bounds prove that the value at 0.9 is smaller than at 0.5. The 20/30 case is additional high-precision counterevidence. The original minimum claim therefore needs an explicit cutoff regime or a hypothesis such as the theorem above.

**Trust boundary:** Python, exact integer/rational arithmetic, and mpmath's experimental interval elementary functions/log-gamma are in the computational trust base. NumPy accuracy is not assumed: unsuitable candidates cause certification to fail. We do not claim an Arb reproduction, a Lean proof, an independent zero count, or priority over the literature.

## 5. Geometry: what survives the visual discussion

For fixed positive weights, the feature map is

\[
 \Phi(\gamma)=\bigl(\sqrt{w_p}\cos(\gamma\log p),
                         \sqrt{w_p}\sin(\gamma\log p)\bigr)_p.
\]

It lies on a product of circles with intrinsic metric \(\sum_p w_p\,d\theta_p^2\), a flat finite torus. The continuously parametrized flow \(\theta_p=\gamma\log p\) is a geodesic. Unique factorization makes the logarithms of distinct primes rationally independent, so this continuous flow is dense in the finite torus. This says nothing by itself about density of the **discrete zero samples**.

The Gram-derived distance is the **extrinsic chordal** distance

\[
 \|\Phi(\gamma_i)-\Phi(\gamma_j)\|^2
 =4\sum_p w_p\sin^2((\gamma_i-\gamma_j)\log p/2),
\]

not the torus's geodesic distance. Spirals from a plotting coordinate choice and nearest-neighbor graph embeddings do not establish intrinsic curvature or a physical force.

For the default 20/120 sample, centering the feature vectors gives:

| Diagnostic | Reproduced value |
|---|---:|
| Numerical centered rank | 19 |
| Top three eigenvalues / total centered variance | 0.2197103302 |
| Top four eigenvalues / total centered variance | 0.2854452816 |
| Participation ratio \((\sum\lambda)^2/\sum\lambda^2\) | 17.95836597 |
| Entropy effective rank | 18.46425804 |

These are **linear sample dimensions**, not the dimension of an underlying manifold. The conversation's approximately 18 corresponds to the participation ratio. Its graph-embedding stress and small GUE/Poisson ensemble comparisons remain exploratory; their full experiment was not reconstructed here.

An infinite-dimensional extension needs a topology and convergence analysis. Radii \(p^{-\beta}\) are square summable only for \(\beta>1/2\); at the proposed central shell \(\sum_p1/p\) diverges. The present raw Gram diagonal \(\sum_p\log p/\sqrt p\) also diverges. Reflection symmetry does not repair either divergence or force zero locations.

## 6. A separate, unconditional completed-xi extremum

Let \(\xi(s)=\tfrac12s(s-1)\pi^{-s/2}\Gamma(s/2)\zeta(s)\), continued to an entire function, and

\[
 \mathcal E(\sigma)=\int_{\mathbb R}|\xi(\sigma+iy)|^2\,dy.
\]

The conversation's strict-convexity argument is correct. For a holomorphic function, \(\partial_\sigma^2|f|^2+\partial_y^2|f|^2=4|f'|^2\). For xi, vertical decay and derivative decay, uniformly on compact real strips, justify differentiating under the integral and discarding the boundary term. These follow from [Stirling's expansion](https://dlmf.nist.gov/5.11), polynomial vertical-strip bounds for zeta, and Cauchy estimates. Hence

\[
 \mathcal E''(\sigma)=4\int_{\mathbb R}|\xi'(\sigma+iy)|^2\,dy>0.
\]

Strictness follows because a nonconstant entire function cannot have its derivative vanish on an entire vertical line. The functional equation gives \(\mathcal E(1-\sigma)=\mathcal E(\sigma)\), so the center is the unique global minimum. This argument does not assume RH.

Truncated quadrature reproduces \(\mathcal E(1/2)\approx2.009059886861309\), \(\mathcal E(1/4)\approx2.015241816253029\), and \(\mathcal E''(1/2)\approx0.197531515155140\). A second difference agrees to about \(1.2\times10^{-7}\); extending the cutoff from 60 to 80 makes a negligible numerical change. These quadrature checks are not certified infinite-integral error bounds.

This extremum does **not** locate zeros. For \(z=s-1/2\), the entire function

\[
 F(s)=e^{z^2}\big((z-1/4)^2+1\big)\big((z+1/4)^2+1\big)
\]

has the same reflection symmetry and rapid vertical decay. Its squared vertical energy is strictly convex by the same argument, but its zeros are \(1/4\pm i\) and \(3/4\pm i\). Thus symmetry plus this energy minimum is insufficient for RH.

## 7. External context and the remaining research target

- The [2021 trapped-ion paper](https://www.nature.com/articles/s41534-021-00446-7) does report resolving the first 80 zeros by a drive constructed from zeta. This supports an engineered analog computation, not discovery of a natural Hamiltonian whose complete spectrum proves RH. A Floquet spectrum is defined modulo the drive frequency; identifying it with unbounded ordinates requires additional structure. Time-quasicrystal and gravity suggestions remain proposals.
- The conversation's August 2026 external-result reference is real: [Anthropic's announcement](https://www.anthropic.com/research/riemann-zeta) reports improving the critical-line proportion bound from 41.6% to about 67.2%. Its [technical note](https://www-cdn.anthropic.com/23455459f8832d06bb175cc0f88d019aed962ef8.pdf) states the constant \(3/2-\cot(1/\sqrt2)/\sqrt2\) and uses the indefinite Weil form. We checked the source and stated theorem, not its full proof or external Lean build. This is literature context, not a theorem established by this repository.
- In the explicit-formula heuristic, an individual zero contributes a factor \(e^{(\beta-1/2)y}\) after square-root normalization. That does not make the **summed** normalized prime error bounded under RH. Convergence conventions, smoothing, and accumulation of contributions matter.

Our finite Gram kernel is positive by construction and takes real ordinates as inputs. It does not recover their real parts, nor is it identified with the full Weil quadratic functional. Likewise, no identity here controls the signed correlations of the **fully optimized Nyman–Beurling residual**.

The active RH target remains [persistent residual reduction](../plans/09-residual-reduction.md), for example \(G_N\ge cE_N/\log N\) uniformly along dyadic scales. The new finite theorem supplies a mechanism worth testing in a specified scaling regime; finite positivity, a finite global extremum, and the completed-xi energy theorem do not supply that uniform estimate.

## 8. Reproduction

From the repository root, with NumPy and mpmath installed:

```bash
python compute/certify_gram_extremum.py \
  --certificate Riemann_Continuation_Verification/shared-conversation-20260923/certificate_supplied.json \
  --output Riemann_Continuation_Verification/shared-conversation-20260923/reconstructed_certificate.json
python compute/verify_gram_extremum.py --xi \
  --output Riemann_Continuation_Verification/shared-conversation-20260923/independent_checks.json
python -m unittest discover -s compute -p test_gram_extremum.py -v
```

Omit `--certificate` to generate fresh brackets rather than using the private input file. Outputs remain ignored. Four failure-control tests check rejection of a root-free bracket, a singular matrix, and the actual two-zero/five-prime dominance counterexample, plus outward serialization of interval endpoints. No Lean source was changed for these additions, and these theorems are not attributed to the existing Lean build.
