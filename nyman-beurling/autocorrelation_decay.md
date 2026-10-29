# Can arithmetic and autocorrelation force sufficient decay?

September 24, 2026. **The sufficient decay estimate remains unproved.**
This continuation of [annihilator rigidity](annihilator_rigidity.md)
establishes the precise limitation of the autocorrelation input. It supplies
normalized L² functions with exact self-autocorrelation and nonsummable
primitives, computes their arithmetic defects, and includes a real-valued
version. No off-critical-line zero is asserted to exist. These are elementary
Mellin witnesses and Hilbert-space projections, not a claimed new RH criterion.

## 1. What the desired decay would accomplish

Retain H=L²(0,∞), e_k(t)={1/(kt)}, χ=1_(0,1], and
W=closure span{e_k:k≥1}. For the canonical remainder
h=χ−P_Wχ, write E=‖h‖² and F_n=∫_0^(1/n)h(t)dt. Earlier we proved

\[
 F_1=E,\qquad F_n=\langle h,D_nh\rangle,\qquad
 |F_n|\le E/\sqrt n,\qquad F_n=o(n^{-1/2}),
 \quad D_nf(t)=f(nt).
\]

The arithmetic equations are

\[
 \sum_{j\ge1}(j\bmod n)(F_j-F_{j+1})=0\quad(n\ge2).
 \tag{1}
\]

If Σ|F_n|<∞, finite-prime Möbius inversion in the preceding note proves
F_n=0, hence E=0. Conversely, RH gives h=0 and this summability. Thus, **for
the canonical remainder**, the proposed summability statement is equivalent
to RH, using the published integer-dilation criterion. It is a legitimate
sufficient target, but cannot be inserted as a harmless regularity assumption.
The external criterion is [Báez-Duarte, Theorem 1.1](https://arxiv.org/pdf/math/0202141).

## 2. The correlation identity follows from projection

Let W be any closed subspace of a real Hilbert space, let h=x−P_Wx,
and let T be a linear map satisfying T(W)⊆W. Then

\[
 \langle h,Tx\rangle-\langle h,Th\rangle
 =\langle h,T(P_Wx)\rangle=0.                         \tag{2}
\]

This holds whether or not h vanishes. Applying it to the integer dilations,
which preserve the actual W, gives the canonical autocorrelation identity.
The logarithmic positive-definite kernel is likewise an ordinary
autocorrelation of an L² function. These remain useful identities, but their
validity is not an additional exclusion of a nonzero projection residual.

More precisely, if some w∈W⊥ has ⟨w,χ⟩≠0, then χ−P_Wχ≠0. This canonical
remainder automatically has all the arithmetic equations and the
autocorrelation identities. Producing an annihilator and then imposing
canonical autocorrelation cannot itself rule out that annihilator: projection
already supplies an object satisfying both. A proof must extract a further
consequence of the particular arithmetic space W.

Both Hilbert-space implications are now formalized in Lean as
`projection_residual_correlation` and
`projection_residual_ne_zero_of_witness`. Their use with the analytic dilation
space, and the analytic results below, are written proofs.

## 3. An exact normalized control, including its tail

We temporarily work over complex L² and use the convention
⟨f,g⟩=∫f(t)overline(g(t))dt, linear in the first slot. This section does
not identify its function with the actual canonical remainder.

Take s=β+iγ with 1/2<β<1, and put

\[
 K_s=\frac1{2\beta-1}+\frac1{|1-s|^2},\qquad
 A_s=\frac1{\overline{s}K_s},\qquad
 B_s=\frac{A_s}{1-s},\qquad E_s=\frac1{|s|^2K_s}>0.
\]

Define the full function, including t>1, by

\[
 h_s(t)=
 \begin{cases}A_s t^{s-1},&0<t\le1,\\ B_s/t,&t>1.\end{cases}       \tag{3}
\]

Direct integration gives

\[
 \|h_s\|^2=|A_s|^2K_s=E_s,
 \qquad \langle h_s,\chi\rangle=A_s/s=E_s,
 \qquad F_s(x):=\int_0^{1/x}h_s(t)dt=E_sx^{-s}.             \tag{4}
\]

Splitting the dilation correlation at t=1/x and t=1, for every real x≥1,
gives

\[
 \begin{aligned}
 \langle h_s,D_xh_s\rangle={}&
 \left(\frac{|A_s|^2}{2\beta-1}
       +\frac{A_s\overline{B_s}}{1-s}\right)x^{-s}\\
 &+\left(|B_s|^2-\frac{A_s\overline{B_s}}{1-s}\right)x^{-1}
 = E_sx^{-s}=F_s(x).                                      \tag{5}
 \end{aligned}
\]

Indeed A_s overline(B_s)/(1−s)=|B_s|². The tail was essential to this
cancellation. This is the exact self-correlation of the same h_s whose
primitive is being computed; it is stronger than choosing an unrelated
positive-definite sequence.

For a_s(y)=e^(y/2)h_s(e^y), its actual correlation is

\[
 C_s(v)=\int_{\mathbb R}a_s(y)\overline{a_s(y+v)}dy
       =E_s e^{-(\beta-1/2)|v|}e^{-i\gamma v}.             \tag{6}
\]

It is positive definite because it is a Gram kernel of translates. Yet

\[
 \sum_{n\ge1}|F_s(n)|=E_s\sum_{n\ge1}n^{-\beta}=\infty.   \tag{7}
\]

Moreover, d_j=F_s(j)−F_s(j+1) satisfies
Σj(j+1)|d_j|²<∞, since d_j=E_s s j^(−s−1)(1+O_s(1/j)). Thus this
control respects the weighted L² threshold, the target/norm normalization,
and exact autocorrelation simultaneously. It is not a critical-line mode
mistakenly treated as square integrable.

## 4. Arithmetic defect: the factor that must be excluded

The classical fractional-part Mellin identity is

\[
 \int_0^\infty t^{s-1}\{1/t\}\,dt=-\zeta(s)/s,
 \qquad 0<\operatorname{Re}s<1.
\]

See [Báez-Duarte, page 4, immediately after (2.6)](https://arxiv.org/pdf/math/0202141).
Scaling t, and subtracting the integral above 1, gives

\[
 \int_0^1t^{s-1}e_k(t)dt
 =-\frac{\zeta(s)}{s k^s}-\frac1{k(1-s)}.
\]

The B_s/t tail in (3) cancels the second term, so

\[
 \boxed{\langle h_s,e_k\rangle=-E_s\zeta(s)k^{-s}},
 \qquad
 \boxed{\sum_{j\ge1}(j\bmod n)d_j
        =E_s\zeta(s)(1-n^{1-s}).}                       \tag{8}
\]

The residue series is absolutely convergent. These are explicit defects,
not statements that the controls satisfy the arithmetic equations.

**Conditional conclusion:** if ζ(s)=0 for some 1/2<Re(s)<1, then (3)
is an L² annihilator with nonzero target overlap, exact normalized
self-autocorrelation, and nonsummable primitives. We have not exhibited
such an s. In every numerical control below ζ(s)≠0 and (1) fails.
Consequently, these examples do not disprove the desired joint decay theorem.
They show exactly how a hypothetical off-line zero would survive the
autocorrelation test; excluding it must use arithmetic beyond positivity.

## 5. A real version and its precise projection meaning

Taking the real part of (3) without changing its coefficients does **not**
in general preserve the nonlinear normalization or self-correlation. Here
is a real construction that does preserve them.

Let T be the closed subspace of H consisting of functions equal to b/t on
t>1 for some constant b. It is invariant under D_x for x≥1. For Re(s)>1/2
and s≠1, define the bounded functional on its complexification

\[
 L_s(f)=\int_0^1f(t)t^{s-1}dt+\frac b{1-s}.
\]

Splitting ∫_0^x at 1 proves

\[
 L_s(D_xf)=x^{-s}L_s(f)\quad(x\ge1).                    \tag{9}
\]

For γ≠0 let g_s(t)=t^(s−1) below 1 and 1/((1−s)t) above 1, and let
u=Re(g_s), v=Im(g_s). Set

\[
 J_s=\int_0^\infty g_s(t)^2dt
     =\frac1{2s-1}+\frac1{(1-s)^2},\quad
 G=\frac12\begin{pmatrix}K_s+\operatorname{Re}J_s&\operatorname{Im}J_s\\
                  \operatorname{Im}J_s&K_s-\operatorname{Re}J_s\end{pmatrix},
 \quad q=\binom{\operatorname{Re}(1/s)}{\operatorname{Im}(1/s)}.
\]

The functions u and v are linearly independent, as their phases vary on
(0,1), so G is positive definite. Let c=G⁻¹q, A=(c₁−ic₂)/2, and

\[
 h_s^{\mathbb R}=c_1u+c_2v=2\operatorname{Re}(A g_s),
 \quad E_s^{\mathbb R}=q^TG^{-1}q>0.                    \tag{10}
\]

This is the canonical remainder for the auxiliary space
W_s={f∈T real : L_s(f)=0}, because its orthogonal complement within T
is span{u,v}. Equation (9) gives D_xW_s⊆W_s. Applying (2) yields

\[
 \|h_s^{\mathbb R}\|^2=\langle h_s^{\mathbb R},\chi\rangle
 =E_s^{\mathbb R},\qquad
 F_s^{\mathbb R}(x)=2\operatorname{Re}\left(\frac A s x^{-s}\right)
 =\langle h_s^{\mathbb R},D_xh_s^{\mathbb R}\rangle.      \tag{11}
\]

Its logarithmic correlation is positive definite and its primitives are
not absolutely summable. To verify the latter, write A/s=R e^(iθ), R>0.
The primitive is 2R x^(−β) cos(θ−γ log x). In each of infinitely many
fixed-width logarithmic intervals the absolute cosine is at least 1/2.
The integral of its absolute value diverges there because β<1. Its
derivative has integrable absolute value on [1,∞), so the difference
between the corresponding integral and integer partial sums stays bounded.
Hence Σ|F_s^R(n)|=∞ as well.

The full arithmetic defect is

\[
 \langle h_s^{\mathbb R},e_k\rangle
 =-2\operatorname{Re}\left(\frac A s\zeta(s)k^{-s}\right). \tag{12}
\]

A hypothetical zero makes it an actual real annihilator. W_s would then
contain the full integer-dilation span W. Its projection residual (10)
is **not asserted to equal** χ−P_Wχ, nor to satisfy the latter's additional
cellwise projection description. However, its nonzero target pairing
forces the actual canonical remainder to be nonzero, and that remainder
automatically satisfies its own exact arithmetic and autocorrelation
identities by §2. No extra invariance of W under adjoints or noninteger
dilations has been assumed.

## 6. What is still needed

The new results justify neither Σ|F_n|<∞ nor a uniform residual-gain bound.
Positive definiteness, the exact autocorrelation identity, and the
weighted-L² threshold allow the slow modes above. Their arithmetic
admissibility depends precisely on a zeta factor. Any claimed decay
argument must show where it excludes these modes, without presupposing
zero-freeness or discarding the Abel boundary. Merely invoking projection
normalization, integer-dilation invariance, or a positive Gram kernel does
not supply that step.

The remaining task is therefore an arithmetic regularity estimate for the
actual canonical remainder, or a direct rigidity argument at the L²
threshold. The earlier sufficient gain estimate remains another possible
route. Neither route has been completed here.

## 7. Reproduction and scope of verification

`autocorrelation_decay.py` checks complex and real controls at
s=0.75+1.3i and s=0.75+i Im(ρ₁). The latter uses a critical-line zero's
height only; **neither test point is a zero**. Independent integration in
log t checks the norm, target pairing, primitive, and dilation covariance,
including infinite tails. Integration over reciprocal cells checks the
full arithmetic defect with its omitted-cell bound retained. Hurwitz-zeta
grouping independently checks the residue formula. Sampled kernel
eigenvalues are diagnostics; positive definiteness is proved above for
all points, not inferred from those samples.

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s nyman-beurling -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python nyman-beurling/autocorrelation_decay.py \
  --output Riemann_Continuation_Verification/residual_capture/autocorrelation_decay.json
```

The report records precision, versions, and a source hash. Numerical
quadrature and high-precision arithmetic are not interval certificates.
All 28 NB tests pass. Across the four controls, the largest independent
quadrature discrepancy is below 1.35e−14 and the largest cell-tail
error/bound ratio is below 0.250. The full pinned Lean build succeeds
(3505 jobs); the audit passes for 203 declarations across nine modules,
allowing only `propext`, `Classical.choice`, and `Quot.sound`.
Only the two general projection lemmas added in this continuation are
Lean-checked; no analytic decay theorem or RH theorem is claimed to have
been formalized. Run `lake build` and then `lake env lean check_axioms.lean`
inside `lean/`, retaining the pinned toolchain and dependency manifest.
