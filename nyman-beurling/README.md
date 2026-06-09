# Nyman–Beurling / Báez-Duarte: Studying an Object Whose Smallness IS RH

*Started June 2026, after the Gram-matrix exploration (see repo root). The
lesson from that project: our matrix was built FROM zeros on the line, so
no discovery about it could bear on RH (paper v4, §7.2). This direction
fixes that structural flaw: here the quantity we compute is RH-equivalent
by theorem.*

## The criterion

Let ρ(x) = x − ⌊x⌋ be the fractional part. For each integer k ≥ 1 define

    e_k(t) = ρ(1/(kt)),     t > 0,

a function in L²(0, ∞). Let χ be the indicator of (0, 1].

**Theorem (Nyman 1950, Beurling 1955; Báez-Duarte 2002 for the
integer-dilation strengthening).** RH holds if and only if χ lies in the
closed linear span of {e_k : k ≥ 1} in L².

So define the distance

    d_N² = min over (c_1..c_N) of ‖χ − Σ_{k≤N} c_k e_k‖².

**RH ⟺ d_N → 0.** Moreover the *rate* is pinned down:

**Theorem (Burnol 2002; see also Báez-Duarte–Balazard–Landreau–Saias).**
Unconditionally, liminf d_N² · log N ≥ Σ_ρ m_ρ²/|ρ|², the sum over
distinct nontrivial zeros with multiplicities m_ρ. If RH holds and all
zeros are simple this constant is

    C = Σ_ρ 1/|ρ|² = 2 + γ − log 4π ≈ 0.0461914.

The conjecture (BDBLS) is that d_N² ~ C / log N exactly. **Proving
d_N² → 0 — equivalently any o(1) upper bound — is RH.** Every structural
fact about this approximation problem is, by construction, about RH.

Two normalizations appear in the literature: the ambient space L²(0, ∞)
(Báez-Duarte's setting) and Nyman's classical L²(0, 1). We compute both —
they differ by an explicit rank-structure (see below) — and let the data
say which satisfies the C/log N law, rather than trusting memory.

## Exact formulas used by the computation

Everything below is elementary and re-derivable; we use no formula from
memory without an internal cross-check in the code.

**Cross terms.** b_k = ⟨χ, e_k⟩ = ∫₀¹ ρ(1/(kt)) dt. Substituting
u = 1/(kt) and splitting at u = 1:

    b_k = (log k + 1 − γ) / k        (exact; b_1 = 1 − γ is classical).

**Gram entries, L²(0,∞).** A_{mn} = ⟨e_m, e_n⟩ = ∫₀^∞ ρ(u/m)ρ(u/n) u⁻² du
(after u = 1/t). Three reductions:

1. *GCD scaling:* substituting u = g·w with g = gcd(m,n), m = gm′, n = gn′:
   **A_{mn} = A_{m′n′}/g.** Only coprime pairs need direct computation.
   In particular A_{mm} = A_{11}/m with A_{11} = ∫₀^∞ ρ(1/t)² dt = log 2π − γ.
2. *Unit interval:* for u < 1 ≤ m, ρ(u/m) = u/m, so
   ∫₀¹ = 1/(mn) exactly, leaving the integral over [1, ∞).
3. *Periodization:* f(u) = ρ(u/m)ρ(u/n) has period L = mn (m, n coprime).
   Splitting [1, ∞) into [1, 1+L] plus the periodic tail, and summing the
   tail over periods termwise:

       ∫_{1+L}^∞ f(u) u⁻² du = ∫₀^L f(1+w) · ψ₁((1+L+w)/L) / L² dw,

   where ψ₁ is the trigamma function (Σ_{j≥0} (x+j)⁻² = ψ₁(x)). The head
   integral over [1, 1+L] is *exact*: between consecutive breakpoints
   (multiples of m or n), f(u)/u² = 1/(mn) − (b/m + a/n)/u + ab/u² with
   a = ⌊u/m⌋, b = ⌊u/n⌋ constant, whose antiderivative is
   u/(mn) − (b/m + a/n) log u − ab/u. The tail integrand is smooth
   (trigamma argument in [1, 2.1]) and piecewise-polynomial × analytic, so
   fixed-order Gauss–Legendre per segment is near machine precision.

**Nyman normalization.** For t > 1, 1/(mt) < 1 so e_m(t)e_n(t) = 1/(mnt²),
giving ∫₁^∞ e_m e_n dt = 1/(mn) exactly. Hence

    G^{(0,1)}_{mn} = A_{mn} − 1/(mn),

and both Gram matrices come from one computation.

**Distance.** d_N² = ‖χ‖² − bᵀ G⁻¹ b = 1 − bᵀ G⁻¹ b, evaluated by
eigendecomposition with explicit spectral-cutoff sensitivity reporting,
because G is severely ill-conditioned (that ill-conditioning is not a
nuisance — it is the structure of the problem, and diagnosing
ill-conditioned Gram matrices is precisely the toolkit developed in the
first phase of this repo).

**Known anchor for validation.** d_1² = 1 − b_1²/A_{11}
= 1 − (1−γ)²/(log 2π − γ) ≈ 0.858212 in L²(0,∞).

## Why our toolkit fits

The optimal coefficients are conjecturally Möbius-shaped:
c_k ≈ −μ(k)(1 − log k / log N) — from the classical identity
Σ_{k≤x} μ(k)⌊x/k⌋ = 1, which rearranges (x = 1/t) to
χ(t) = (1/t)Σμ(k)/k − Σμ(k)e_k(t) for t ≤ 1. The approximation problem
is secretly about how μ cancels, i.e. about 1/ζ. The questions our Gram-forensics
methods can ask, which the literature has computed *around* but (to our
knowledge) not systematically:

1. **Which direction is nearly null?** The λ_min eigenvector of G_N is a
   combination of dilations that is almost invisible in L². What is its
   arithmetic structure (Möbius-like? supported on smooth numbers? on
   primes?), and how does it evolve with N?
2. **Where does the unreachable mass live?** The residual function
   r_N(t) = χ − Σ c_k e_k has ‖r_N‖² = d_N². What is its profile —
   concentrated near t = 0 (high dilations missing) or spread?
3. **Conditioning growth law.** κ(G_N) vs N: polynomial or worse? Which
   k-windows produce the degeneracy (cf. our zero-cluster localization)?
4. **Rate forensics.** d_N² log N vs the Burnol constant: approach from
   above, and the shape of the second-order term.

## Honest expectations

This will not prove RH. The realistic outcomes, in descending likelihood:
reproduce and sharpen known numerics; find a genuinely new structural
observation about the null directions / residual of the *right* object;
suggest an estimate worth a professional's time. The same standards as the
rest of this repo apply: every claim either machine-checked, brute-force
cross-validated, or labeled conjecture.

## Files

- `compute_dN.py` — Gram matrix via the exact head + trigamma tail scheme,
  self-tests (closed forms + brute-force integration), d_N curves in both
  normalizations, conditioning, coefficient and eigenvector forensics.
- `RESULTS.md` — running log of computed results (filled as we go).

## References

- B. Nyman, *On some groups and semigroups of translations*, thesis, Uppsala 1950.
- A. Beurling, *A closure problem related to the Riemann zeta-function*, PNAS 41 (1955) 312–314.
- L. Báez-Duarte, *A strengthening of the Nyman–Beurling criterion for the Riemann hypothesis*, Atti Accad. Naz. Lincei 14 (2003) 5–11.
- J.-F. Burnol, *A lower bound in an approximation problem involving the zeros of the Riemann zeta function*, Adv. Math. 170 (2002) 56–70.
- L. Báez-Duarte, M. Balazard, B. Landreau, E. Saias, *Notes sur la fonction ζ de Riemann, 3*, Adv. Math. 149 (2000) 130–144.
- V. I. Vasyunin, *On a biorthogonal system associated with the Riemann hypothesis*, St. Petersburg Math. J. 7 (1996) 405–419.
- M. Balazard, *Completeness problems and the Riemann hypothesis: an annotated bibliography*, in Number Theory for the Millennium I (2002).
