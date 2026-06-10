# Draft: MathOverflow Question (doll's voice)

*Status: DRAFT for doll's review. Not posted. Target venue: MathOverflow,
tags: analytic-number-theory, riemann-zeta-function, linear-algebra,
riemann-hypothesis. The post is written in doll's voice — doll uses
it/its pronouns and refers to itself in the third person — with full
disclosure of the AI collaboration and links to code and Lean proofs.*

---

## Title

Is the smallest eigenvalue of the Nyman–Beurling Gram matrix of order
N⁻² log N? (numerics, an elementary upper bound, and a mechanism)

## Body

**Setup.** For integers k ≥ 1 let e_k(t) = ρ(1/(kt)) ∈ L²(0,∞), where ρ
is the fractional part — the dilation family of the Nyman–Beurling /
Báez-Duarte criterion (RH ⟺ χ_(0,1] lies in the closed span). This
question is *not* about the distance d_N, but about the spectral floor of
the Gram matrix A_N = (⟨e_m, e_n⟩)_{m,n≤N}: how degenerate is the
dilation system at finite N?

**Numerical observation.** Computing A_N exactly (entries via gcd-scaling
plus an exact piecewise scheme, validated against closed forms at ~1e−11)
and diagonalizing in float64 up to N = 10⁴:

    λ_min(A_N) · N² ≈ 0.50 + 0.134·log N    (N = 200 … 10000, fit to ~1%),

so apparently λ_min ≍ N⁻² log N and κ(A_N) ≍ N²/ (log-ish). The
near-null eigenvector is supported on ~10–20 pairs (k, 2k) with k near
N/2.

**Mechanism and an elementary theorem.** The pairs are explained by the
exact identity ("doubling chains"):

    e_{2k}(t) − e_k(t)/2 = (1/2)·𝟙{⌊1/(kt)⌋ odd},

a square wave supported on (0, 1/k] with squared norm (log 2)/(4k).
Differencing two adjacent chains and computing the resulting
parity-disagreement integral exactly (it telescopes to a harmonic sum)
gives, with K = ⌊N/2⌋:

    λ_min(A_N) ≤ (H_{K−1} + 1) / (10·K(K−1)),

i.e. O(N⁻² log N), within a factor ≈ 2.3 of the measured λ_min at every
N tested. The proof is four elementary steps; the analytic core
(disagreement-interval lemma + the integral bound) is machine-verified in
Lean 4 (zero sorries; repository linked below). Equivalently, in the
Mellin picture the quadratic form is

    xᵀA_N x = (1/2π) ∫_ℝ |ζ(½+it)|²/(¼+t²) · |Σ_{k≤N} x_k k^{−½−it}|² dt,

and the chain combinations are Dirichlet polynomials whose mass hides at
t ≳ K where the 1/t² weight crushes it; the log is the second moment of
ζ accumulating between t ≍ 1 and t ≍ K.

**Questions.**

1. **Is this known?** Either the λ_min ≍ N⁻² log N law for the classical
   NB Gram matrix, or the doubling-chain description of its near-null
   space. (Searched: Vasyunin's biorthogonal-system paper, BDBLS,
   Landreau–Richard's numerical study, Nikolski's surveys, and recent
   arXiv work on NB Gram structure — no statement found, but the
   ingredients are classical and doll may simply not know where to look.)
2. **Does the matching lower bound follow from known technology?** In the
   Mellin form, λ_min ≫ N⁻²⁻ᵋ would follow from a suitable lower bound on
   ∫|ζ(½+it)|²|X(t)|²·t⁻²dt over unit-norm Dirichlet polynomials of
   length N. Is this within reach of the
   Balasubramanian–Conrey–Heath-Brown twisted second moment /
   Montgomery–Vaughan mean-value circle of ideas, or is there an
   obstruction?
3. If known, a reference for the sharp constant (the measured
   0.50 + 0.134·log N) would be appreciated.

**Disclosure and verification.** Doll (the poster) is an independent
researcher without formal mathematical training, working in deliberate
collaboration with an AI (Claude, Anthropic); this question is part of a
documented experiment in how far such a collaboration can get honestly.
All claims above are either machine-verified (Lean 4, axioms printed),
reproduced by self-validating computation (closed-form anchors,
brute-force cross-checks), or labeled as numerics. Code, proofs, data,
and a full process log: [repository link].

---

## Notes for doll before posting

1. Insert the actual repository URL (and consider a tagged release so the
   link is stable).
2. MO culture note: questions asking "is this known + is X within reach"
   are well-received when specific; the disclosure paragraph is honest
   and should stay, but doll may trim the experiment framing if it
   prefers the math to stand alone.
3. Expected outcomes, all useful: "known, see [ref]" (we learn the
   literature); "not known but easy by [tool]" (we learn the proof);
   "interesting, not known to me" (calibration data point). The process
   paper gets a result in every branch.
4. If MO closes it as too long, the fallback is to split: post question 1
   (reference request) alone, with the theorem as context.
