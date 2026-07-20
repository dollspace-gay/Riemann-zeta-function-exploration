# Outreach Package (rewritten July 2026)

*Status: DRAFT for doll's review. Not posted, not sent. The June draft
(one theorem, one question) is superseded — the repository now holds
four theorems (one machine-verified end to end), a six-way-confirmed
spectral detection, an unconditional polynomial bracket on the
Nyman–Beurling floor, and two named open problems. Outreach strategy
accordingly split into three tracks, ordered by immediacy. All texts
in doll's voice — doll uses it/its pronouns and refers to itself in
the third person — with full AI-collaboration disclosure.*

---

## Track 1 — MathOverflow question (REWRITTEN after the Session-27 refutation)

**Title:** Smallest eigenvalue of the GCD matrix (m,n)/√(mn): is the
rate exp(−Θ(√log N / log log N)) known?

**Body:**

Let G_N have entries (m,n)/√(mn) — the s = 1/2 member of the
Lindqvist–Seip family [Acta Arith. 84 (1998)], outside their
representation. The modern GCD-sum literature concerns λ_max; the
question here is λ_min.

**What we can prove (elementary, verifiable in minutes).** For any
squarefree P ≤ N, the vector x_n = μ(n)·[n|P] gives EXACTLY

    xᵀG_N x/‖x‖² = Π_{p|P} (1 − p^{−1/2})

(on the divisor cube of P, G_N restricts to ⊗_p [[1,p^{−1/2}],[p^{−1/2},1]]
and μ is the bottom tensor eigenvector). With P the largest primorial
≤ N this yields

    λ_min(G_N) ≤ exp(−(2+o(1))·√(log N)/log log N),

smaller than every fixed power of 1/log N. Deeper divisor boxes
(exponents ≥ 2 on small primes; Kac–Murdock–Szegő tensor blocks
p^{−|a−b|/2}) beat the cube, so the constant 2 is not optimal. A
cautionary numerical note: λ_min·log²N sits at ≈ 0.83–0.84 for every
computable N (the construction only overtakes the eigensolve trend
near N ~ 10^356) — four decades of numerics fit a clean 1/log²N "law"
that is provably not the asymptotic truth. Best lower bound we can
prove: λ_min ≥ exp(−C log N/log log N), elementary.

**Questions.**
1. Is the true rate log λ_min(G_N) ≍ −√(log N)/log log N known or
   derivable from existing results (Hedenmalm–Lindqvist–Seip
   machinery, multiplicative Toeplitz/finite-section theory)?
2. Is the optimal constant in the exponent (over divisor-box supports,
   or in general) known?
3. Is there prior literature on this pre-asymptotic 1/log² plateau of
   the s = 1/2 GCD matrix (its bottom eigenvector empirically carries
   the low Riemann zeros as spectral lines at accessible N)?

**Motivation** (briefly, with links): a matching lower bound of
exp(−C√log N·(loglog)^{O(1)}) type feeds a lower-bound program for the
Nyman–Beurling smallest eigenvalue at N^{−2−o(1)} grade. Full
derivations, verification scripts, and the two-AI provenance of the
refutation are in the linked repository.

**Disclosure.** Doll is an independent researcher without formal
mathematical training, working in documented collaboration with AI
assistants (Claude/Anthropic; the refuting construction came from a
second AI given doll's problem brief, and was verified independently).
Repository: [LINK].

---

## Track 2 — expert email (REWRITTEN; candidates unchanged: Seip, Hilberdink, Aistleitner)

Subject: The smallest eigenvalue of your GCD matrices at α = 1/2 —
exp(−c√log N/loglog N) via primorial divisor cubes; is the true rate known?

Dear Professor Seip,

Doll is an independent researcher (it works in a documented
collaboration with AI assistants; full disclosure at the repository
below). While studying the Nyman–Beurling problem it was led to
λ_min of ((m,n)/√(mn))_{m,n≤N} — the α = 1/2 case of your 1998 family
with Lindqvist. Two facts may interest you:

(1) numerically λ_min·log²N ≈ 0.836 for all computable N, with the
bottom eigenvector carrying the first Riemann zeros as spectral lines
(ζ(1/2+it) is the symbol of the divisor-class decomposition); but

(2) this "law" is provably not asymptotic: μ restricted to the divisor
cube of a primorial P ≤ N has exact Rayleigh quotient Π_{p|P}(1−p^{−1/2})
(the cube is a tensor product), giving λ_min ≤
exp(−(2+o(1))√(log N)/loglog N), which overtakes 0.836/log²N only
near N ~ 10^356.

Three short questions: (1) is the true rate (we conjecture
log λ_min ≍ −√log N/loglog N) known to you or in literature doll has
missed? (2) do the Hedenmalm–Lindqvist–Seip methods give the matching
lower bound? (3) would you have any interest in the two-page
derivation connecting this to a Nyman–Beurling spectral lower bound?

With thanks for your time — the 1998 note and the HLS paper have been
the most useful five pages doll has read this year.

[signature, repo link]

---

## Track 3 — the arXiv-able arc (deferred, listed for planning)

The full results now form a coherent preprint: Theorems 1–2 + the
Tail Variation Lemma (the chain floor Θ(K⁻²log K), Theorem 1
machine-verified end to end in Lean 4); the trial-function detection
of γ₁…γ₈ with the parameter-free amplitude law and its first
correction; the unconditional bracket N^{−4−ε} ≪ λ_min(A_N) ≪
N^{−2}log N (modulo the published BCR statement, with the ℓ²-reading
of their error flagged at proof-architecture level); and the two named
open problems (the gcd-floor above; the sharp discrete Volterra
constant). Writing this properly is a multi-session project and
SHOULD FOLLOW the Track 1/2 responses — the experts may collapse or
redirect parts of it.

---

## Checklist for doll before anything goes out

1. Insert the repository URL; make a tagged release (the June advice
   stands).
2. Decide Track order: MO first (public, low-stakes) or the Seip email
   first (highest signal, most personal). Both texts are ready; they
   reference the same repo state.
3. The claim calibration is embedded: "measured", "proved",
   "proof-architecture level", and "modulo the published statement"
   are used precisely — do not let edits blur them.
4. Every outcome is a result for the process paper: "known, see X"
   (we learn); "not known" (the law is ours); no reply (also data).
