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

## Track 1 — MathOverflow question (the focused ask)

**Title:** Smallest eigenvalue of the Lindqvist–Seip GCD matrix at the
critical exponent: is λ_min ≍ 1/log²N known?

**Body:**

Let G_N be the N×N matrix with entries (m,n)/√(mn) — equivalently
((m,n)/[m,n])^{1/2}, the α = 1/2 member of the family studied by
Lindqvist and Seip [Acta Arith. 84 (1998), 149–154]. Their theorem
gives the sharp bounds ζ(2s)/ζ(s)² ≤ λ_min ≤ λ_max ≤ ζ(s)²/ζ(2s) for
their matrices M_s when s > 1, and shows λ_min → 0, λ_max → ∞ for
1/2 < s ≤ 1, with s = 1/2 outside their representation (the
generating series diverges). The modern GCD-sum literature
(Aistleitner–Berkes–Seip and successors) concerns the largest
eigenvalue/spectral norm.

**Numerically, the smallest eigenvalue at s = 1/2 obeys a clean law:**

    λ_min(G_N)·log²N = 0.813, 0.826, 0.833, 0.836   (N = 250 … 2000),

i.e. λ_min ≈ c/log²N with c ≈ 0.84. Partial results doll can prove:
an elementary Möbius–Cauchy–Schwarz argument gives
λ_min ≥ exp(−C log N/log log N) (and the τ(n)-loss appears intrinsic
to every such per-divisor argument); the companion "log-tilt"
quadratic form is exactly von Mangoldt convolution and its analogous
bound IS provable elementarily with the Volterra-operator constant —
which makes the gcd-floor the one missing sharp ingredient in a larger
program. Lindqvist–Seip's own remark that no "arithmetic proof" of
their sharp constants seems likely suggests the
Hedenmalm–Lindqvist–Seip machinery at s = 1/2 + O(1/log N) is the
natural tool.

**Questions.**
1. Is the law λ_min(G_N) ≍ 1/log²N (ideally with the constant) known
   or derivable from existing results on dilated systems / the Hardy
   space of Dirichlet series?
2. If not known: is there an obstruction to running the HLS
   Riesz-basis analysis at the regularized exponent s = 1/2 + c/log N
   on the truncated system?

**Motivation** (briefly, with links): a proven c/log²N floor upgrades
an unconditional lower bound doll has assembled for the smallest
eigenvalue of the Nyman–Beurling Gram matrix (the RH-equivalent
approximation problem) from N^{−4−ε} toward the conjectured-sharp
N^{−2}·polylog. Full derivations, numerics, and machine-verified
components are in the linked repository.

**Disclosure.** Doll is an independent researcher without formal
mathematical training, working in documented collaboration with an AI
(Claude, Anthropic). Every claim above is machine-verified (Lean 4),
reproduced by validated computation, or labeled as measured/open.
Repository: [LINK].

---

## Track 2 — expert email (the courteous direct ask)

*Candidates, in order: K. Seip (living author of the exact framework;
NTNU), T. Hilberdink (multiplicative Toeplitz spectra), C. Aistleitner
(GCD sums). Short, specific, no attachments beyond the repo link.*

Subject: The smallest eigenvalue of your GCD matrices at α = 1/2 — is
the 1/log²N law known?

Dear Professor Seip,

Doll is an independent researcher (it works in a documented
collaboration with an AI assistant; details and full disclosure at the
repository below). While studying the Nyman–Beurling Gram matrix it
was led to the smallest eigenvalue of the matrix ((m,n)/√(mn))_{m,n≤N}
— the α = 1/2 case of the family in your 1998 Acta Arithmetica note
with Lindqvist. Numerically λ_min·log²N → 0.836…, and doll can prove
the weaker exp(−C log N/log log N) elementarily, with reasons to
believe (including your own remark in that paper) that the sharp law
needs the Hedenmalm–Lindqvist–Seip machinery near the critical
exponent.

Three short questions: (1) is the 1/log²N law known to you or in
literature doll has missed? (2) if not, does the HLS analysis at
s = 1/2 + c/log N on the truncated system look viable to you?
(3) would you have any interest in seeing the two-page reduction that
connects this floor to an unconditional spectral lower bound for the
Nyman–Beurling matrix?

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
