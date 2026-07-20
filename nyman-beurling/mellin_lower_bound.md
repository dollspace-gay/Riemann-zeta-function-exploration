# Toward a Lower Bound on λ_min(A_N): the Mellin Route

*Working notes, July 2026 (Plan 5, `plans/05`). Status: M0 sourced, M1
proved (elementary) + numerically checked (`plan5_sr.py`), M2 assembly
sketched with its analytic care points named, M3 not attempted. Recalled
analytic inputs are flagged; nothing recalled is load-bearing yet.*

## M0. The identity and its provenance

The object is the verified Session-6 identity

    xᵀA_N x = (1/2π) ∫_ℝ |ζ(½+it)|²/(¼+t²) · |X(t)|² dt,
    X(t) = Σ_{k≤N} x_k k^{−½−it},

equivalently: the Mellin transform of e_k on Re s = ½ is −k^{−s}ζ(s)/s,
and Plancherel. Literature anchors located (July 2026): Ehm,
*On certain Gram matrices and their associated series*, arXiv:2405.06349,
studies exactly the weighted kernels G^(q)_{u,v} =
(1/2πi)∫ u^{−s}v^{−(1−s)} ζ(s)ζ(1−s)/(s(1−s))^q ds and gives closed
evaluations (his Theorem 2.1, q = 1, equivalent to BDBLS Prop. 90) — a
rigorous published foundation for the identity's kernel side. The
formula-level statement ê_k = −k^{−s}ζ(s)/s is standard in this
literature (Báez-Duarte). What remains for full rigor in OUR use is only
the Plancherel bookkeeping for finite combinations — textbook material,
no research content.

## M1. The coefficient uncertainty lemma (proved, elementary, explicit)

**Lemma.** Let x ∈ ℝ^N with Σ_{d≤N} x_d² = 1, and define the divisor
partial sums

    s_r = Σ_{d | r} x_d      (r ≤ N).

Then

    Σ_{r≤N} τ(r) · s_r² / r  ≥  1 / (N (log N + 1)),

τ = number-of-divisors. In particular Σ s_r²/r ≥ N^{−1−ε} for every
ε > 0, but the τ-weighted form is unconditional and constant-explicit.

*Proof.* Möbius inversion on the divisor lattice: x_d = Σ_{r|d} μ(d/r)s_r
(all divisors of d ≤ N are ≤ N, so this is closed within the index
range). Cauchy–Schwarz with μ² ≤ 1: x_d² ≤ τ(d)·Σ_{r|d} s_r². Summing
over d ≤ N and swapping:

    1 ≤ Σ_{r≤N} s_r² Σ_{m ≤ N/r} τ(rm)
      ≤ Σ_{r≤N} s_r² τ(r) Σ_{m ≤ N/r} τ(m)          [τ(rm) ≤ τ(r)τ(m)]
      ≤ Σ_{r≤N} s_r² τ(r) · (N/r)(log N + 1),

using the elementary Σ_{m≤M} τ(m) = Σ_{j≤M} ⌊M/j⌋ ≤ M(log M + 1). ∎

**Interpretation.** ζ·X has Dirichlet coefficient exactly s_r at every
index m = qr with q prime, q > N (the only divisors of qr that are ≤ N
are the divisors of r). The lemma says: no unit coefficient vector can
make ALL these products small — you cannot hide from all the primes at
once. This is the mass that any mean-value lower bound harvests.

**Numerical check** (`plan5_sr.py`): saturation Q(x)·N(log N + 1),
Q(x) = Σ τ(r)s_r²/r, tabulated for the actual λ_min eigenvectors of
A_N, the Theorem-1 chain witness, random unit vectors, and δ_1. The
eigenvectors are the adversarial case: their saturation measures how
lossy the lemma is against the true minimizers.

## M2. Assembly sketch toward λ_min ≫ N^{−3−ε} (not yet a proof)

Window [T, 2T] with T ≍ N^{2+ε}; the weight costs T⁻² ≈ N^{−4−2ε}; on
the window, replace ζ by an approximate-functional-equation truncation
(length √(T/2π) ≈ N^{1+ε/2}), so ζ·X is a Dirichlet polynomial of
length ≈ N^{2+ε} ≲ εT — inside Montgomery–Vaughan diagonal dominance.
The special indices m = qr (q prime in (N, T/r]) contribute

    ∫_T^{2T} |ζX|² ≳ T · Σ_{r≤N} s_r²/r · (Σ_{q ∈ (N, ·]} 1/q)  ≳  T · N^{−1−ε},

whence λ_min ≳ T⁻² · T · N^{−1−ε} = N^{−3−ε}. Combined with Theorem 1:

    N^{−3−ε} ≪ λ_min(A_N) ≪ N^{−2} log N        (conjectured truth: right edge).

Care points that make M2 real work, in order of danger:
1. AFE error terms integrated against |X|² (and the χ-factor's second
   sum) — the standard statements are RECALLED and must be taken from a
   source with explicit error terms before use.
2. MV's O(m) diagonal error against indices m ~ N²: needs the m ≤ εT
   bookkeeping done honestly.
3. The (N, T/r] prime harvest loses a log for r near N — absorbed in ε,
   but should be stated.
Every intermediate inequality is checkable at finite N against the
cached ζ-grid and the exact A_N before it enters the writeup.

## M3 (not attempted): closing N^{−3} → N^{−2−ε}

The loss is structural: T ≍ N² for diagonal dominance overpays in the
t⁻² weight while the true minimizers' |X|² mass sits at t ≍ N (measured
in Session 6: chain-difference mass concentrates at t ≳ K). Candidate
routes: mollified Cauchy–Schwarz (first moments reach shorter windows),
twisted-second-moment technology (check current admissible length/height
trade-offs in the literature — do not recall them), or off-diagonal MV
cancellation for norm-constrained coefficients. If none close, the
sharpest statement of the obstruction is itself the deliverable.
