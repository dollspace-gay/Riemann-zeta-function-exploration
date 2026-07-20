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

## M2 revision (July 2026, Thread-1 session): one route dead, the target sharpened

Numerics this session (`~/rh_output/thread1_harvest.txt`, true
λ_min eigenvector of A_2000):

1. **The harvest mechanism is real and measured.** At the special
   indices m = qr (q prime > N) the coefficients of ζ·X are exactly
   s_r, and the harvested mass Σ c_m²/m responds to the window
   exponent T = N^{2+δ} exactly as the geometry predicts: the
   adversary's s_r mass sits at r ≈ N/2…N, so at δ = 0 the prime
   window (N, M/r] pinches shut there (measured harvest: 0.01× the
   lemma floor) and reopens for δ bounded away from 0 (measured:
   0.67×, 1.72×, 2.59× at δ = ¼, ½, ¾). **δ > 0 is forced.** (One
   instrument bug — unclamped empty windows — caught and fixed in-log.)
2. **The truncated-sum route is DEAD, with the arithmetic recorded.**
   Using the length-T truncated Dirichlet sum for ζ on [T, 2T] makes
   ζ·X a polynomial of length NT; the Montgomery–Vaughan error term
   Σ b_m² over m ≤ NT is then ~ NT·polylog, exceeding the main term
   T·(harvest) ~ T·N^{−1−ε} by ~N². No choice of δ repairs this.
   The sketch's "care point 2" was real and fatal for this variant.
3. **Two live routes remain:**
   - (a) the genuine √T-length approximate functional equation:
     product length N√T ≤ εT ⟺ T ≥ N²/ε² ✓, harvest window still
     open ✓ — but the χ-factor's conjugate sum contributes at the
     same order as the main sum, and controlling the cross term is
     the classical hard part of every second-moment argument. Not
     attempted today.
   - (b) **the BCHB reduction (new, preferred).** At T = N^{2+δ} the
     polynomial length is N = T^{1/(2+δ)} < T^{1/2} — inside the
     Balasubramanian–Conrey–Heath-Brown twisted-second-moment range.
     Taking BCHB as the sourced input, Tier 2 reduces to a
     self-contained finite problem: **lower-bound the explicit BCHB
     quadratic form Q(x) = Σ_{h,k≤N} x_h x_k·(h,k)²/(hk)·(log-weights)
     over unit x.** The kernel is PSD (gcd Gram factorization
     (h,k) = Σ_{d|h,d|k} φ(d)); a Möbius-inverse route to its floor
     is sketched (z_d = Σ_{d|h} x_h d/h has polylog-bounded inverse),
     but Gershgorin fails at a log log factor — the floor is delicate
     at polylog level and is now THE open kernel of Tier 2.

**Status: M2 not closed; upgraded from sketch to a sharply posed
problem (the BCHB floor) with every reduction numerically anchored and
one dead end documented.** Risk assessment unchanged from the plan:
this was the high-risk thread, and the honest outcome of the session
is a better problem, not a theorem.

## M2 second revision: the BCHB floor MEASURED — constant over log

`~/rh_output/bchb_floor.txt`. With the recalled kernel shape
K_{hk} = (h,k)/√(hk)·(log(T(h,k)²/2πhk) + 2γ − 1) at T = N^{2.5}:

    λ_min(K)·log T = 6.69, 6.77, 6.82, 6.82   (N = 250…2000)

— **the reduced problem's floor is c/log T with c ≈ 6.8, not a power
of N.** If the actual BCHB main term has this kernel and its error is
uniform over coefficients (both RECALLED — the entire weight of Tier 2
now rests on sourcing the 1985 statement), the assembly yields

    λ_min(A_N) ≳ c′·N^{−2−δ}/log N   for every δ > 0,

which is within N^δ·log² of the truth — Tier 2 and Tier 3 would
collapse into one near-optimal theorem. Next action (one session):
obtain BCHB 1985 (and successors) and verify kernel + uniformity;
the K-floor law itself (our measurement) then needs a proof — likely
via the gcd-Gram factorization (h,k) = Σ_{d|h,d|k} φ(d), which makes
K a positive combination of rank-structured pieces with log weights.

## M2 third revision: BCHB SOURCED — both checks pass

Source obtained (arXiv:1411.7764, Bettin–Chandee–Radziwiłł, published
Crelle 729 (2017); quoting and improving BCHB, Crelle 357 (1985)):

**BCHB (their eq. 1.2), θ < ½, sharp cutoff [T, 2T]:**
    ∫|ζ(½+it)|²|A(½+it)|²dt
      = T·Σ_{d,e≤T^θ} (a_d a_e/[d,e])·(log(T(d,e)²/(2πde)) + 2γ + log 4 − 1) + o(T).

**BCR Theorem 1, θ < 17/33, smooth φ:** same main term (t inside the
log, integrated against φ(t/T)) with EXPLICIT error
O(T^{3/20+ε}N^{33/20} + T^{1/3+ε}), N = T^θ, for a_n ≪ n^ε.

**Check 1 (kernel) — PASSES.** With a_h = x_h√h and 1/[d,e] = (d,e)/de,
the main-term kernel is exactly (d,e)/√(de)·(log(T(d,e)²/2πde) + c₀) —
the measured kernel, with c₀ = 2γ + log 4 − 1 (the recalled version had
dropped log 4). Floor re-measured with the sourced constant at the
admissible window T = N^{3.2}:

    λ_min(K)·log T = 10.94, 11.03, 11.06, 11.03   (N = 250…2000)

— constant ≈ 11.0. The c/log T law stands with the exact kernel.

**Check 2 (error/uniformity) — PASSES, two readings.** BCR's error is
explicit in (T, N) for the class a_n ≪ n^ε. Our vectors have
a_h = x_h√h, sup ≤ N^{1/2}: crude sup-norm rescaling multiplies the
error by N, giving admissibility for δ > 1.117 (main N^{2+δ}/log vs
error N^{2.95+0.15δ+ε}); at δ = 1.2, θ = 1/3.2 < 17/33 ✓ and all log
arguments positive ✓. In ℓ² mass our vectors sit INSIDE the class
scale (Σ|a_h|² ≤ N vs class ~N^{1+ε}), so if BCR's error is uniform in
ℓ²-normalized classes (a Section-3 read of their proof), every δ > 0
is admissible.

**Tier 2, reduced to one problem.** Everything is now sourced and
verified except a PROOF of the measured floor law:

    (⋆)  λ_min( K_N(T) ) ≥ c / log T,
         K = (d,e)/√(de)·(log(T(d,e)²/2πde) + c₀),  d, e ≤ N ≤ T^{1/3},

an explicit finite arithmetic-kernel problem (gcd-Gram factorization
(d,e) = Σ_{δ|d,δ|e} φ(δ) makes K a positive combination of
rank-structured pieces; the Möbius-inverse z-map has polylog-bounded
inverse). (⋆) ⟹ **λ_min(A_N) ≫ N^{−3.2}/log N unconditionally** (crude
reading), or N^{−2−δ}/log N ∀δ > 0 (ℓ²-uniform reading) — the latter
within log² of Theorem 1's upper bound. (⋆) is the whole game now.
