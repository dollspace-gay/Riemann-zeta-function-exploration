import Mathlib.Tactic
import Mathlib.Analysis.Normed.Group.InfiniteSum
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
PR #1 verification, against commit 2962cc484ac89370f8ad55db1c5414e12a59a5a8.

This file verifies the finite pair/cluster algebra and exponential envelope
comparisons in papers/rh_synthesis_2026_10.md. The abstract tail-summation
theorem takes the individual operator bounds and scalar summability as
hypotheses. It does not formalize integration by parts, zero counting,
Weil's explicit formula, the Taylor remainder, or either external strip result.
-/

set_option autoImplicit false

noncomputable section

open scoped BigOperators
open Matrix

namespace WeilVerification

/-- Contribution per unit multiplicity of a reflected pair for a real
entire probe transform. Its contribution at multiplicity `m` is `m • signedPair v`. -/
def signedPair (v : Fin 2 → ℂ) : Matrix (Fin 2) (Fin 2) ℝ :=
  fun i j => (v i * v j + star (v i * v j)).re

/-- The Hermitian outer-product replacement per unit multiplicity, used as
a negative control. -/
def positivePair (v : Fin 2 → ℂ) : Matrix (Fin 2) (Fin 2) ℝ :=
  fun i j => (v i * star (v j) + star (v i) * v j).re

theorem signedPair_entry (v : Fin 2 → ℂ) (i j : Fin 2) :
    signedPair v i j = 2 * (v i).re * (v j).re - 2 * (v i).im * (v j).im := by
  simp [signedPair, Complex.mul_re]
  ring

theorem positivePair_entry (v : Fin 2 → ℂ) (i j : Fin 2) :
    positivePair v i j = 2 * (v i).re * (v j).re + 2 * (v i).im * (v j).im := by
  simp [positivePair, Complex.mul_re]
  ring

theorem signedPair_det (v : Fin 2 → ℂ) :
    (signedPair v).det =
      -4 * ((v 0).re * (v 1).im - (v 1).re * (v 0).im) ^ 2 := by
  simp only [Matrix.det_fin_two, signedPair_entry]
  ring

/-- A common multiplicity scales the two-probe determinant quadratically. -/
theorem weightedPair_det (v : Fin 2 → ℂ) (m : ℝ) :
    (m • signedPair v).det =
      -4 * m ^ 2 * ((v 0).re * (v 1).im - (v 1).re * (v 0).im) ^ 2 := by
  simp only [Matrix.det_fin_two, Matrix.smul_apply, smul_eq_mul, signedPair_entry]
  ring

theorem positivePair_quadratic (v : Fin 2 → ℂ) (x : Fin 2 → ℝ) :
    x ⬝ᵥ (positivePair v *ᵥ x) =
      2 * (x 0 * (v 0).re + x 1 * (v 1).re) ^ 2 +
      2 * (x 0 * (v 0).im + x 1 * (v 1).im) ^ 2 := by
  simp [dotProduct, Matrix.mulVec, Fin.sum_univ_two, positivePair_entry]
  ring

theorem positivePair_nonneg (v : Fin 2 → ℂ) (x : Fin 2 → ℝ) :
    0 ≤ x ⬝ᵥ (positivePair v *ᵥ x) := by
  rw [positivePair_quadratic]
  positivity

theorem signedPair_negative_certificate (v : Fin 2 → ℂ) :
    (![-(v 1).re, (v 0).re]) ⬝ᵥ
        (signedPair v *ᵥ ![-(v 1).re, (v 0).re]) =
      -2 * ((v 0).re * (v 1).im - (v 1).re * (v 0).im) ^ 2 := by
  simp [dotProduct, Matrix.mulVec, Fin.sum_univ_two, signedPair_entry]
  ring

theorem signedPair_positive_certificate (v : Fin 2 → ℂ) :
    (![-(v 1).im, (v 0).im]) ⬝ᵥ
        (signedPair v *ᵥ ![-(v 1).im, (v 0).im]) =
      2 * ((v 0).re * (v 1).im - (v 1).re * (v 0).im) ^ 2 := by
  simp [dotProduct, Matrix.mulVec, Fin.sum_univ_two, signedPair_entry]
  ring

theorem signedPair_both_signs (v : Fin 2 → ℂ)
    (h : (v 0).re * (v 1).im - (v 1).re * (v 0).im ≠ 0) :
    (∃ x : Fin 2 → ℝ, x ⬝ᵥ (signedPair v *ᵥ x) < 0) ∧
    (∃ x : Fin 2 → ℝ, 0 < x ⬝ᵥ (signedPair v *ᵥ x)) := by
  constructor
  · refine ⟨![-(v 1).re, (v 0).re], ?_⟩
    rw [signedPair_negative_certificate]
    nlinarith [sq_pos_of_ne_zero h]
  · refine ⟨![-(v 1).im, (v 0).im], ?_⟩
    rw [signedPair_positive_certificate]
    nlinarith [sq_pos_of_ne_zero h]

/-- Evenness and conjugation symmetry give the two symmetric probe values. -/
theorem symmetric_probe_values (F : ℂ → ℂ)
    (hEven : ∀ z, F (-z) = F z)
    (hConj : ∀ z, F (star z) = star (F z)) (γ h δ : ℝ) :
    F (((γ : ℂ) + Complex.I * δ) - ((γ : ℂ) + h)) =
      star (F (((γ : ℂ) + Complex.I * δ) - ((γ : ℂ) - h))) := by
  rw [← hConj]
  calc
    _ = F (-star (((γ : ℂ) + Complex.I * δ) - ((γ : ℂ) - h))) := by
      congr 1
      apply Complex.ext <;> simp
    _ = _ := hEven _

theorem symmetricPair_sum_channel (w : ℂ) :
    signedPair ![w, star w] *ᵥ ![1, 1] = (4 * w.re ^ 2) • ![1, 1] := by
  ext i
  fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two, signedPair_entry] <;> ring

theorem symmetricPair_difference_channel (w : ℂ) :
    signedPair ![w, star w] *ᵥ ![1, -1] = (-4 * w.im ^ 2) • ![1, -1] := by
  ext i
  fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two, signedPair_entry] <;> ring

theorem symmetricPair_characteristic (w : ℂ) (t : ℝ) :
    (t • (1 : Matrix (Fin 2) (Fin 2) ℝ) - signedPair ![w, star w]).det =
      (t - 4 * w.re ^ 2) * (t + 4 * w.im ^ 2) := by
  simp [Matrix.det_fin_two, signedPair_entry]
  ring

/-- For an actual reflected zero pair, its common multiplicity scales both
channels. The unweighted matrices in this module are per-unit contributions. -/
theorem weightedPair_channels (w : ℂ) (m : ℝ) :
    (m • signedPair ![w, star w]) *ᵥ ![1, 1] =
        (m * (4 * w.re ^ 2)) • ![1, 1] ∧
    (m • signedPair ![w, star w]) *ᵥ ![1, -1] =
        (m * (-4 * w.im ^ 2)) • ![1, -1] := by
  constructor <;> ext i <;> fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two, signedPair_entry] <;> ring

/-- Two unit-multiplicity real on-line zero vectors in the symmetric probe geometry. -/
def onlineCluster (p m : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  fun i j => (![p, m] : Fin 2 → ℝ) i * (![p, m] : Fin 2 → ℝ) j +
    (![m, p] : Fin 2 → ℝ) i * (![m, p] : Fin 2 → ℝ) j

theorem onlineCluster_sum_channel (p m : ℝ) :
    onlineCluster p m *ᵥ ![1, 1] = (p + m) ^ 2 • ![1, 1] := by
  ext i
  fin_cases i <;>
    simp [onlineCluster, Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

theorem onlineCluster_difference_channel (p m : ℝ) :
    onlineCluster p m *ᵥ ![1, -1] = (p - m) ^ 2 • ![1, -1] := by
  ext i
  fin_cases i <;>
    simp [onlineCluster, Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

/-- Equal on-line multiplicities scale both symmetric channels equally. -/
theorem weightedOnlineCluster_channels (p q m : ℝ) :
    (m • onlineCluster p q) *ᵥ ![1, 1] = (m * (p + q) ^ 2) • ![1, 1] ∧
    (m • onlineCluster p q) *ᵥ ![1, -1] = (m * (p - q) ^ 2) • ![1, -1] := by
  constructor <;> ext i <;> fin_cases i <;>
    simp [onlineCluster, Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

theorem onlineCluster_characteristic (p m t : ℝ) :
    (t • (1 : Matrix (Fin 2) (Fin 2) ℝ) - onlineCluster p m).det =
      (t - (p + m) ^ 2) * (t - (p - m) ^ 2) := by
  simp [Matrix.det_fin_two, onlineCluster]
  ring

theorem onlineCluster_nonneg (p m : ℝ) (x : Fin 2 → ℝ) :
    0 ≤ x ⬝ᵥ (onlineCluster p m *ᵥ x) := by
  have h : x ⬝ᵥ (onlineCluster p m *ᵥ x) =
      (p * x 0 + m * x 1) ^ 2 + (m * x 0 + p * x 1) ^ 2 := by
    simp [onlineCluster, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    ring
  rw [h]
  positivity

/-- First two spectral moments coincide despite the negative eigenvalue. -/
theorem equal_moments_different_signs :
    (1 : ℤ) + 15 + 29 = -1 + 21 + 25 ∧
    (1 : ℤ) ^ 2 + 15 ^ 2 + 29 ^ 2 = (-1) ^ 2 + 21 ^ 2 + 25 ^ 2 ∧
    (0 : ℤ) < 1 ∧ (0 : ℤ) < 15 ∧ (0 : ℤ) < 29 ∧ (-1 : ℤ) < 0 := by
  norm_num

/-- The two numerical strip improvements are exact exponential identities. -/
theorem envelope_seven_eighths (B : ℝ) :
    Real.exp (2 * B * (3 / 8)) = Real.exp (-B / 4) * Real.exp (2 * B * (1 / 2)) := by
  rw [← Real.exp_add]
  congr 1
  ring

theorem envelope_eleven_twelfths (B : ℝ) :
    Real.exp (2 * B * (5 / 12)) = Real.exp (-B / 6) * Real.exp (2 * B * (1 / 2)) := by
  rw [← Real.exp_add]
  congr 1
  ring

/-- The Banach-valued summation step of the tail lemma, with the analytic
pointwise estimate and scalar summability supplied explicitly as hypotheses.
It applies, in particular, to continuous linear operators with their operator norm. -/
theorem tail_sum_bound {ι E : Type*} [NormedAddCommGroup E] [CompleteSpace E] (T : ι → E) (q : ι → ℝ) (K : ℝ)
    (hq : Summable q) (hbound : ∀ i, ‖T i‖ ≤ K * q i) :
    Summable T ∧ ‖∑' i, T i‖ ≤ K * ∑' i, q i := by
  have hnorm : Summable (fun i => ‖T i‖) :=
    Summable.of_nonneg_of_le (fun i => norm_nonneg _) hbound (hq.mul_left K)
  constructor
  · exact hnorm.of_norm
  · calc
      ‖∑' i, T i‖ ≤ ∑' i, ‖T i‖ := norm_tsum_le_tsum_norm hnorm
      _ ≤ ∑' i, K * q i := Summable.tsum_le_tsum hbound hnorm (hq.mul_left K)
      _ = K * ∑' i, q i := tsum_mul_left

/-- Component bounds imply the paper's dimension factor in the actual
Hilbert-space operator norm, rather than an entrywise matrix norm. -/
theorem rankOne_norm_bound {d : ℕ} (v w : EuclideanSpace ℂ (Fin d)) (M : ℝ)
    (hv : ∀ j, ‖v j‖ ≤ M) (hw : ∀ j, ‖w j‖ ≤ M) :
    ‖InnerProductSpace.rankOne ℂ v w‖ ≤ (d : ℝ) * M ^ 2 := by
  have bound : ∀ x : EuclideanSpace ℂ (Fin d),
      (∀ j, ‖x j‖ ≤ M) → ‖x‖ ^ 2 ≤ (d : ℝ) * M ^ 2 := by
    intro x hx
    rw [EuclideanSpace.norm_sq_eq]
    calc
      ∑ j, ‖x j‖ ^ 2 ≤ ∑ _j : Fin d, M ^ 2 :=
        Finset.sum_le_sum fun j _ => pow_le_pow_left₀ (norm_nonneg _) (hx j) 2
      _ = _ := by simp
  rw [InnerProductSpace.norm_rankOne]
  nlinarith [bound v hv, bound w hw, sq_nonneg (‖v‖ - ‖w‖)]

/-- Weighted rank-one operator tail. The hypotheses expose precisely the two
analytic inputs still needed: componentwise Fourier decay and summability
of its squared envelope with multiplicity. -/
theorem rankOne_tail_bound {ι : Type*} {d : ℕ}
    (v w : ι → EuclideanSpace ℂ (Fin d)) (m M : ι → ℝ)
    (hm : ∀ i, 0 ≤ m i) (hM : Summable (fun i => m i * M i ^ 2))
    (hv : ∀ i j, ‖v i j‖ ≤ M i) (hw : ∀ i j, ‖w i j‖ ≤ M i) :
    Summable (fun i => m i • InnerProductSpace.rankOne ℂ (v i) (w i)) ∧
      ‖∑' i, m i • InnerProductSpace.rankOne ℂ (v i) (w i)‖ ≤
        (d : ℝ) * ∑' i, m i * M i ^ 2 := by
  apply tail_sum_bound _ _ (d : ℝ) hM
  intro i
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hm i)]
  calc
    _ ≤ m i * ((d : ℝ) * M i ^ 2) :=
      mul_le_mul_of_nonneg_left (rankOne_norm_bound _ _ _ (hv i) (hw i)) (hm i)
    _ = _ := by ring

end WeilVerification
