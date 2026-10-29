import NymanBeurlingL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Tactic

/-! Exact finite identities for the annihilator investigation.

These do not assert that the infinite annihilator vanishes or that any
summability hypothesis needed for infinite inversion holds.
-/

namespace NymanBeurling.Annihilator

/-- A projection residual has the correlation identity for every linear map
preserving the projected subspace. This identity does not imply that the
residual is zero. -/
theorem projection_residual_correlation
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (W : Submodule ℝ H) [W.HasOrthogonalProjection]
    (x : H) (T : H →ₗ[ℝ] H) (hT : ∀ w ∈ W, T w ∈ W) :
    inner (𝕜 := ℝ) (x - W.starProjection x) (T x) =
      inner (𝕜 := ℝ) (x - W.starProjection x) (T (x - W.starProjection x)) := by
  rw [map_sub, inner_sub_right]
  have hz := Submodule.starProjection_inner_eq_zero x (T (W.starProjection x))
    (hT _ (W.starProjection_apply_mem x))
  rw [hz, sub_zero]

/-- An annihilator with nonzero target pairing forces the canonical projection
residual to be nonzero. No arithmetic or decay claim is used here. -/
theorem projection_residual_ne_zero_of_witness
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (W : Submodule ℝ H) [W.HasOrthogonalProjection] (x w : H)
    (hw : ∀ v ∈ W, inner (𝕜 := ℝ) w v = 0)
    (hx : inner (𝕜 := ℝ) w x ≠ 0) : x - W.starProjection x ≠ 0 := by
  intro h
  have heq : x = W.starProjection x := sub_eq_zero.mp h
  exact hx (heq ▸ hw x (heq ▸ W.starProjection_apply_mem x))

/-- The residue-class identity for every positive integer modulus. -/
theorem fract_div_nat (x : ℝ) (n : ℕ) (hn : 0 < n) :
    Int.fract (x / n) - Int.fract x / n = ((⌊x⌋ % n : ℤ) : ℝ) / n := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have he : (⌊x⌋ % n : ℤ) + n * (⌊x⌋ / n) = ⌊x⌋ := Int.emod_add_mul_ediv _ _
  have he' : ((⌊x⌋ % n : ℤ) : ℝ) + (n : ℝ) * ((⌊x⌋ / n : ℤ) : ℝ) = ⌊x⌋ := by
    exact_mod_cast he
  rw [Int.fract, Int.fract, Int.floor_div_natCast]
  field_simp
  nlinarith

/-- A dilation minus its old tail is a function of the reciprocal cell index. -/
theorem dilation_residue (t : ℝ) (n : ℕ) (hn : 0 < n) :
    eDil n t - eDil 1 t / n = ((⌊1 / t⌋ % n : ℤ) : ℝ) / n := by
  unfold eDil
  norm_num only [Nat.cast_one, one_mul]
  have harg : (1 : ℝ) / ((n : ℝ) * t) = (1 / t) / n := by
    rw [div_div, mul_comm t]
  rw [harg]
  exact fract_div_nat (1 / t) n hn

/-- The finite summation-by-parts identity. The last term must be retained
before passing to an infinite arithmetic sum. -/
theorem finite_abel_boundary (a F : ℕ → ℝ) (L : ℕ) :
    (∑ j ∈ Finset.range (L + 1), a (j + 1) * (F (j + 1) - F (j + 2))) =
      a 1 * F 1 +
        (∑ j ∈ Finset.range L, (a (j + 2) - a (j + 1)) * F (j + 2)) -
        a (L + 1) * F (L + 2) := by
  induction L with
  | zero => simp; ring
  | succ L ih =>
    rw [Finset.sum_range_succ, ih, Finset.sum_range_succ]
    ring

end NymanBeurling.Annihilator

#print axioms NymanBeurling.Annihilator.projection_residual_correlation
#print axioms NymanBeurling.Annihilator.projection_residual_ne_zero_of_witness
#print axioms NymanBeurling.Annihilator.fract_div_nat
#print axioms NymanBeurling.Annihilator.dilation_residue
#print axioms NymanBeurling.Annihilator.finite_abel_boundary
