import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Tactic.FunProp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Exact local analytic estimate used in the jump lower bound. -/

namespace NymanBeurling.Jump

open MeasureTheory

theorem integral_affine_sq (a b l r : ℝ) :
    (∫ u in l..r, (b + a * u) ^ 2) =
      b ^ 2 * (r - l) + a * b * (r ^ 2 - l ^ 2) +
        a ^ 2 * (r ^ 3 - l ^ 3) / 3 := by
  have hd (x : ℝ) : HasDerivAt
      (fun u : ℝ => b ^ 2 * u + a * b * u ^ 2 + a ^ 2 * u ^ 3 / 3)
      ((b + a * x) ^ 2) x := by
    convert! (((hasDerivAt_id x).const_mul (b ^ 2)).add
      (((hasDerivAt_id x).pow 2).const_mul (a * b))).add
      ((((hasDerivAt_id x).pow 3).const_mul (a ^ 2)).div_const 3) using 1
    norm_num
    ring
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => hd x)
    ((by fun_prop : Continuous (fun u : ℝ => (b + a * u) ^ 2)).intervalIntegrable l r)]
  ring

/-- The exact square completion is valid for a jump at any real centre. -/
theorem local_square_completion (a b s n : ℝ) :
    (∫ u in (n - 1 / 2)..n, (b + a * u) ^ 2) +
      (∫ u in n..(n + 1 / 2), (b + a * u - s) ^ 2) =
        (b + a * n - s / 2) ^ 2 + (a - 3 * s / 2) ^ 2 / 12 + s ^ 2 / 16 := by
  have h (u : ℝ) : b + a * u - s = (b - s) + a * u := by ring
  simp_rw [h]
  rw [integral_affine_sq, integral_affine_sq]
  ring

/-- Every affine jump costs at least its squared size divided by sixteen. -/
theorem local_jump_energy_le (a b s n : ℝ) :
    s ^ 2 / 16 ≤ (∫ u in (n - 1 / 2)..n, (b + a * u) ^ 2) +
      (∫ u in n..(n + 1 / 2), (b + a * u - s) ^ 2) := by
  rw [local_square_completion]
  nlinarith [sq_nonneg (b + a * n - s / 2), sq_nonneg (a - 3 * s / 2)]

/-- The complex case splits into its real and imaginary parts. -/
theorem complex_local_square_completion (a b s : ℂ) :
    Complex.normSq b + Complex.normSq a / 12 + Complex.normSq s / 2 -
      (b * star s).re - (a * star s).re / 4 =
        Complex.normSq (b - s / 2) + Complex.normSq (a - 3 * s / 2) / 12 +
          Complex.normSq s / 16 := by
  simp [Complex.normSq_apply, Complex.mul_re]
  ring

end NymanBeurling.Jump

#print axioms NymanBeurling.Jump.local_square_completion
#print axioms NymanBeurling.Jump.local_jump_energy_le
#print axioms NymanBeurling.Jump.complex_local_square_completion
