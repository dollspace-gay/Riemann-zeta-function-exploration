import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic

/-! Algebraic correction identities and the exact convergence implication.

The step inequality is an explicit hypothesis. This file does not assert that
the Nyman--Beurling residuals satisfy it.
-/

namespace NymanBeurling.Refinement

open Filter
open scoped Topology RealInnerProductSpace

theorem correction_identity {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (r f : H) (t : ℝ) :
    ⟪r - t • f, r - t • f⟫ = ⟪r, r⟫ - 2 * t * ⟪r, f⟫ + t ^ 2 * ⟪f, f⟫ := by
  simp only [inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right, real_inner_comm f r]
  ring

/-- The exact best scalar correction in any nonzero direction. -/
theorem best_scalar_correction {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (r f : H) (hf : (⟪f, f⟫ : ℝ) ≠ 0) :
    ⟪r - (⟪r, f⟫ / ⟪f, f⟫) • f, r - (⟪r, f⟫ / ⟪f, f⟫) • f⟫ =
      ⟪r, r⟫ - ⟪r, f⟫ ^ 2 / ⟪f, f⟫ := by
  rw [correction_identity]
  field_simp
  ring

/-- Removing an old-span component preserves correlation with an orthogonal residual
and can only increase the best scalar gain. The orthogonality hypotheses are explicit. -/
theorem orthogonal_cleanup_gain {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (r old fresh : H) (hr : (⟪r, old⟫ : ℝ) = 0)
    (ho : (⟪old, fresh⟫ : ℝ) = 0) (hf : fresh ≠ 0) :
    ⟪r, old + fresh⟫ ^ 2 / ⟪old + fresh, old + fresh⟫ ≤
      ⟪r, fresh⟫ ^ 2 / ⟪fresh, fresh⟫ := by
  have ho' : (⟪fresh, old⟫ : ℝ) = 0 := by rw [real_inner_comm]; exact ho
  have hnonneg : 0 ≤ (⟪old, old⟫ : ℝ) := real_inner_self_nonneg
  have hpos : 0 < (⟪fresh, fresh⟫ : ℝ) := real_inner_self_pos.mpr hf
  simp only [inner_add_left, inner_add_right, hr, ho, ho', zero_add, add_zero]
  apply (div_le_div_iff₀ (by linarith : 0 < (⟪old, old⟫ : ℝ) + ⟪fresh, fresh⟫)
    hpos).mpr
  nlinarith [mul_nonneg (sq_nonneg (⟪r, fresh⟫ : ℝ)) hnonneg]

/-- The trial-to-optimal correlation discrepancy is controlled by the part of
the correction lying in the old space. This is an upper error bound, not a
positive lower bound on the actual residual correlation. -/
theorem trial_correlation_transfer {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (r delta old fresh : H) (hr : (⟪r, old⟫ : ℝ) = 0)
    (hd : (⟪delta, fresh⟫ : ℝ) = 0) :
    |⟪r + delta, old + fresh⟫ - ⟪r, fresh⟫| ≤ ‖delta‖ * ‖old‖ := by
  have identity : (⟪r + delta, old + fresh⟫ - ⟪r, fresh⟫ : ℝ) = ⟪delta, old⟫ := by
    simp only [inner_add_left, inner_add_right, hr, hd]
    ring
  rw [identity]
  exact abs_real_inner_le_norm delta old

/-- The excess squared energy of a trial residual is the squared distance
from the optimal approximant, when its error is orthogonal to that difference. -/
theorem trial_excess_energy {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (r delta : H) (h : (⟪r, delta⟫ : ℝ) = 0) :
    (⟪r + delta, r + delta⟫ : ℝ) - ⟪r, r⟫ = ⟪delta, delta⟫ := by
  have h' : (⟪delta, r⟫ : ℝ) = 0 := by rw [real_inner_comm]; exact h
  simp only [inner_add_left, inner_add_right, h, h']
  ring

/-- A correction cannot remove a component orthogonal to it. This bounds gain
by visible energy, not by the energy of the entire residual. -/
theorem gain_le_visible_energy {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (blind visible g : H) (hb : (⟪blind, g⟫ : ℝ) = 0) (hg : g ≠ 0) :
    ⟪blind + visible, g⟫ ^ 2 / ⟪g, g⟫ ≤ ⟪visible, visible⟫ := by
  simp only [inner_add_left, hb, zero_add]
  apply (div_le_iff₀ (real_inner_self_pos.mpr hg)).mpr
  simpa only [pow_two] using real_inner_mul_inner_self_le visible g

/-- Capturing at least half of E forces a large correction norm if only a
small visible component can correlate. No RH or arithmetic hypothesis is used. -/
theorem cost_lower_of_half_capture {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (blind visible g : H) (E : ℝ) (hE : 0 ≤ E)
    (hb : (⟪blind, g⟫ : ℝ) = 0) (hv : 0 < (⟪visible, visible⟫ : ℝ))
    (hcap : E / 2 ≤ |(⟪blind + visible, g⟫ : ℝ)|) :
    E ^ 2 / (4 * ⟪visible, visible⟫) ≤ ⟪g, g⟫ := by
  simp only [inner_add_left, hb, zero_add] at hcap
  have hcs := real_inner_mul_inner_self_le visible g
  have habs := sq_abs (⟪visible, g⟫ : ℝ)
  have hsq : E ^ 2 ≤ 4 * (⟪visible, g⟫ : ℝ) ^ 2 := by
    nlinarith [abs_nonneg (⟪visible, g⟫ : ℝ)]
  apply (div_le_iff₀ (by positivity : 0 < 4 * (⟪visible, visible⟫ : ℝ))).mpr
  nlinarith

/-- A guaranteed fractional decrease has an exponential cumulative bound. -/
theorem energy_le_exp_neg_sum (E a : ℕ → ℝ)
    (hE : ∀ n, 0 ≤ E n)
    (hstep : ∀ n, E (n + 1) ≤ (1 - a n) * E n) (n : ℕ) :
    E n ≤ E 0 * Real.exp (-(∑ j ∈ Finset.range n, a j)) := by
  induction n with
  | zero => simp
  | succ n ih =>
    calc
      E (n + 1) ≤ (1 - a n) * E n := hstep n
      _ ≤ Real.exp (-a n) * E n := by
        apply mul_le_mul_of_nonneg_right _ (hE n)
        have := Real.add_one_le_exp (-a n)
        linarith
      _ ≤ Real.exp (-a n) * (E 0 * Real.exp (-(∑ j ∈ Finset.range n, a j))) :=
        mul_le_mul_of_nonneg_left ih (Real.exp_pos _).le
      _ = E 0 * Real.exp (-(∑ j ∈ Finset.range (n + 1), a j)) := by
        rw [Finset.sum_range_succ, neg_add, Real.exp_add]
        ring

/-- Persistent fractional gains force the remaining energy to zero.

No assertion about RH, primes, or the existence of such gains is an input
hidden in a definition: the required step bound and divergence are explicit.
-/
theorem tendsto_zero_of_divergent_gains (E a : ℕ → ℝ)
    (hE : ∀ n, 0 ≤ E n)
    (hstep : ∀ n, E (n + 1) ≤ (1 - a n) * E n)
    (hdiv : Tendsto (fun n => ∑ j ∈ Finset.range n, a j) atTop atTop) :
    Tendsto E atTop (𝓝 0) := by
  have hexp : Tendsto (fun n => Real.exp (-(∑ j ∈ Finset.range n, a j))) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp hdiv)
  have hbound : Tendsto (fun n => E 0 * Real.exp (-(∑ j ∈ Finset.range n, a j)))
      atTop (𝓝 0) := by simpa using hexp.const_mul (E 0)
  exact squeeze_zero hE (energy_le_exp_neg_sum E a hE hstep) hbound

end NymanBeurling.Refinement

#print axioms NymanBeurling.Refinement.best_scalar_correction
#print axioms NymanBeurling.Refinement.orthogonal_cleanup_gain
#print axioms NymanBeurling.Refinement.trial_correlation_transfer
#print axioms NymanBeurling.Refinement.trial_excess_energy
#print axioms NymanBeurling.Refinement.gain_le_visible_energy
#print axioms NymanBeurling.Refinement.cost_lower_of_half_capture
#print axioms NymanBeurling.Refinement.energy_le_exp_neg_sum
#print axioms NymanBeurling.Refinement.tendsto_zero_of_divergent_gains
