/-
  Nyman-Beurling dilation family: e_k ∈ L²(0,∞)   (Plan 4, Milestone 1)
  ====================================================================
  The dilation functions e_k(t) = fract(1/(k·t)) of the Nyman–Beurling
  criterion, as square-integrable functions on (0,∞).

  PROVEN HERE (no sorry):
  ✓ eDil is measurable
  ✓ 0 ≤ eDil k t < 1, and eDil k t = 1/(k·t) for t > 1/k
  ✓ (eDil k)² is integrable on Ioi 0
  ✓ MemLp (eDil k) 2 (volume.restrict (Ioi 0))  — the L² statement

  Foundation for Milestones 2–4 (square-wave identity in L², witness
  norm = disagreement integral, Rayleigh assembly of NB Theorem 1).
  Reuses `NymanBeurling.integrableOn_inv_sq_Ioi` from the June module.
-/

import Mathlib.Tactic
import Mathlib.MeasureTheory.Function.Floor
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import NymanBeurling

open MeasureTheory Set

namespace NymanBeurling

noncomputable section

/-- The Nyman–Beurling dilation function e_k(t) = {1/(k·t)}. -/
def eDil (k : ℕ) : ℝ → ℝ := fun t => Int.fract (1 / (k * t))

lemma eDil_measurable (k : ℕ) : Measurable (eDil k) :=
  (measurable_const.div (measurable_const.mul measurable_id)).fract

lemma eDil_nonneg (k : ℕ) (t : ℝ) : 0 ≤ eDil k t := Int.fract_nonneg _

lemma eDil_lt_one (k : ℕ) (t : ℝ) : eDil k t < 1 := Int.fract_lt_one _

/-- For t > 1/k (k ≥ 1), the dilation is the pure power law 1/(k·t). -/
lemma eDil_eq_inv (k : ℕ) (hk : 1 ≤ k) {t : ℝ} (ht : 1 / (k : ℝ) < t) :
    eDil k t = 1 / (k * t) := by
  have hk0 : (0 : ℝ) < k := by exact_mod_cast hk
  have ht0 : 0 < t := lt_trans (by positivity) ht
  have h1 : 1 / (k * t) < 1 := by
    rw [div_lt_one (by positivity)]
    calc (1 : ℝ) = k * (1 / k) := by field_simp
    _ < k * t := mul_lt_mul_of_pos_left ht hk0
  have h0 : 0 ≤ 1 / ((k : ℝ) * t) := by positivity
  exact Int.fract_eq_self.mpr ⟨h0, h1⟩

/-- The dominating function for (e_k)²: 1 on (0, 1/k], and t⁻² beyond
(using k ≥ 1, so (k·t)⁻² ≤ t⁻²). -/
def dom (k : ℕ) : ℝ → ℝ := fun t =>
  (Ioc (0:ℝ) (1 / k)).indicator (fun _ => (1:ℝ)) t
  + (Ioi (1 / (k:ℝ))).indicator (fun s => (s ^ 2)⁻¹) t

lemma eDil_sq_le_dom (k : ℕ) (hk : 1 ≤ k) {t : ℝ} (ht : 0 < t) :
    (eDil k t) ^ 2 ≤ dom k t := by
  have hk1 : (1 : ℝ) ≤ k := by exact_mod_cast hk
  unfold dom
  rcases le_or_gt t (1 / k) with hle | hgt
  · have h1 : (Ioc (0:ℝ) (1 / k)).indicator (fun _ => (1:ℝ)) t = 1 :=
      indicator_of_mem (mem_Ioc.mpr ⟨ht, hle⟩) _
    have h2 : (0:ℝ) ≤ (Ioi (1 / (k:ℝ))).indicator (fun s => (s ^ 2)⁻¹) t :=
      indicator_nonneg (fun s _ => by positivity) t
    nlinarith [eDil_nonneg k t, eDil_lt_one k t]
  · have h1 : (Ioc (0:ℝ) (1 / k)).indicator (fun _ => (1:ℝ)) t = 0 :=
      indicator_of_notMem (fun h => absurd (mem_Ioc.mp h).2 (not_le.mpr hgt)) _
    have h2 : (Ioi (1 / (k:ℝ))).indicator (fun s => (s ^ 2)⁻¹) t = (t ^ 2)⁻¹ :=
      indicator_of_mem (mem_Ioi.mpr hgt) _
    rw [h1, h2, eDil_eq_inv k hk hgt]
    have hkt : t ≤ (k : ℝ) * t := le_mul_of_one_le_left ht.le hk1
    have hsq : t ^ 2 ≤ ((k:ℝ) * t) ^ 2 := by nlinarith
    have hpos : (0:ℝ) < t ^ 2 := by positivity
    have key : 1 / ((k:ℝ) * t) ^ 2 ≤ 1 / t ^ 2 :=
      one_div_le_one_div_of_le hpos hsq
    simp only [one_div] at key
    simp only [one_div, inv_pow, zero_add]
    exact key

lemma dom_integrable (k : ℕ) (hk : 1 ≤ k) :
    IntegrableOn (dom k) (Ioi (0:ℝ)) := by
  have hk0 : (0 : ℝ) < k := by exact_mod_cast hk
  unfold dom
  apply Integrable.add
  · rw [integrable_indicator_iff measurableSet_Ioc]
    have hfin : (volume.restrict (Ioi (0:ℝ))) (Ioc (0:ℝ) (1 / k)) ≠ ⊤ := by
      rw [Measure.restrict_apply measurableSet_Ioc]
      exact ne_top_of_le_ne_top
        (by rw [Real.volume_Ioc]; exact ENNReal.ofReal_ne_top)
        (measure_mono inter_subset_left)
    exact integrableOn_const hfin
  · rw [integrable_indicator_iff measurableSet_Ioi]
    have base : IntegrableOn (fun s : ℝ => (s ^ 2)⁻¹)
        (Ioi (1 / (k:ℝ)) ∩ Ioi 0) volume :=
      (integrableOn_inv_sq_Ioi (show (0:ℝ) < 1 / k by positivity)).mono_set
        inter_subset_left
    simpa [IntegrableOn, Measure.restrict_restrict measurableSet_Ioi]
      using base

/-- **e_k is square-integrable on (0, ∞).** -/
theorem eDil_sq_integrable (k : ℕ) (hk : 1 ≤ k) :
    IntegrableOn (fun t => (eDil k t) ^ 2) (Ioi (0:ℝ)) := by
  apply Integrable.mono' (dom_integrable k hk)
  · exact ((eDil_measurable k).pow_const 2).aestronglyMeasurable
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    exact eDil_sq_le_dom k hk ht

/-- **MILESTONE 1: the dilation family lives in L²(0, ∞).** -/
theorem eDil_memLp_two (k : ℕ) (hk : 1 ≤ k) :
    MemLp (eDil k) 2 (volume.restrict (Ioi (0:ℝ))) := by
  rw [memLp_two_iff_integrable_sq
    ((eDil_measurable k).aestronglyMeasurable.restrict)]
  exact eDil_sq_integrable k hk

end

end NymanBeurling

#print axioms NymanBeurling.eDil_memLp_two
