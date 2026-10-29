import JumpArithmetic
import JumpLocal
import NymanBeurlingL2
import Mathlib.MeasureTheory.Integral.IntegralEqImproper

/-! The fractional-part synthesis and its jumps on the positive half-line. -/

namespace NymanBeurling.Jump

open MeasureTheory Set

noncomputable def synthesis (N : ℕ) (c : ℕ → ℝ) (u : ℝ) : ℝ :=
  ∑ k ∈ Finset.Ioc 0 N, c k * Int.fract (u / k)

noncomputable def slope (N : ℕ) (c : ℕ → ℝ) : ℝ :=
  ∑ k ∈ Finset.Ioc 0 N, c k / k

noncomputable def floorSum (N : ℕ) (c : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k ∈ Finset.Ioc 0 N, c k * (n / k : ℕ)

lemma synthesis_on_unit (N n : ℕ) (c : ℕ → ℝ) {u : ℝ}
    (hu : (n : ℝ) ≤ u) (hu' : u < n + 1) :
    synthesis N c u = slope N c * u - floorSum N c n := by
  have hfloor : ⌊u⌋ = (n : ℤ) := Int.floor_eq_iff.mpr (by exact_mod_cast And.intro hu hu')
  unfold synthesis slope floorSum
  rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro k _
  rw [Int.fract, Int.floor_div_natCast, hfloor, ← Int.natCast_div]
  simp only [Int.cast_natCast]
  ring

lemma floorSum_jump (N n : ℕ) (c : ℕ → ℝ) (hn : 0 < n) (hnN : n ≤ N) :
    floorSum N c n = floorSum N c (n - 1) + divisorSum c n := by
  have hn' : n - 1 + 1 = n := by omega
  unfold floorSum divisorSum
  rw [← filter_Ioc_dvd N n hn hnN, Finset.sum_filter]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  have hdiv := Nat.succ_div (a := n - 1) (b := k)
  rw [hn'] at hdiv
  rw [hdiv]
  split_ifs <;> push_cast <;> ring

lemma synthesis_left (N n : ℕ) (c : ℕ → ℝ) (hn : 0 < n) {u : ℝ}
    (hu : u ∈ Ioo ((n : ℝ) - 1 / 2) n) :
    synthesis N c u = -floorSum N c (n - 1) + slope N c * u := by
  have hn' : ((n - 1 : ℕ) : ℝ) = n - 1 := by
    rw [Nat.cast_sub (by omega : 1 ≤ n)]
    norm_num
  rw [synthesis_on_unit N (n - 1) c (by rw [hn']; linarith [hu.1])
    (by rw [hn']; linarith [hu.2])]
  ring

lemma synthesis_right (N n : ℕ) (c : ℕ → ℝ) (hn : 0 < n) (hnN : n ≤ N) {u : ℝ}
    (hu : u ∈ Ioo (n : ℝ) (n + 1 / 2)) :
    synthesis N c u = -floorSum N c (n - 1) + slope N c * u - divisorSum c n := by
  rw [synthesis_on_unit N n c hu.1.le (by linarith [hu.2]), floorSum_jump N n c hn hnN]
  ring

noncomputable def dilationSynthesis (N : ℕ) (c : ℕ → ℝ) (t : ℝ) : ℝ :=
  ∑ k ∈ Finset.Ioc 0 N, c k * eDil k t

lemma dilationSynthesis_inv (N : ℕ) (c : ℕ → ℝ) (u : ℝ) :
    dilationSynthesis N c u⁻¹ = synthesis N c u := by
  unfold dilationSynthesis synthesis eDil
  congr 1
  ext k
  congr 1
  simp [div_eq_mul_inv, mul_comm]

lemma dilationSynthesis_memLp (N : ℕ) (c : ℕ → ℝ) :
    MemLp (dilationSynthesis N c) 2 (volume.restrict (Ioi (0 : ℝ))) := by
  apply memLp_finsetSum
  intro k hk
  exact (eDil_memLp_two k (by have := (Finset.mem_Ioc.mp hk).1; omega)).const_mul (c k)

noncomputable def weightedEnergy (N : ℕ) (c : ℕ → ℝ) (u : ℝ) : ℝ :=
  synthesis N c u ^ 2 / u ^ 2

lemma weightedEnergy_integrable (N : ℕ) (c : ℕ → ℝ) :
    IntegrableOn (weightedEnergy N c) (Ioi (0 : ℝ)) := by
  have hm := dilationSynthesis_memLp N c
  have hi := (memLp_two_iff_integrable_sq hm.aestronglyMeasurable).mp hm
  have ht := (integrableOn_Ioi_comp_rpow_iff
    (fun t => dilationSynthesis N c t ^ 2) (p := -1) (by norm_num)).mpr hi
  convert ht using 1
  ext u
  norm_num [weightedEnergy, Real.rpow_neg_one, Real.rpow_neg_natCast,
    dilationSynthesis_inv, div_eq_mul_inv, mul_comm]
  exact Or.inl (zpow_natCast u 2).symm

theorem weightedEnergy_integral_eq (N : ℕ) (c : ℕ → ℝ) :
    (∫ u in Ioi (0 : ℝ), weightedEnergy N c u) =
      ∫ t in Ioi (0 : ℝ), dilationSynthesis N c t ^ 2 := by
  have ht := integral_comp_rpow_Ioi (fun t => dilationSynthesis N c t ^ 2)
    (p := -1) (by norm_num)
  convert ht using 1
  congr 1
  ext u
  norm_num [weightedEnergy, Real.rpow_neg_one, Real.rpow_neg_natCast,
    dilationSynthesis_inv, div_eq_mul_inv, mul_comm]
  exact Or.inl (zpow_natCast u 2).symm

lemma weightedEnergy_intervalIntegrable (N : ℕ) (c : ℕ → ℝ) {a b : ℝ}
    (ha : 0 ≤ a) (hab : a ≤ b) :
    IntervalIntegrable (weightedEnergy N c) volume a b := by
  rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hab]
  exact (weightedEnergy_integrable N c).mono_set fun _ hu => lt_of_le_of_lt ha hu.1

lemma synthesis_sq_le_weighted (N n : ℕ) (c : ℕ → ℝ) (hn : 0 < n) {u : ℝ}
    (hu : u ∈ Ioo ((n : ℝ) - 1 / 2) (n + 1 / 2)) :
    synthesis N c u ^ 2 ≤ (9 * (n : ℝ) ^ 2 / 4) * weightedEnergy N c u := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hu0 : 0 < u := by linarith [hu.1]
  have huBound : u ≤ 3 * (n : ℝ) / 2 := by linarith [hu.2]
  have hw : 0 ≤ weightedEnergy N c u := by unfold weightedEnergy; positivity
  have hid : synthesis N c u ^ 2 = u ^ 2 * weightedEnergy N c u := by
    unfold weightedEnergy
    field_simp
  rw [hid]
  apply mul_le_mul_of_nonneg_right _ hw
  nlinarith

theorem jump_cell_bound (N n : ℕ) (c : ℕ → ℝ) (hn : 0 < n) (hnN : n ≤ N) :
    (divisorSum c n / n) ^ 2 ≤
      36 * ∫ u in ((n : ℝ) - 1 / 2)..(n + 1 / 2), weightedEnergy N c u := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hil := weightedEnergy_intervalIntegrable N c
    (a := (n : ℝ) - 1 / 2) (b := n) (by linarith) (by linarith)
  have hir := weightedEnergy_intervalIntegrable N c
    (a := (n : ℝ)) (b := n + 1 / 2) (by linarith) (by linarith)
  have hl := intervalIntegral.integral_mono_on_of_le_Ioo (by linarith : (n : ℝ) - 1/2 ≤ n)
    ((by fun_prop : Continuous (fun u : ℝ =>
      (-floorSum N c (n-1) + slope N c * u)^2)).intervalIntegrable _ _)
    (hil.const_mul (9 * (n : ℝ)^2 / 4)) (fun u hu => by
      rw [← synthesis_left N n c hn hu]
      exact synthesis_sq_le_weighted N n c hn ⟨hu.1, by linarith [hu.2]⟩)
  have hr := intervalIntegral.integral_mono_on_of_le_Ioo (by linarith : (n : ℝ) ≤ n+1/2)
    ((by fun_prop : Continuous (fun u : ℝ =>
      (-floorSum N c (n-1) + slope N c * u - divisorSum c n)^2)).intervalIntegrable _ _)
    (hir.const_mul (9 * (n : ℝ)^2 / 4)) (fun u hu => by
      rw [← synthesis_right N n c hn hnN hu]
      exact synthesis_sq_le_weighted N n c hn ⟨by linarith [hu.1], hu.2⟩)
  rw [intervalIntegral.integral_const_mul] at hl hr
  have hj := local_jump_energy_le (slope N c) (-floorSum N c (n-1))
    (divisorSum c n) n
  have hadd := intervalIntegral.integral_add_adjacent_intervals hil hir
  have hbound : divisorSum c n ^ 2 ≤ 36 * (n : ℝ)^2 *
      ∫ u in ((n : ℝ) - 1/2)..(n+1/2), weightedEnergy N c u := by
    nlinarith
  rw [div_pow, div_le_iff₀ (by positivity : (0 : ℝ) < (n : ℝ)^2)]
  nlinarith [hbound]

def jumpCell (n : ℕ) : Set ℝ := Ioc ((n : ℝ) - 1 / 2) (n + 1 / 2)

lemma jumpCell_subset (n : ℕ) (hn : 0 < n) : jumpCell n ⊆ Ioi (0 : ℝ) := by
  intro u hu
  change 0 < u
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have := hu.1
  linarith

lemma jumpCells_disjoint : Pairwise (fun m n => Disjoint (jumpCell m) (jumpCell n)) := by
  intro m n hmn
  apply Set.disjoint_left.mpr
  intro u hu hv
  rcases lt_or_gt_of_ne hmn with h | h
  · have hmn' : (m : ℝ) + 1 ≤ n := by exact_mod_cast h
    have hu' := hu.2
    have hv' := hv.1
    linarith
  · have hnm' : (n : ℝ) + 1 ≤ m := by exact_mod_cast h
    have hu' := hu.1
    have hv' := hv.2
    linarith

theorem weighted_jumps_le_integral (N : ℕ) (c : ℕ → ℝ) :
    (∑ n ∈ Finset.Ioc 0 N, (divisorSum c n / n)^2) ≤
      36 * ∫ u in Ioi (0 : ℝ), weightedEnergy N c u := by
  have hi (n : ℕ) (hn : n ∈ Finset.Ioc 0 N) :
      IntegrableOn (weightedEnergy N c) (jumpCell n) :=
    (weightedEnergy_integrable N c).mono_set (jumpCell_subset n (Finset.mem_Ioc.mp hn).1)
  have hunion := integral_biUnion_finset (Finset.Ioc 0 N)
    (fun _ _ => (measurableSet_Ioc : MeasurableSet (jumpCell _)))
    (fun i _ j _ hij => jumpCells_disjoint hij) hi
  have hsub : (⋃ n ∈ Finset.Ioc 0 N, jumpCell n) ⊆ Ioi (0 : ℝ) := by
    intro u hu
    obtain ⟨n, hn, hu⟩ := Set.mem_iUnion₂.mp hu
    exact jumpCell_subset n (Finset.mem_Ioc.mp hn).1 hu
  have hmono := setIntegral_mono_set (weightedEnergy_integrable N c)
    (Filter.Eventually.of_forall fun u => by dsimp [weightedEnergy]; positivity)
    (Filter.Eventually.of_forall fun _ hu => hsub hu)
  calc
    _ ≤ ∑ n ∈ Finset.Ioc 0 N, 36 * ∫ u in jumpCell n, weightedEnergy N c u := by
      apply Finset.sum_le_sum
      intro n hn
      have h := jump_cell_bound N n c (Finset.mem_Ioc.mp hn).1 (Finset.mem_Ioc.mp hn).2
      rwa [intervalIntegral.integral_of_le (by linarith : (n : ℝ)-1/2 ≤ n+1/2)] at h
    _ = 36 * ∫ u in ⋃ n ∈ Finset.Ioc 0 N, jumpCell n, weightedEnergy N c u := by
      simpa only [jumpCell, Finset.mul_sum] using congrArg (fun x : ℝ => 36*x) hunion.symm
    _ ≤ _ := mul_le_mul_of_nonneg_left hmono (by norm_num)

/-- Unconditional lower bound for the actual real Nyman--Beurling synthesis.
The denominator is left multiplied out, so this also covers N = 0. -/
theorem synthesis_coefficient_bound (N : ℕ) (c : ℕ → ℝ) :
    (∑ n ∈ Finset.Ioc 0 N, c n ^ 2) ≤ 36 * (N : ℝ)^2 * harmonicSum N *
      ∫ t in Ioi (0 : ℝ), dilationSynthesis N c t ^ 2 := by
  have h := mul_le_mul_of_nonneg_left (weighted_jumps_le_integral N c)
    (mul_nonneg (sq_nonneg (N : ℝ)) (harmonicSum_nonneg N))
  have ha := coefficient_recovery_bound c N
  rw [weightedEnergy_integral_eq] at h
  nlinarith

lemma harmonicSum_pos {N : ℕ} (hN : 0 < N) : 0 < harmonicSum N := by
  apply Finset.sum_pos'
  · intro d _
    positivity
  · exact ⟨1, Finset.mem_Ioc.mpr ⟨by omega, by omega⟩, by norm_num⟩

/-- The complete real integral lower bound, with no analytic hypotheses. -/
theorem synthesis_integral_lower_bound (N : ℕ) (c : ℕ → ℝ) (hN : 0 < N) :
    (∑ n ∈ Finset.Ioc 0 N, c n ^ 2) / (36 * (N : ℝ)^2 * harmonicSum N) ≤
      ∫ t in Ioi (0 : ℝ), dilationSynthesis N c t ^ 2 := by
  have hh := harmonicSum_pos hN
  rw [div_le_iff₀ (by positivity)]
  nlinarith [synthesis_coefficient_bound N c]

end NymanBeurling.Jump

#print axioms NymanBeurling.Jump.synthesis_integral_lower_bound
