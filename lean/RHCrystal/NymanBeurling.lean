/-
  Nyman-Beurling: the disagreement-integral bound behind Theorem 1
  =================================================================
  Paper: nyman-beurling/chains_theory.md §0 (June 2026).

  FORMALIZED HERE: with J ≥ 2 (J = K−1 in the paper) and
  D = {u > 0 : ⌊u/J⌋ − ⌊u/(J+1)⌋ odd} the parity-disagreement set of
  adjacent dilation counters,

      ∫_{D} u⁻² du ≤ (H_J + 1) / (J(J+1)),

  where H_J is the J-th harmonic number. This is the analytic core of
  Theorem 1 (λ_min(G_N) ≤ (H_{K−1}+1)/(10·K(K−1))): the steps remaining
  on paper are the L² bilinearity bookkeeping ‖f_{K−1} − f_K‖² = P/8 and
  the Rayleigh principle (already formalized in RHCrystal.lean as
  `exists_eigenvalue_mul_le_rayleigh`).
-/

import Mathlib.Tactic
import Mathlib.MeasureTheory.Function.Floor
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

open MeasureTheory Set Function

namespace NymanBeurling

noncomputable section

/-- Harmonic number H_m = Σ_{ℓ=1}^m 1/ℓ. -/
def harmonic (m : ℕ) : ℝ := ∑ ℓ ∈ Finset.Icc 1 m, (1 : ℝ) / ℓ

/-- The parity-disagreement set of the adjacent counters ⌊u/J⌋, ⌊u/(J+1)⌋. -/
def D (J : ℕ) : Set ℝ := {u | 0 < u ∧ Odd (⌊u / (J : ℝ)⌋ - ⌊u / (J + 1 : ℝ)⌋)}

variable {J : ℕ}

/-! ### §1 Floor arithmetic -/

lemma floor_div_succ_le (hJ : 1 ≤ J) {u : ℝ} (hu : 0 ≤ u) :
    ⌊u / (J + 1 : ℝ)⌋ ≤ ⌊u / (J : ℝ)⌋ := by
  have hJ0 : (0 : ℝ) < J := by exact_mod_cast hJ
  apply Int.floor_mono
  gcongr
  · linarith

/-- **Disagreement-interval lemma.** If the counters disagree in parity at
u ∈ (0, J(J+1)), then u ∈ [Jℓ, (J+1)ℓ) for some 1 ≤ ℓ ≤ J. -/
lemma interval_of_odd (hJ : 2 ≤ J) {u : ℝ} (hu : 0 < u)
    (hu2 : u < (J : ℝ) * (J + 1))
    (hodd : Odd (⌊u / (J : ℝ)⌋ - ⌊u / (J + 1 : ℝ)⌋)) :
    ∃ ℓ : ℕ, 1 ≤ ℓ ∧ ℓ ≤ J ∧ (J : ℝ) * ℓ ≤ u ∧ u < ((J : ℝ) + 1) * ℓ := by
  have hJ0 : (0 : ℝ) < J := by
    have : (2 : ℝ) ≤ J := by exact_mod_cast hJ
    linarith
  have hJ1 : (0 : ℝ) < (J : ℝ) + 1 := by linarith
  set n1 := ⌊u / (J : ℝ)⌋ with hn1
  set n2 := ⌊u / (J + 1 : ℝ)⌋ with hn2
  have h21 : n2 ≤ n1 := floor_div_succ_le (by omega) hu.le
  have hn2nn : 0 ≤ n2 := Int.floor_nonneg.mpr (by positivity)
  have hfl1 : (n1 : ℝ) ≤ u / J := Int.floor_le _
  have hfl2 : u / (J + 1 : ℝ) < (n2 : ℝ) + 1 := by
    have := Int.lt_floor_add_one (u / (J + 1 : ℝ))
    linarith
  have hgap : u / (J : ℝ) - u / ((J : ℝ) + 1) < 1 := by
    have heq : u / (J : ℝ) - u / ((J : ℝ) + 1) = u / ((J : ℝ) * ((J : ℝ) + 1)) := by
      field_simp
      ring
    rw [heq, div_lt_one (by positivity)]
    linarith
  have hd2 : (n1 : ℝ) - (n2 : ℝ) < 2 := by
    linarith [hfl1, hfl2, hgap]
  have hdle : n1 - n2 ≤ 1 := by
    have : (n1 - n2 : ℤ) < 2 := by exact_mod_cast hd2
    omega
  obtain ⟨c, hc⟩ := hodd
  have hdiff : n1 - n2 = 1 := by omega
  have hn1pos : 1 ≤ n1 := by omega
  refine ⟨n1.toNat, by omega, ?_, ?_, ?_⟩
  · -- n1 ≤ J since n1 ≤ u/J < J+1
    have hlt : (n1 : ℝ) < (J : ℝ) + 1 := by
      apply lt_of_le_of_lt hfl1
      rw [div_lt_iff₀ hJ0]
      nlinarith
    have : n1 < (J : ℤ) + 1 := by exact_mod_cast hlt
    omega
  · -- J·ℓ ≤ u
    have hle : (J : ℝ) * (n1 : ℝ) ≤ u := by
      have := (le_div_iff₀ hJ0).mp hfl1
      linarith
    have hcast : ((n1.toNat : ℕ) : ℝ) = (n1 : ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg (by omega)
    rw [hcast]
    exact hle
  · -- u < (J+1)·ℓ
    have hlt : ⌊u / (J + 1 : ℝ)⌋ < n1 := by omega
    have hdiv : u / (J + 1 : ℝ) < (n1 : ℝ) := by
      have := Int.floor_lt.mp hlt
      linarith
    have hu' : u < ((J : ℝ) + 1) * (n1 : ℝ) := by
      rw [div_lt_iff₀ (by linarith : (0:ℝ) < (J + 1 : ℝ))] at hdiv
      linarith
    have hcast : ((n1.toNat : ℕ) : ℝ) = (n1 : ℝ) := by
      exact_mod_cast Int.toNat_of_nonneg (by omega)
    rw [hcast]
    exact hu'

/-- Elements of D are at least J. -/
lemma D_subset_Ici (hJ : 2 ≤ J) : D J ⊆ Ici (J : ℝ) := by
  rintro u ⟨hu, hodd⟩
  by_contra hlt
  simp only [mem_Ici, not_le] at hlt
  have hJ0 : (0 : ℝ) < J := by
    have : (2 : ℝ) ≤ J := by exact_mod_cast hJ
    linarith
  have hn1 : ⌊u / (J : ℝ)⌋ = 0 := by
    rw [Int.floor_eq_zero_iff]
    refine ⟨by positivity, ?_⟩
    rw [div_lt_one hJ0]
    exact hlt
  have hn2 : ⌊u / (J + 1 : ℝ)⌋ = 0 := by
    rw [Int.floor_eq_zero_iff]
    have hJ1 : (0 : ℝ) < (J : ℝ) + 1 := by linarith
    refine ⟨by positivity, ?_⟩
    have : u / (J + 1 : ℝ) ≤ u / (J : ℝ) := by
      gcongr
      all_goals linarith
    have h2 : u / (J : ℝ) < 1 := by rw [div_lt_one hJ0]; exact hlt
    linarith
  rw [hn1, hn2] at hodd
  simp at hodd

/-- D is measurable. -/
lemma D_measurable : MeasurableSet (D J) := by
  have h1 : Measurable fun u : ℝ => ⌊u / (J : ℝ)⌋ - ⌊u / (J + 1 : ℝ)⌋ :=
    ((measurable_id.div_const _).floor).sub ((measurable_id.div_const _).floor)
  have h2 : MeasurableSet {n : ℤ | Odd n} := MeasurableSet.of_discrete
  exact measurableSet_Ioi.inter (h1 h2)

/-! ### §2 Integrals -/

/-- ∫ over [a,b) of x⁻² is a⁻¹ − b⁻¹. -/
lemma integral_Ico_inv_sq {a b : ℝ} (ha : 0 < a) (hab : a ≤ b) :
    ∫ x in Ico a b, (x ^ 2)⁻¹ = a⁻¹ - b⁻¹ := by
  rw [MeasureTheory.integral_Ico_eq_integral_Ioc,
      ← intervalIntegral.integral_of_le hab]
  have key : ∀ x ∈ uIcc a b, HasDerivAt (fun y : ℝ => -y⁻¹) ((x ^ 2)⁻¹) x := by
    intro x hx
    rw [Set.uIcc_of_le hab] at hx
    have hx0 : x ≠ 0 := ne_of_gt (lt_of_lt_of_le ha hx.1)
    have h := (hasDerivAt_inv hx0).neg
    rw [neg_neg] at h
    exact h
  have hint : IntervalIntegrable (fun x : ℝ => (x ^ 2)⁻¹) volume a b := by
    apply ContinuousOn.intervalIntegrable
    apply ContinuousOn.inv₀ (by fun_prop)
    intro x hx
    rw [Set.uIcc_of_le hab] at hx
    have : 0 < x := lt_of_lt_of_le ha hx.1
    positivity
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt key hint]
  ring

lemma inv_sq_eq_rpow {c : ℝ} (hc : 0 < c) :
    ∀ x ∈ Ioi c, (x ^ 2)⁻¹ = x ^ (-2 : ℝ) := by
  intro x hx
  have hx0 : (0 : ℝ) < x := lt_trans hc hx
  rw [show (-2 : ℝ) = -((2 : ℕ) : ℝ) by norm_num, Real.rpow_neg hx0.le,
      Real.rpow_natCast]

/-- The tail integral: ∫ over (c, ∞) of x⁻² is c⁻¹. -/
lemma integral_Ioi_inv_sq {c : ℝ} (hc : 0 < c) :
    ∫ x in Ioi c, (x ^ 2)⁻¹ = c⁻¹ := by
  rw [setIntegral_congr_fun measurableSet_Ioi (inv_sq_eq_rpow hc),
      integral_Ioi_rpow_of_lt (by norm_num) hc]
  rw [show (-2 : ℝ) + 1 = -1 by norm_num, Real.rpow_neg_one]
  field_simp

lemma integrableOn_inv_sq_Ioi {c : ℝ} (hc : 0 < c) :
    IntegrableOn (fun x : ℝ => (x ^ 2)⁻¹) (Ioi c) := by
  rw [integrableOn_congr_fun (inv_sq_eq_rpow hc) measurableSet_Ioi]
  exact integrableOn_Ioi_rpow_of_lt (by norm_num) hc

/-! ### §3 Assembly -/

/-- **The disagreement-integral bound** (analytic core of Theorem 1). -/
theorem disagreement_integral_le (hJ : 2 ≤ J) :
    ∫ u in D J, (u ^ 2)⁻¹ ≤ (harmonic J + 1) / ((J : ℝ) * ((J : ℝ) + 1)) := by
  have hJR : (2 : ℝ) ≤ J := by exact_mod_cast hJ
  have hJ0 : (0 : ℝ) < J := by linarith
  have hJ1 : (0 : ℝ) < (J : ℝ) + 1 := by linarith
  set Ustar : ℝ := (J : ℝ) * ((J : ℝ) + 1) with hU
  have hU0 : 0 < Ustar := by positivity
  have hnn : ∀ u : ℝ, 0 ≤ (u ^ 2)⁻¹ := fun u => by positivity
  have hDmeas : MeasurableSet (D J) := D_measurable
  have hDIci : D J ⊆ Ici (J : ℝ) := D_subset_Ici hJ
  have hIciIoi : Ici (J : ℝ) ⊆ Ioi ((J : ℝ) / 2) := by
    intro u hu
    simp only [mem_Ici] at hu
    simp only [mem_Ioi]
    linarith
  have hint_total : IntegrableOn (fun u : ℝ => (u ^ 2)⁻¹) (D J) :=
    (integrableOn_inv_sq_Ioi (by positivity : (0:ℝ) < (J : ℝ)/2)).mono_set
      (subset_trans hDIci hIciIoi)
  -- split D at U*
  have hsplit : D J = (D J ∩ Ioo 0 Ustar) ∪ (D J ∩ Ici Ustar) := by
    ext u
    constructor
    · intro hu
      rcases lt_trichotomy u Ustar with h | h | h
      · exact Or.inl ⟨hu, hu.1, h⟩
      · exact Or.inr ⟨hu, h.symm.le⟩
      · exact Or.inr ⟨hu, h.le⟩
    · rintro (⟨hu, -⟩ | ⟨hu, -⟩) <;> exact hu
  have hdisj : Disjoint (D J ∩ Ioo 0 Ustar) (D J ∩ Ici Ustar) := by
    apply Set.disjoint_left.mpr
    rintro u ⟨-, -, hu1⟩ ⟨-, hu2⟩
    simp only [mem_Ici] at hu2
    linarith
  have hint1 : IntegrableOn (fun u : ℝ => (u ^ 2)⁻¹) (D J ∩ Ioo 0 Ustar) :=
    hint_total.mono_set inter_subset_left
  have hint2 : IntegrableOn (fun u : ℝ => (u ^ 2)⁻¹) (D J ∩ Ici Ustar) :=
    hint_total.mono_set inter_subset_left
  -- HEAD: covered by the disagreement intervals
  have hcover : D J ∩ Ioo 0 Ustar ⊆
      ⋃ ℓ ∈ Finset.Icc 1 J, Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ) := by
    rintro u ⟨⟨hu, hodd⟩, -, hub⟩
    obtain ⟨ℓ, hℓ1, hℓJ, hlo, hhi⟩ := interval_of_odd hJ hu (by linarith) hodd
    simp only [mem_iUnion, exists_prop]
    exact ⟨ℓ, Finset.mem_Icc.mpr ⟨hℓ1, hℓJ⟩, hlo, hhi⟩
  -- each interval is integrable, intervals are pairwise disjoint
  have hico_int : ∀ ℓ ∈ Finset.Icc 1 J, IntegrableOn (fun u : ℝ => (u ^ 2)⁻¹)
      (Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ)) := by
    intro ℓ hℓ
    have hℓ1 : (1 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast (Finset.mem_Icc.mp hℓ).1
    apply (integrableOn_inv_sq_Ioi (by positivity : (0:ℝ) < (J : ℝ)/2)).mono_set
    intro u hu
    have := hu.1
    simp only [mem_Ioi]
    nlinarith
  have hpair : Set.Pairwise (↑(Finset.Icc 1 J))
      (Disjoint on fun ℓ : ℕ => Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ)) := by
    intro a ha b hb hab
    simp only [Finset.coe_Icc, mem_Icc] at ha hb
    have key : ∀ x y : ℕ, 1 ≤ x → x ≤ J → x < y →
        Disjoint (Ico ((J : ℝ) * x) (((J : ℝ) + 1) * x))
                 (Ico ((J : ℝ) * y) (((J : ℝ) + 1) * y)) := by
      intro x y hx1 hxJ hxy
      apply Set.disjoint_left.mpr
      rintro u ⟨-, hu2⟩ ⟨hu3, -⟩
      have hxR : (x : ℝ) ≤ J := by exact_mod_cast hxJ
      have hyR : (x : ℝ) + 1 ≤ (y : ℝ) := by exact_mod_cast hxy
      nlinarith
    rcases lt_or_gt_of_ne hab with h | h
    · exact key a b ha.1 ha.2 h
    · exact (key b a hb.1 hb.2 h).symm
  -- head ≤ harmonic sum
  have hhead : ∫ u in D J ∩ Ioo 0 Ustar, (u ^ 2)⁻¹ ≤
      harmonic J / ((J : ℝ) * ((J : ℝ) + 1)) := by
    have hunion_int : IntegrableOn (fun u : ℝ => (u ^ 2)⁻¹)
        (⋃ ℓ ∈ Finset.Icc 1 J, Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ)) := by
      rw [show (⋃ ℓ ∈ Finset.Icc 1 J, Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ)) =
          ⋃ ℓ ∈ (Finset.Icc 1 J : Finset ℕ), Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ) from rfl]
      exact integrableOn_finset_iUnion.mpr hico_int
    calc ∫ u in D J ∩ Ioo 0 Ustar, (u ^ 2)⁻¹
        ≤ ∫ u in ⋃ ℓ ∈ Finset.Icc 1 J, Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ),
            (u ^ 2)⁻¹ :=
          setIntegral_mono_set hunion_int
            (Filter.Eventually.of_forall fun u => hnn u)
            (HasSubset.Subset.eventuallyLE hcover)
      _ = ∑ ℓ ∈ Finset.Icc 1 J, ∫ u in Ico ((J : ℝ) * ℓ) (((J : ℝ) + 1) * ℓ),
            (u ^ 2)⁻¹ :=
          integral_biUnion_finset _ (fun ℓ _ => measurableSet_Ico) hpair hico_int
      _ = ∑ ℓ ∈ Finset.Icc 1 J, (((J : ℝ) * ℓ)⁻¹ - (((J : ℝ) + 1) * ℓ)⁻¹) := by
          apply Finset.sum_congr rfl
          intro ℓ hℓ
          have hℓ1 : (1 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast (Finset.mem_Icc.mp hℓ).1
          apply integral_Ico_inv_sq (by positivity)
          nlinarith
      _ = harmonic J / ((J : ℝ) * ((J : ℝ) + 1)) := by
          rw [harmonic, Finset.sum_div]
          apply Finset.sum_congr rfl
          intro ℓ hℓ
          have hℓ1 : (1 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast (Finset.mem_Icc.mp hℓ).1
          have hℓ0 : (0 : ℝ) < ℓ := by linarith
          field_simp
          ring
  -- tail ≤ 1/U*
  have htail : ∫ u in D J ∩ Ici Ustar, (u ^ 2)⁻¹ ≤ Ustar⁻¹ := by
    have hIci_int : IntegrableOn (fun u : ℝ => (u ^ 2)⁻¹) (Ici Ustar) :=
      (integrableOn_inv_sq_Ioi (by positivity : (0:ℝ) < Ustar/2)).mono_set
        (fun u hu => by
          simp only [mem_Ici] at hu
          simp only [mem_Ioi]
          linarith)
    calc ∫ u in D J ∩ Ici Ustar, (u ^ 2)⁻¹
        ≤ ∫ u in Ici Ustar, (u ^ 2)⁻¹ :=
          setIntegral_mono_set hIci_int
            (Filter.Eventually.of_forall fun u => hnn u)
            (HasSubset.Subset.eventuallyLE inter_subset_right)
      _ = ∫ u in Ioi Ustar, (u ^ 2)⁻¹ := MeasureTheory.integral_Ici_eq_integral_Ioi
      _ = Ustar⁻¹ := integral_Ioi_inv_sq hU0
  -- assemble
  calc ∫ u in D J, (u ^ 2)⁻¹
      = (∫ u in D J ∩ Ioo 0 Ustar, (u ^ 2)⁻¹) +
        (∫ u in D J ∩ Ici Ustar, (u ^ 2)⁻¹) := by
        conv_lhs => rw [hsplit]
        exact setIntegral_union hdisj (hDmeas.inter measurableSet_Ici) hint1 hint2
    _ ≤ harmonic J / ((J : ℝ) * ((J : ℝ) + 1)) + Ustar⁻¹ := add_le_add hhead htail
    _ = (harmonic J + 1) / ((J : ℝ) * ((J : ℝ) + 1)) := by
        rw [hU]
        field_simp

end

end NymanBeurling

#print axioms NymanBeurling.interval_of_odd
#print axioms NymanBeurling.disagreement_integral_le
