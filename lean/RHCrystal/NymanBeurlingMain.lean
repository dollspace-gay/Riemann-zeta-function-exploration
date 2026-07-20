/-
  Theorem 1 in t-coordinates: the witness norm IS a set measure
  =============================================================
  (Plan 4, Milestone 3.)

  KEY SIMPLIFICATION over the plan: under u = 1/t the weight u⁻²du is
  plain Lebesgue dt, so the disagreement integral is literally the
  MEASURE of the t-space disagreement set — no change of variables, no
  Jacobians. Everything reduces to interval arithmetic plus the June
  lemma `interval_of_odd`.

  PROVEN HERE (no sorry):
  ✓ Dt (the t-space parity-disagreement set) is measurable
  ✓ interval cover: Dt ⊆ (0, 1/(J(J+1))] ∪ ⋃_{ℓ≤J} (1/((J+1)ℓ), 1/(Jℓ)]
  ✓ volume (Dt J) ≤ ofReal ((H_J + 1)/(J(J+1)))
  ✓ pointwise: (chainDiff J − chainDiff (J+1))² = ¼·𝟙_{Dt J} on t > 0
  ✓ ∫_{(0,∞)} (chainDiff J − chainDiff (J+1))² = (volume (Dt J)).toReal/4
  ✓ MILESTONE 3:  ∫ w² ≤ (H_J + 1)/(4·J·(J+1))
-/

import Mathlib.Tactic
import NymanBeurling
import NymanBeurlingL2
import RHCrystal

open MeasureTheory Set

namespace NymanBeurling

noncomputable section

variable {J : ℕ}

/-- The t-space parity-disagreement set of adjacent chain counters. -/
def Dt (J : ℕ) : Set ℝ :=
  {t | 0 < t ∧ Odd (⌊1 / ((J : ℝ) * t)⌋ - ⌊1 / (((J : ℝ) + 1) * t)⌋)}

lemma Dt_measurable : MeasurableSet (Dt J) := by
  have h1 : Measurable fun t : ℝ =>
      ⌊1 / ((J : ℝ) * t)⌋ - ⌊1 / (((J : ℝ) + 1) * t)⌋ :=
    ((measurable_const.div (measurable_const.mul measurable_id)).floor).sub
      ((measurable_const.div (measurable_const.mul measurable_id)).floor)
  have h2 : MeasurableSet {n : ℤ | Odd n} := MeasurableSet.of_discrete
  exact measurableSet_Ioi.inter (h1 h2)

lemma Dt_subset_Ioi : Dt J ⊆ Ioi (0 : ℝ) := fun _ ht => ht.1

/-- **The interval cover** (via the June lemma `interval_of_odd`). -/
lemma Dt_cover (hJ : 2 ≤ J) :
    Dt J ⊆ Ioc 0 (1 / ((J : ℝ) * ((J : ℝ) + 1)))
      ∪ ⋃ ℓ ∈ Finset.Icc 1 J,
          Ioc (1 / (((J : ℝ) + 1) * ℓ)) (1 / ((J : ℝ) * ℓ)) := by
  have hJ0 : (0 : ℝ) < J := by
    have : (2 : ℝ) ≤ J := by exact_mod_cast hJ
    linarith
  have hJ1 : (0 : ℝ) < (J : ℝ) + 1 := by linarith
  rintro t ⟨ht, hodd⟩
  set u := 1 / t with hu
  have hu0 : 0 < u := by positivity
  have htu : t = 1 / u := by rw [hu]; field_simp
  have hcJ : 1 / ((J : ℝ) * t) = u / (J : ℝ) := by
    rw [hu, div_div, mul_comm]
  have hcJ1 : 1 / (((J : ℝ) + 1) * t) = u / ((J : ℝ) + 1) := by
    rw [hu, div_div, mul_comm]
  rw [hcJ, hcJ1] at hodd
  rcases lt_or_ge u ((J : ℝ) * ((J : ℝ) + 1)) with hlt | hge
  · obtain ⟨ℓ, hℓ1, hℓJ, hlo, hhi⟩ := interval_of_odd hJ hu0 hlt hodd
    right
    have hℓ0 : (0 : ℝ) < ℓ := by
      have : (1 : ℝ) ≤ (ℓ : ℝ) := by exact_mod_cast hℓ1
      linarith
    simp only [mem_iUnion, exists_prop]
    refine ⟨ℓ, Finset.mem_Icc.mpr ⟨hℓ1, hℓJ⟩, ?_, ?_⟩
    · rw [htu]
      exact one_div_lt_one_div_of_lt hu0 hhi
    · rw [htu]
      exact one_div_le_one_div_of_le (by positivity) hlo
  · left
    refine ⟨ht, ?_⟩
    rw [htu]
    exact one_div_le_one_div_of_le (by positivity) hge

/-- **The measure bound**: volume(Dt) ≤ (H_J + 1)/(J(J+1)). -/
theorem Dt_volume_le (hJ : 2 ≤ J) :
    volume (Dt J) ≤ ENNReal.ofReal ((harmonic J + 1) / ((J : ℝ) * ((J : ℝ) + 1))) := by
  have hJ0 : (0 : ℝ) < J := by
    have : (2 : ℝ) ≤ J := by exact_mod_cast hJ
    linarith
  have hJ1 : (0 : ℝ) < (J : ℝ) + 1 := by linarith
  have hharm : 0 ≤ harmonic J := by
    refine Finset.sum_nonneg fun ℓ hℓ => ?_
    positivity
  calc volume (Dt J)
      ≤ volume (Ioc 0 (1 / ((J : ℝ) * ((J : ℝ) + 1)))
          ∪ ⋃ ℓ ∈ Finset.Icc 1 J,
              Ioc (1 / (((J : ℝ) + 1) * ℓ)) (1 / ((J : ℝ) * ℓ))) :=
        measure_mono (Dt_cover hJ)
    _ ≤ volume (Ioc (0:ℝ) (1 / ((J : ℝ) * ((J : ℝ) + 1))))
        + ∑ ℓ ∈ Finset.Icc 1 J,
            volume (Ioc (1 / (((J : ℝ) + 1) * ℓ)) (1 / ((J : ℝ) * ℓ))) :=
        le_trans (measure_union_le _ _)
          (by gcongr; exact measure_biUnion_finset_le _ _)
    _ = ENNReal.ofReal (1 / ((J : ℝ) * ((J : ℝ) + 1)))
        + ∑ ℓ ∈ Finset.Icc 1 J,
            ENNReal.ofReal (1 / ((J : ℝ) * ℓ) - 1 / (((J : ℝ) + 1) * ℓ)) := by
        rw [Real.volume_Ioc, sub_zero]
        congr 1
        refine Finset.sum_congr rfl fun ℓ _ => ?_
        rw [Real.volume_Ioc]
    _ = ENNReal.ofReal ((harmonic J + 1) / ((J : ℝ) * ((J : ℝ) + 1))) := by
        rw [← ENNReal.ofReal_sum_of_nonneg, ← ENNReal.ofReal_add]
        · congr 1
          have hsum : ∑ ℓ ∈ Finset.Icc 1 J,
              (1 / ((J : ℝ) * ℓ) - 1 / (((J : ℝ) + 1) * ℓ))
              = harmonic J / ((J : ℝ) * ((J : ℝ) + 1)) := by
            rw [harmonic, Finset.sum_div]
            refine Finset.sum_congr rfl fun ℓ hℓ => ?_
            have hℓ1 : (1 : ℝ) ≤ (ℓ : ℝ) := by
              exact_mod_cast (Finset.mem_Icc.mp hℓ).1
            have hℓ0 : (0 : ℝ) < ℓ := by linarith
            field_simp
            ring
          rw [hsum]
          field_simp
          ring
        · exact div_nonneg (by norm_num) (by positivity)
        · refine Finset.sum_nonneg fun ℓ hℓ => ?_
          have hℓ1 : (1 : ℝ) ≤ (ℓ : ℝ) := by
            exact_mod_cast (Finset.mem_Icc.mp hℓ).1
          have hℓ0 : (0 : ℝ) < ℓ := by linarith
          have h1 : 1 / (((J : ℝ) + 1) * ℓ) ≤ 1 / ((J : ℝ) * ℓ) := by
            apply one_div_le_one_div_of_le (by positivity)
            nlinarith
          linarith
        · intro ℓ hℓ
          have hℓ1 : (1 : ℝ) ≤ (ℓ : ℝ) := by
            exact_mod_cast (Finset.mem_Icc.mp hℓ).1
          have hℓ0 : (0 : ℝ) < ℓ := by linarith
          have h1 : 1 / (((J : ℝ) + 1) * ℓ) ≤ 1 / ((J : ℝ) * ℓ) := by
            apply one_div_le_one_div_of_le (by positivity)
            nlinarith
          linarith

/-- **Pointwise**: on t > 0 the squared witness is ¼·𝟙_{Dt}. -/
lemma witness_sq_eq (J : ℕ) {t : ℝ} (ht : 0 < t) :
    (chainDiff J t - chainDiff (J + 1) t) ^ 2
      = (Dt J).indicator (fun _ => (1 : ℝ) / 4) t := by
  have hcast : ((J + 1 : ℕ) : ℝ) = (J : ℝ) + 1 := by push_cast; ring
  have hmem_iff : t ∈ Dt J ↔
      Odd (⌊1 / ((J : ℝ) * t)⌋ - ⌊1 / (((J : ℝ) + 1) * t)⌋) := by
    simp [Dt, ht]
  rw [chainDiff_eq_squareWave, chainDiff_eq_squareWave, hcast]
  set a := ⌊1 / ((J : ℝ) * t)⌋ with ha
  set b := ⌊1 / (((J : ℝ) + 1) * t)⌋ with hb
  by_cases hp : Odd (a - b)
  · rw [indicator_of_mem (hmem_iff.mpr hp)]
    rcases Int.emod_two_eq_zero_or_one a with hA | hA <;>
      rcases Int.emod_two_eq_zero_or_one b with hB | hB
    · exfalso; rcases hp with ⟨c, hc⟩; omega
    · rw [hA, hB]; norm_num
    · rw [hA, hB]; norm_num
    · exfalso; rcases hp with ⟨c, hc⟩; omega
  · rw [indicator_of_notMem (fun hm => hp (hmem_iff.mp hm))]
    rcases Int.emod_two_eq_zero_or_one a with hA | hA <;>
      rcases Int.emod_two_eq_zero_or_one b with hB | hB
    · rw [hA, hB]; norm_num
    · exact absurd ⟨(a - b - 1) / 2, by omega⟩ hp
    · exact absurd ⟨(a - b - 1) / 2, by omega⟩ hp
    · rw [hA, hB]; norm_num

/-- The witness integral equals ¼ of the disagreement measure. -/
theorem witness_integral_eq (hJ : 2 ≤ J) :
    ∫ t in Ioi (0:ℝ), (chainDiff J t - chainDiff (J + 1) t) ^ 2
      = (volume (Dt J)).toReal / 4 := by
  rw [setIntegral_congr_fun measurableSet_Ioi
      (fun t ht => witness_sq_eq J (mem_Ioi.mp ht))]
  rw [integral_indicator Dt_measurable, setIntegral_const]
  rw [measureReal_def, Measure.restrict_apply Dt_measurable,
      inter_eq_self_of_subset_left Dt_subset_Ioi, smul_eq_mul]
  ring

/-- **MILESTONE 3.** -/
theorem witness_integral_le (hJ : 2 ≤ J) :
    ∫ t in Ioi (0:ℝ), (chainDiff J t - chainDiff (J + 1) t) ^ 2
      ≤ (harmonic J + 1) / (4 * (J : ℝ) * ((J : ℝ) + 1)) := by
  have hJ0 : (0 : ℝ) < J := by
    have : (2 : ℝ) ≤ J := by exact_mod_cast hJ
    linarith
  have hharm : 0 ≤ harmonic J := by
    refine Finset.sum_nonneg fun ℓ hℓ => ?_
    positivity
  have h := ENNReal.toReal_mono (by exact ENNReal.ofReal_ne_top) (Dt_volume_le hJ)
  rw [ENNReal.toReal_ofReal
    (div_nonneg (by linarith) (by positivity))] at h
  rw [witness_integral_eq hJ]
  have halg : (harmonic J + 1) / ((J : ℝ) * ((J : ℝ) + 1)) / 4
      = (harmonic J + 1) / (4 * (J : ℝ) * ((J : ℝ) + 1)) := by
    rw [div_div]
    ring_nf
  linarith [halg]

end

end NymanBeurling

#print axioms NymanBeurling.Dt_volume_le
#print axioms NymanBeurling.witness_integral_le

/-! ## Milestone 4: THEOREM 1 END-TO-END

The Nyman–Beurling Gram matrix (entries = L² inner products of the
dilation family), the four-dilation witness, and the Rayleigh
conclusion: some eigenvalue of the Gram matrix is at most
(H_{K−1} + 1)/(10·(K−1)·K). -/

namespace NymanBeurling

noncomputable section

open scoped RealInnerProductSpace
open Matrix

abbrev mu0 : Measure ℝ := volume.restrict (Set.Ioi (0:ℝ))

/-- The dilation family as elements of L²((0,∞)). -/
def E (k : ℕ) (hk : 1 ≤ k) : Lp ℝ 2 mu0 := (eDil_memLp_two k hk).toLp

/-- Gram quadratic form = norm of the combination (general lemma). -/
lemma dot_mulVec_inner {n : ℕ} {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℝ H] (v : Fin n → H) (x : Fin n → ℝ) :
    x ⬝ᵥ ((Matrix.of fun i j => (⟪v i, v j⟫ : ℝ)) *ᵥ x)
      = ⟪∑ i, x i • v i, ∑ j, x j • v j⟫ := by
  simp only [dotProduct, Matrix.mulVec, Matrix.of_apply, sum_inner,
    inner_sum, real_inner_smul_left, real_inner_smul_right]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => by ring

/-- Collapsing a Pi.single against a sum of smuls. -/
lemma sum_single_smul {n : ℕ} {H : Type*} [AddCommMonoid H] [Module ℝ H]
    (a : Fin n) (c : ℝ) (v : Fin n → H) :
    (∑ i, (Pi.single a c : Fin n → ℝ) i • v i) = c • v a := by
  rw [Finset.sum_eq_single_of_mem a (Finset.mem_univ a)
    (fun b _ hb => by rw [Pi.single_eq_of_ne hb, zero_smul])]
  rw [Pi.single_eq_same]

end

end NymanBeurling

/-! ## The end-to-end theorem -/

namespace NymanBeurling

noncomputable section

open scoped RealInnerProductSpace
open Matrix

variable (N : ℕ)

/-- The dilation family over Fin N (dilation index i+1 ≥ 1). -/
def Ev : Fin N → Lp ℝ 2 mu0 := fun i => E (i + 1) (by omega)

/-- Dilation-index congruence for E (proof argument is irrelevant). -/
lemma E_congr {a b : ℕ} (h : a = b) (ha : 1 ≤ a) (hb : 1 ≤ b) :
    E a ha = E b hb := by subst h; rfl

/-- The Nyman–Beurling Gram matrix. -/
def nbGram : Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun i j => ⟪Ev N i, Ev N j⟫

theorem nbGram_isHermitian : (nbGram N).IsHermitian := by
  ext i j
  simp only [nbGram, conjTranspose_apply, Matrix.of_apply, star_trivial]
  exact real_inner_comm _ _

/-- **THEOREM 1, END TO END** (chains_theory.md §0, machine-verified in
full): for 2 ≤ J and 2(J+1) ≤ N, some eigenvalue of the N×N
Nyman–Beurling Gram matrix is at most (H_J + 1)/(10·J·(J+1)).
With K = J + 1 this is Theorem 1's bound (H_{K−1}+1)/(10·K(K−1)). -/
theorem nb_gram_eigenvalue_le {J : ℕ} (hJ : 2 ≤ J) (hN : 2 * (J + 1) ≤ N) :
    ∃ k, (nbGram_isHermitian N).eigenvalues k
      ≤ (harmonic J + 1) / (10 * (J : ℝ) * ((J : ℝ) + 1)) := by
  have hJ0 : (0 : ℝ) < J := by
    have : (2 : ℝ) ≤ J := by exact_mod_cast hJ
    linarith
  have hb1 : J - 1 < N := by omega
  have hb2 : 2 * J - 1 < N := by omega
  have hb3 : J < N := by omega
  have hb4 : 2 * J + 1 < N := by omega
  set i1 : Fin N := ⟨J - 1, hb1⟩ with hi1
  set i2 : Fin N := ⟨2 * J - 1, hb2⟩ with hi2
  set i3 : Fin N := ⟨J, hb3⟩ with hi3
  set i4 : Fin N := ⟨2 * J + 1, hb4⟩ with hi4
  have h12 : i1 ≠ i2 := by simp only [hi1, hi2, ne_eq, Fin.mk.injEq]; omega
  have h13 : i1 ≠ i3 := by simp only [hi1, hi3, ne_eq, Fin.mk.injEq]; omega
  have h14 : i1 ≠ i4 := by simp only [hi1, hi4, ne_eq, Fin.mk.injEq]; omega
  have h23 : i2 ≠ i3 := by simp only [hi2, hi3, ne_eq, Fin.mk.injEq]; omega
  have h24 : i2 ≠ i4 := by simp only [hi2, hi4, ne_eq, Fin.mk.injEq]; omega
  have h34 : i3 ≠ i4 := by simp only [hi3, hi4, ne_eq, Fin.mk.injEq]; omega
  set x : Fin N → ℝ :=
    (Pi.single i1 (-(2:ℝ)⁻¹) : Fin N → ℝ) + Pi.single i2 1
      + Pi.single i3 (2:ℝ)⁻¹ + Pi.single i4 (-1) with hx
  -- coefficient norm: x ⬝ᵥ x = 5/2
  have hxx : x ⬝ᵥ x = 5 / 2 := by
    simp only [hx, add_dotProduct, single_dotProduct, Pi.add_apply,
      Pi.single_apply, if_neg h12, if_neg h13, if_neg h14, if_neg h23,
      if_neg h24, if_neg h34, if_neg (Ne.symm h12), if_neg (Ne.symm h13),
      if_neg (Ne.symm h14), if_neg (Ne.symm h23), if_neg (Ne.symm h24),
      if_neg (Ne.symm h34)]
    norm_num
  -- the L² combination collapses to the chain-difference witness
  set S : Lp ℝ 2 mu0 := ∑ i, x i • Ev N i with hS
  have hScollapse : S = (-(2:ℝ)⁻¹) • Ev N i1 + (1:ℝ) • Ev N i2
      + (2:ℝ)⁻¹ • Ev N i3 + (-1:ℝ) • Ev N i4 := by
    rw [hS, hx]
    simp only [Pi.add_apply, add_smul, Finset.sum_add_distrib]
    rw [sum_single_smul, sum_single_smul, sum_single_smul, sum_single_smul]
  -- identify the four dilations
  have hv1 : (i1 : ℕ) + 1 = J := by
    have : (i1 : ℕ) = J - 1 := rfl
    omega
  have hv2 : (i2 : ℕ) + 1 = 2 * J := by
    have : (i2 : ℕ) = 2 * J - 1 := rfl
    omega
  have hv3 : (i3 : ℕ) + 1 = J + 1 := rfl
  have hv4 : (i4 : ℕ) + 1 = 2 * J + 2 := by
    have : (i4 : ℕ) = 2 * J + 1 := rfl
    omega
  have hE1 : Ev N i1 = (eDil_memLp_two J (by omega)).toLp :=
    E_congr hv1 (by omega) (by omega)
  have hE2 : Ev N i2 = (eDil_memLp_two (2 * J) (by omega)).toLp :=
    E_congr hv2 (by omega) (by omega)
  have hE3 : Ev N i3 = (eDil_memLp_two (J + 1) (by omega)).toLp :=
    E_congr hv3 (by omega) (by omega)
  have hE4 : Ev N i4 = (eDil_memLp_two (2 * J + 2) (by omega)).toLp :=
    E_congr hv4 (by omega) (by omega)
  -- a.e. identification of S with the witness function
  have hSae : (S : ℝ → ℝ) =ᵐ[mu0]
      fun t => chainDiff J t - chainDiff (J + 1) t := by
    rw [hScollapse, hE1, hE2, hE3, hE4]
    filter_upwards [Lp.coeFn_add ((-(2:ℝ)⁻¹) • (eDil_memLp_two J (by omega)).toLp
        + (1:ℝ) • (eDil_memLp_two (2*J) (by omega)).toLp
        + (2:ℝ)⁻¹ • (eDil_memLp_two (J+1) (by omega)).toLp)
        ((-1:ℝ) • (eDil_memLp_two (2*J+2) (by omega)).toLp),
      Lp.coeFn_add ((-(2:ℝ)⁻¹) • (eDil_memLp_two J (by omega)).toLp
        + (1:ℝ) • (eDil_memLp_two (2*J) (by omega)).toLp)
        ((2:ℝ)⁻¹ • (eDil_memLp_two (J+1) (by omega)).toLp),
      Lp.coeFn_add ((-(2:ℝ)⁻¹) • (eDil_memLp_two J (by omega)).toLp)
        ((1:ℝ) • (eDil_memLp_two (2*J) (by omega)).toLp),
      Lp.coeFn_smul (-(2:ℝ)⁻¹) ((eDil_memLp_two J (by omega)).toLp),
      Lp.coeFn_smul (1:ℝ) ((eDil_memLp_two (2*J) (by omega)).toLp),
      Lp.coeFn_smul ((2:ℝ)⁻¹) ((eDil_memLp_two (J+1) (by omega)).toLp),
      Lp.coeFn_smul (-1:ℝ) ((eDil_memLp_two (2*J+2) (by omega)).toLp),
      (eDil_memLp_two J (by omega : 1 ≤ J)).coeFn_toLp,
      (eDil_memLp_two (2*J) (by omega : 1 ≤ 2*J)).coeFn_toLp,
      (eDil_memLp_two (J+1) (by omega : 1 ≤ J+1)).coeFn_toLp,
      (eDil_memLp_two (2*J+2) (by omega : 1 ≤ 2*J+2)).coeFn_toLp]
      with t ha hb hc hs1 hs2 hs3 hs4 he1 he2 he3 he4
    simp only [ha, hb, hc, Pi.add_apply, hs1, hs2, hs3, hs4, Pi.smul_apply,
      he1, he2, he3, he4, smul_eq_mul]
    have h2J2 : 2 * (J + 1) = 2 * J + 2 := by ring
    simp only [chainDiff, h2J2]
    ring
  -- the quadratic form equals the witness integral
  have hform : x ⬝ᵥ ((nbGram N) *ᵥ x)
      = ∫ t in Set.Ioi (0:ℝ), (chainDiff J t - chainDiff (J + 1) t) ^ 2 := by
    have h1 : x ⬝ᵥ ((nbGram N) *ᵥ x) = ⟪S, S⟫ := by
      rw [hS]
      exact dot_mulVec_inner (Ev N) x
    rw [h1, MeasureTheory.L2.inner_def]
    refine integral_congr_ae ?_
    filter_upwards [hSae] with t ht
    rw [Real.inner_apply, ht]
    ring
  -- Rayleigh + Milestone 3
  haveI : Nonempty (Fin N) := ⟨i3⟩
  obtain ⟨k, hk⟩ := exists_eigenvalue_mul_le_rayleigh (nbGram_isHermitian N) x
  refine ⟨k, ?_⟩
  rw [hxx, hform] at hk
  have hint := witness_integral_le hJ
  have h52 : (0:ℝ) < 5 / 2 := by norm_num
  have hbound : (nbGram_isHermitian N).eigenvalues k
      ≤ ((harmonic J + 1) / (4 * (J : ℝ) * ((J : ℝ) + 1))) / (5 / 2) := by
    rw [le_div_iff₀ h52]
    calc (nbGram_isHermitian N).eigenvalues k * (5 / 2) ≤ _ := hk
    _ ≤ _ := hint
  calc (nbGram_isHermitian N).eigenvalues k
      ≤ ((harmonic J + 1) / (4 * (J : ℝ) * ((J : ℝ) + 1))) / (5 / 2) := hbound
    _ = (harmonic J + 1) / (10 * (J : ℝ) * ((J : ℝ) + 1)) := by
        rw [div_div]
        congr 1
        ring

end

end NymanBeurling

#print axioms NymanBeurling.witness_integral_le
#print axioms NymanBeurling.nb_gram_eigenvalue_le
