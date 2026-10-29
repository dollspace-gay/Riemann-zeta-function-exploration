import Mathlib.NumberTheory.ArithmeticFunction.Moebius
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Tactic

/-! Finite operator estimate for weighted coefficient recovery. -/

namespace NymanBeurling.Jump

open Finset

/-- The elementary Schur test, squared, for arbitrary finite real matrices. -/
theorem finite_schur_bound {ι κ : Type*} (s : Finset ι) (t : Finset κ)
    (M : ι → κ → ℝ) (x : κ → ℝ) (R C : ℝ) (hR : 0 ≤ R)
    (hrow : ∀ i ∈ s, ∑ j ∈ t, |M i j| ≤ R)
    (hcol : ∀ j ∈ t, ∑ i ∈ s, |M i j| ≤ C) :
    (∑ i ∈ s, (∑ j ∈ t, M i j * x j) ^ 2) ≤ R * C * ∑ j ∈ t, x j ^ 2 := by
  have hcs (i : ι) : (∑ j ∈ t, M i j * x j) ^ 2 ≤
      (∑ j ∈ t, |M i j|) * ∑ j ∈ t, |M i j| * x j ^ 2 := by
    apply Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul t
      (fun j _ => abs_nonneg (M i j)) (fun j _ => mul_nonneg (abs_nonneg _) (sq_nonneg _))
    intro j _
    rw [mul_pow, ← sq_abs (M i j)]
    nlinarith
  calc
    _ ≤ ∑ i ∈ s, R * ∑ j ∈ t, |M i j| * x j ^ 2 := by
      apply sum_le_sum
      intro i hi
      exact (hcs i).trans (mul_le_mul_of_nonneg_right (hrow i hi)
        (sum_nonneg fun j _ => mul_nonneg (abs_nonneg _) (sq_nonneg _)))
    _ = R * ∑ j ∈ t, (∑ i ∈ s, |M i j|) * x j ^ 2 := by
      rw [← mul_sum, sum_comm]
      simp_rw [sum_mul]
    _ ≤ R * ∑ j ∈ t, C * x j ^ 2 := by
      apply mul_le_mul_of_nonneg_left _ hR
      exact sum_le_sum fun j hj => mul_le_mul_of_nonneg_right (hcol j hj) (sq_nonneg _)
    _ = R * C * ∑ j ∈ t, x j ^ 2 := by rw [← mul_sum, mul_assoc]

noncomputable def harmonicSum (N : ℕ) : ℝ := ∑ d ∈ Ioc 0 N, 1 / (d : ℝ)

noncomputable def weightedMobius (n d : ℕ) : ℝ :=
  if d ∣ n then (ArithmeticFunction.moebius (n / d) : ℝ) * d else 0

lemma harmonicSum_nonneg (N : ℕ) : 0 ≤ harmonicSum N := by
  exact sum_nonneg fun d _ => by positivity

lemma abs_weightedMobius_le (n d : ℕ) :
    |weightedMobius n d| ≤ if d ∣ n then (d : ℝ) else 0 := by
  unfold weightedMobius
  split_ifs
  · rw [abs_mul, show |(d : ℝ)| = d from abs_of_nonneg (Nat.cast_nonneg d)]
    have hm : |(ArithmeticFunction.moebius (n / d) : ℝ)| ≤ 1 := by
      exact_mod_cast (ArithmeticFunction.abs_moebius_le_one (n := n / d))
    simpa using mul_le_mul_of_nonneg_right hm (Nat.cast_nonneg (α := ℝ) d)
  · simp

lemma filter_Ioc_dvd (N n : ℕ) (hn : 0 < n) (hnN : n ≤ N) :
    (Ioc 0 N).filter (fun d => d ∣ n) = n.divisors := by
  ext d
  simp only [mem_filter, mem_Ioc, Nat.mem_divisors]
  constructor
  · exact fun h => ⟨h.2, hn.ne'⟩
  · intro h
    exact ⟨⟨Nat.pos_of_dvd_of_pos h.1 hn, (Nat.le_of_dvd hn h.1).trans hnN⟩, h.1⟩

theorem weightedMobius_row_le (N n : ℕ) (hn : 0 < n) (hnN : n ≤ N) :
    (∑ d ∈ Ioc 0 N, |weightedMobius n d|) ≤ N * harmonicSum N := by
  calc
    _ ≤ ∑ d ∈ Ioc 0 N, if d ∣ n then (d : ℝ) else 0 :=
      sum_le_sum fun d _ => abs_weightedMobius_le n d
    _ = ∑ d ∈ n.divisors, (d : ℝ) := by
      rw [← sum_filter, filter_Ioc_dvd N n hn hnN]
    _ = ∑ d ∈ n.divisors, ((n / d : ℕ) : ℝ) := (Nat.sum_div_divisors n _).symm
    _ = ∑ d ∈ n.divisors, (n : ℝ) / d := by
      apply sum_congr rfl
      intro d hd
      exact Nat.cast_div_charZero (K := ℝ) (Nat.dvd_of_mem_divisors hd)
    _ = (n : ℝ) * ∑ d ∈ n.divisors, 1 / (d : ℝ) := by
      simp only [mul_sum, mul_one_div]
    _ ≤ (n : ℝ) * harmonicSum N := by
      apply mul_le_mul_of_nonneg_left _ (Nat.cast_nonneg n)
      apply sum_le_sum_of_subset_of_nonneg
      · intro d hd
        exact mem_Ioc.mpr ⟨Nat.pos_of_mem_divisors hd,
          (Nat.le_of_dvd hn (Nat.dvd_of_mem_divisors hd)).trans hnN⟩
      · intro d _ _
        positivity
    _ ≤ (N : ℝ) * harmonicSum N :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast hnN) (harmonicSum_nonneg N)

theorem weightedMobius_column_le (N d : ℕ) :
    (∑ n ∈ Ioc 0 N, |weightedMobius n d|) ≤ (N : ℝ) := by
  calc
    _ ≤ ∑ n ∈ Ioc 0 N, if d ∣ n then (d : ℝ) else 0 :=
      sum_le_sum fun n _ => abs_weightedMobius_le n d
    _ = ((N / d : ℕ) : ℝ) * d := by
      rw [← sum_filter, sum_const, nsmul_eq_mul, Nat.Ioc_filter_dvd_card_eq_div]
    _ ≤ N := by exact_mod_cast Nat.div_mul_le_self N d

noncomputable def divisorSum (c : ℕ → ℝ) (n : ℕ) : ℝ := ∑ d ∈ n.divisors, c d

theorem recover_coefficients (c : ℕ → ℝ) (N n : ℕ) (hn : 0 < n) (hnN : n ≤ N) :
    (∑ d ∈ Ioc 0 N, weightedMobius n d * (divisorSum c d / d)) = c n := by
  have hinv := (ArithmeticFunction.sum_eq_iff_sum_mul_moebius_eq
    (f := c) (g := divisorSum c)).mp (fun _ _ => rfl) n hn
  rw [Nat.sum_divisorsAntidiagonal' (fun a b =>
    (ArithmeticFunction.moebius a : ℝ) * divisorSum c b)] at hinv
  rw [← hinv]
  unfold weightedMobius
  simp_rw [ite_mul, zero_mul]
  rw [← sum_filter, filter_Ioc_dvd N n hn hnN]
  apply sum_congr rfl
  intro d hd
  have hd0 : (d : ℝ) ≠ 0 := by exact_mod_cast (Nat.pos_of_mem_divisors hd).ne'
  field_simp

/-- Weighted Möbius inversion loses at most N² H_N in squared norm. -/
theorem coefficient_recovery_bound (c : ℕ → ℝ) (N : ℕ) :
    (∑ n ∈ Ioc 0 N, c n ^ 2) ≤
      (N : ℝ) ^ 2 * harmonicSum N * ∑ d ∈ Ioc 0 N, (divisorSum c d / d) ^ 2 := by
  have h := finite_schur_bound (Ioc 0 N) (Ioc 0 N) weightedMobius
    (fun d => divisorSum c d / d) (N * harmonicSum N) N
    (mul_nonneg (Nat.cast_nonneg N) (harmonicSum_nonneg N))
    (fun n hn => weightedMobius_row_le N n (mem_Ioc.mp hn).1 (mem_Ioc.mp hn).2)
    (fun d _ => weightedMobius_column_le N d)
  have hsum : (∑ n ∈ Ioc 0 N, (∑ d ∈ Ioc 0 N,
      weightedMobius n d * (divisorSum c d / d)) ^ 2) = ∑ n ∈ Ioc 0 N, c n ^ 2 := by
    apply sum_congr rfl
    intro n hn
    rw [recover_coefficients c N n (mem_Ioc.mp hn).1 (mem_Ioc.mp hn).2]
  rw [hsum] at h
  convert h using 1
  ring

end NymanBeurling.Jump

#print axioms NymanBeurling.Jump.finite_schur_bound
#print axioms NymanBeurling.Jump.coefficient_recovery_bound
