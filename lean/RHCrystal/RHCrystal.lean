/-
  RH Crystal: Formal Verification of Xi-Weight Symmetry
  ======================================================
  PROVEN (no sorry, no new axioms):
  ✓ cosh(-x) = cosh(x)
  ✓ cosh(0) = 1
  ✓ cosh(x) ≥ 1
  ✓ w(p,σ) = w(p,1-σ)
  ✓ w(p,1/2) = log(p)·1/√p
  ✓ G_{ij}(σ) = G_{ij}(1-σ)
  ✓ G(σ) = G(1-σ), hence κ(σ) = κ(1-σ)
  ✓ Rayleigh bound: ∃ k, λ_k·(x⬝x) ≤ x⬝(Ax)  (real symmetric A)
  ✓ Pair bound: ∃ k, 2λ_k ≤ A_ii + A_jj − 2A_ij
  ✓ Gram interlacing bound (paper v4 §3.3):
      ∃ k, λ_k(G) ≤ Σ_p w(p,σ)·(1 − cos((γ_i−γ_j)·log p))
  ✓ Nyman-Beurling disagreement-integral bound (NymanBeurling.lean,
      analytic core of nyman-beurling Theorem 1):
      ∫_D u⁻² ≤ (H_J + 1)/(J(J+1)) for the parity-disagreement set D

  ZERO SORRIES. ZERO NEW AXIOMS.
-/

import Mathlib.Tactic
import NymanBeurling
import Mathlib.Data.List.Basic
import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.LinearAlgebra.Matrix.Hermitian

noncomputable section

/-- Hyperbolic cosine: cosh(x) = (exp(x) + exp(-x)) / 2 -/
def cosh (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

/-- **cosh is even**: cosh(-x) = cosh(x) -/
theorem cosh_neg (x : ℝ) : cosh (-x) = cosh x := by
  unfold cosh
  rw [neg_neg]
  ring

/-- **cosh(0) = 1** -/
theorem cosh_zero : cosh 0 = 1 := by
  unfold cosh
  simp [Real.exp_zero]

/-- **cosh(x) ≥ 1** for all x.
    AM-GM on exp(x/2) and exp(-x/2):
    (e^{x/2} - e^{-x/2})² ≥ 0  ⟹  e^x + e^{-x} ≥ 2·e^{x/2}e^{-x/2} = 2. -/
theorem cosh_ge_one (x : ℝ) : 1 ≤ cosh x := by
  unfold cosh
  have h1 : Real.exp (x/2) * Real.exp (-(x/2)) = 1 := by
    rw [← Real.exp_add]; simp
  have h2 : Real.exp (x/2) * Real.exp (x/2) = Real.exp x := by
    rw [← Real.exp_add]; congr 1; ring
  have h3 : Real.exp (-(x/2)) * Real.exp (-(x/2)) = Real.exp (-x) := by
    rw [← Real.exp_add]; congr 1; ring
  nlinarith [sq_nonneg (Real.exp (x/2) - Real.exp (-(x/2))), h1, h2, h3]

/-- The Xi weight: w(p, σ) = log(p) · cosh((σ - 1/2) · log(p)) / √p -/
def xi_weight (p : ℝ) (σ : ℝ) : ℝ :=
  Real.log p * cosh ((σ - 1 / 2) * Real.log p) / Real.sqrt p

/-- Negation distributes into product. -/
lemma neg_mul_log (σ logp : ℝ) :
    ((1 - σ) - 1 / 2) * logp = -((σ - 1 / 2) * logp) := by ring

/-- **THEOREM (Weight Symmetry).**
    w(p, σ) = w(p, 1-σ) for all p and σ. -/
theorem xi_weight_symm (p σ : ℝ) :
    xi_weight p σ = xi_weight p (1 - σ) := by
  unfold xi_weight
  rw [neg_mul_log σ (Real.log p)]
  rw [cosh_neg]

/-- **THEOREM.** w(p, 1/2) simplifies: cosh(0) = 1. -/
theorem xi_weight_half (p : ℝ) :
    xi_weight p (1 / 2) = Real.log p * 1 / Real.sqrt p := by
  unfold xi_weight
  have h : (1 / 2 - 1 / 2 : ℝ) * Real.log p = 0 := by ring
  rw [h, cosh_zero]

/-- Gram matrix entry as a sum over primes. -/
def gram_entry_re {n : ℕ} (primes : List ℝ) (γ : Fin n → ℝ)
    (σ : ℝ) (i j : Fin n) : ℝ :=
  (primes.map fun p =>
    xi_weight p σ * Real.cos ((γ i - γ j) * Real.log p)).sum

/-- **THEOREM (Gram Entry Symmetry).**
    G_{ij}(σ) = G_{ij}(1-σ). -/
theorem gram_entry_symm {n : ℕ} (primes : List ℝ)
    (γ : Fin n → ℝ) (σ : ℝ) (i j : Fin n) :
    gram_entry_re primes γ σ i j = gram_entry_re primes γ (1 - σ) i j := by
  unfold gram_entry_re
  congr 1
  apply List.map_congr_left
  intro p _
  rw [xi_weight_symm]

/-- **THEOREM (Full Gram Matrix Symmetry).**
    G(σ) = G(1-σ), hence κ(σ) = κ(1-σ). -/
theorem gram_matrix_symm {n : ℕ} (primes : List ℝ)
    (γ : Fin n → ℝ) (σ : ℝ) :
    (fun i j => gram_entry_re primes γ σ i j) =
    (fun i j => gram_entry_re primes γ (1 - σ) i j) := by
  ext i j
  exact gram_entry_symm primes γ σ i j

/-! ## The interlacing bound (paper v4, §3.3)

For a real symmetric matrix `A`, the Rayleigh quotient of the test vector
`e_i − e_j` bounds the smallest eigenvalue:  some eigenvalue is at most
`(A i i + A j j)/2 − A i j`.  Instantiated for the Gram matrix this gives

    λ_min(G) ≤ Σ_p w(p,σ) · (1 − cos((γ_i − γ_j) · log p))

— the pair bound B(δ) of the paper, the first machine-verified statement
about the *spectrum* of the prime-zero coupling. -/

section Interlacing

open Matrix

variable {n : ℕ}

/-- Sums of mapped lists subtract pointwise. -/
lemma list_sum_map_sub (l : List ℝ) (f g : ℝ → ℝ) :
    (l.map fun a => f a - g a).sum = (l.map f).sum - (l.map g).sum := by
  induction l with
  | nil => simp
  | cons a t ih =>
      simp only [List.map_cons, List.sum_cons, ih]
      ring

/-- Conjugation by a real unitary preserves quadratic forms: with
`y = star U *ᵥ x`, both `x ⬝ᵥ ((U * D * star U) *ᵥ x) = y ⬝ᵥ (D *ᵥ y)`
and `x ⬝ᵥ x = y ⬝ᵥ y`. -/
private lemma quadform_conj {U D : Matrix (Fin n) (Fin n) ℝ}
    (hUU : U * star U = 1) (x : Fin n → ℝ) :
    x ⬝ᵥ ((U * D * star U) *ᵥ x) = (star U *ᵥ x) ⬝ᵥ (D *ᵥ (star U *ᵥ x)) ∧
    x ⬝ᵥ x = (star U *ᵥ x) ⬝ᵥ (star U *ᵥ x) := by
  have hstar : star U = Uᵀ := by
    rw [Matrix.star_eq_conjTranspose, conjTranspose_eq_transpose_of_trivial]
  have hUUT : U * Uᵀ = 1 := by rw [← hstar]; exact hUU
  constructor
  · rw [← mulVec_mulVec, ← mulVec_mulVec, dotProduct_mulVec]
    congr 1
    rw [hstar, mulVec_transpose]
  · conv_rhs => rw [dotProduct_mulVec, ← mulVec_transpose, hstar, transpose_transpose,
      mulVec_mulVec, hUUT, one_mulVec]

/-- **Rayleigh bound.** For a real symmetric matrix `A` and any vector `x`,
some eigenvalue `λ_k` satisfies `λ_k · (x ⬝ᵥ x) ≤ x ⬝ᵥ (A *ᵥ x)`. -/
theorem exists_eigenvalue_mul_le_rayleigh [Nonempty (Fin n)]
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) (x : Fin n → ℝ) :
    ∃ k, hA.eigenvalues k * (x ⬝ᵥ x) ≤ x ⬝ᵥ (A *ᵥ x) := by
  classical
  obtain ⟨k₀, -, hk₀⟩ := Finset.exists_min_image (Finset.univ : Finset (Fin n))
    hA.eigenvalues Finset.univ_nonempty
  refine ⟨k₀, ?_⟩
  have hUU : (hA.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ) *
      star (hA.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ) = 1 := by
    rw [← Unitary.coe_star, Unitary.coe_mul_star_self]
  have hAeq : A = (hA.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ) *
      diagonal hA.eigenvalues * star (hA.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ) := by
    conv_lhs => rw [hA.spectral_theorem]
    rw [Unitary.conjStarAlgAut_apply]
    simp only [RCLike.ofReal_real_eq_id, Function.id_comp]
  obtain ⟨hq1, hq2⟩ := quadform_conj (D := diagonal hA.eigenvalues) hUU x
  have hdiag : ∀ v : Fin n → ℝ,
      v ⬝ᵥ (diagonal hA.eigenvalues *ᵥ v) = ∑ k, hA.eigenvalues k * v k ^ 2 := fun v => by
    simp only [dotProduct, mulVec_diagonal]
    exact Finset.sum_congr rfl fun k _ => by ring
  have hsq : ∀ v : Fin n → ℝ, v ⬝ᵥ v = ∑ k, v k ^ 2 := fun v => by
    simp only [dotProduct]
    exact Finset.sum_congr rfl fun k _ => by ring
  calc hA.eigenvalues k₀ * (x ⬝ᵥ x)
      = ∑ k, hA.eigenvalues k₀ *
          ((star (hA.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ) *ᵥ x) k) ^ 2 := by
        rw [hq2, hsq, Finset.mul_sum]
    _ ≤ ∑ k, hA.eigenvalues k *
          ((star (hA.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ) *ᵥ x) k) ^ 2 :=
        Finset.sum_le_sum fun k _ =>
          mul_le_mul_of_nonneg_right (hk₀ k (Finset.mem_univ k)) (sq_nonneg _)
    _ = x ⬝ᵥ (A *ᵥ x) := by
        rw [← hdiag, ← hq1, ← hAeq]

/-- **Pair bound.** For a real symmetric matrix, every off-diagonal pair
bounds some eigenvalue: `2·λ_k ≤ A i i + A j j − 2·A i j`. -/
theorem exists_eigenvalue_le_pair_bound
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) {i j : Fin n} (hij : i ≠ j) :
    ∃ k, 2 * hA.eigenvalues k ≤ A i i + A j j - 2 * A i j := by
  haveI : Nonempty (Fin n) := ⟨i⟩
  obtain ⟨k, hk⟩ := exists_eigenvalue_mul_le_rayleigh hA (Pi.single i 1 - Pi.single j 1)
  refine ⟨k, ?_⟩
  have hsym : A j i = A i j := by
    have h := congrFun (congrFun hA j) i
    simp only [conjTranspose_apply, star_trivial] at h
    exact h.symm
  have h1 : (Pi.single i 1 - Pi.single j 1 : Fin n → ℝ) ⬝ᵥ
      (Pi.single i 1 - Pi.single j 1) = 2 := by
    simp [dotProduct_sub, hij, Ne.symm hij]
    norm_num
  have h2 : (Pi.single i 1 - Pi.single j 1 : Fin n → ℝ) ⬝ᵥ
      (A *ᵥ (Pi.single i 1 - Pi.single j 1)) = A i i + A j j - 2 * A i j := by
    simp [mulVec_sub, dotProduct_sub, sub_dotProduct, single_dotProduct, hsym]
    ring
  rw [h1, h2] at hk
  linarith

/-- The Gram matrix as a `Matrix`, assembled from `gram_entry_re`. -/
noncomputable def gramMatrix (primes : List ℝ) (γ : Fin n → ℝ) (σ : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => gram_entry_re primes γ σ i j

/-- The Gram matrix is real symmetric (Hermitian over ℝ):
`cos((γ_j − γ_i) log p) = cos((γ_i − γ_j) log p)`. -/
theorem gramMatrix_isHermitian (primes : List ℝ) (γ : Fin n → ℝ) (σ : ℝ) :
    (gramMatrix primes γ σ).IsHermitian := by
  ext i j
  simp only [conjTranspose_apply, gramMatrix, Matrix.of_apply, star_trivial]
  unfold gram_entry_re
  congr 1
  apply List.map_congr_left
  intro p _
  have h : (γ i - γ j) * Real.log p = -((γ j - γ i) * Real.log p) := by ring
  rw [h, Real.cos_neg]

/-- The Gram matrix diagonal: `G i i = Σ_p w(p,σ)`. -/
theorem gramMatrix_diag (primes : List ℝ) (γ : Fin n → ℝ) (σ : ℝ) (i : Fin n) :
    gramMatrix primes γ σ i i = (primes.map fun p => xi_weight p σ).sum := by
  simp only [gramMatrix, Matrix.of_apply]
  unfold gram_entry_re
  congr 1
  apply List.map_congr_left
  intro p _
  rw [sub_self, zero_mul, Real.cos_zero, mul_one]

/-- **THEOREM (Interlacing pair bound, paper v4 §3.3).**
Some eigenvalue of the Gram matrix is at most the pair functional

    B(i,j) = Σ_p w(p,σ) · (1 − cos((γ_i − γ_j) · log p)).

In particular the smallest eigenvalue is bounded by the minimum of B over
all pairs — close zero pairs force small eigenvalues. -/
theorem gram_eigenvalue_le_pair_bound
    (primes : List ℝ) (γ : Fin n → ℝ) (σ : ℝ) {i j : Fin n} (hij : i ≠ j) :
    ∃ k, (gramMatrix_isHermitian primes γ σ).eigenvalues k ≤
      (primes.map fun p =>
        xi_weight p σ * (1 - Real.cos ((γ i - γ j) * Real.log p))).sum := by
  obtain ⟨k, hk⟩ := exists_eigenvalue_le_pair_bound (gramMatrix_isHermitian primes γ σ) hij
  refine ⟨k, ?_⟩
  rw [gramMatrix_diag, gramMatrix_diag] at hk
  have hB : (primes.map fun p =>
      xi_weight p σ * (1 - Real.cos ((γ i - γ j) * Real.log p))).sum =
      (primes.map fun p => xi_weight p σ).sum -
      (primes.map fun p =>
        xi_weight p σ * Real.cos ((γ i - γ j) * Real.log p)).sum := by
    rw [← list_sum_map_sub]
    congr 1
    apply List.map_congr_left
    intro p _
    ring
  have hGij : gramMatrix primes γ σ i j =
      (primes.map fun p =>
        xi_weight p σ * Real.cos ((γ i - γ j) * Real.log p)).sum := rfl
  rw [hGij] at hk
  rw [hB]
  linarith

end Interlacing

#print axioms cosh_neg
#print axioms cosh_ge_one
#print axioms xi_weight_symm
#print axioms gram_entry_symm
#print axioms gram_matrix_symm
#print axioms exists_eigenvalue_mul_le_rayleigh
#print axioms exists_eigenvalue_le_pair_bound
#print axioms gram_eigenvalue_le_pair_bound

end
