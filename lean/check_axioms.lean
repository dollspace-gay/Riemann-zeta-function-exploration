import RHCrystal
import NymanBeurling
import NymanBeurlingL2
import NymanBeurlingMain
import JumpLocal
import JumpArithmetic
import JumpSynthesis
import Refinement
import Annihilator
import Lean.Util.CollectAxioms

/-! Audit every declaration belonging to the project modules, including private
and generated declarations. Unexpected axioms fail the command, not just print.
Run after `lake build`: `lake env lean check_axioms.lean`.
-/

open Lean Elab Command in
run_elab do
  let modules : Array Name := #[`RHCrystal, `NymanBeurling, `NymanBeurlingL2,
    `NymanBeurlingMain, `JumpLocal, `JumpArithmetic, `JumpSynthesis, `Refinement, `Annihilator]
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let env ← getEnv
  let mut names : Array Name := #[]
  for (name, _) in env.constants do
    if let some idx := env.getModuleIdxFor? name then
      if modules.contains env.header.moduleNames[idx.toNat]! then
        names := names.push name
  if names.isEmpty then throwError "No project declarations found"
  for name in names.qsort Name.quickLt do
    let axioms ← collectAxioms name
    for axiomName in axioms do
      unless allowed.contains axiomName do
        throwError "Unexpected axiom {axiomName} in {name}"
  logInfo m!"PASS: audited {names.size} declarations from {modules.size} project modules; only propext, Classical.choice, Quot.sound permitted."
