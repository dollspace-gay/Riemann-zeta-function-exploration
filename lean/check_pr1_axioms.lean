import RHCrystal
import NymanBeurling
import NymanBeurlingL2
import NymanBeurlingMain
import WeilVerification
import Lean.Util.CollectAxioms

/-! Fail on any unexpected axiom in any project declaration, including private
and generated declarations. Run after `lake build` with
`lake env lean check_pr1_axioms.lean`.
-/

open Lean Elab Command in
run_elab do
  let modules : Array Name := #[`RHCrystal, `NymanBeurling, `NymanBeurlingL2,
    `NymanBeurlingMain, `WeilVerification]
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let env ← getEnv
  let mut counts : Array Nat := Array.replicate modules.size 0
  for (name, _) in env.constants do
    if let some idx := env.getModuleIdxFor? name then
      let owner := env.header.moduleNames[idx.toNat]!
      for i in [:modules.size] do
        if modules[i]! == owner then
          counts := counts.modify i (· + 1)
          for axiomName in (← collectAxioms name) do
            unless allowed.contains axiomName do
              throwError "Unexpected axiom {axiomName} in {name}"
  for i in [:modules.size] do
    if counts[i]! == 0 then
      throwError "No declarations found for {modules[i]!}"
    logInfo m!"PASS: {modules[i]!}: audited {counts[i]!} declarations"
  logInfo "Only propext, Classical.choice, Quot.sound permitted; no sorryAx."
