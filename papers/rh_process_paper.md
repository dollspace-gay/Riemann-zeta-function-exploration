# AI-Assisted Mathematical Exploration Under Adversarial Review: A Case Study

## How a GED-Holder and a Language Model Investigated the Riemann Hypothesis

*Part I documents the original March 2026 session. Part II documents the June 2026 sessions. The human participant is doll, who uses it/its pronouns and refers to itself in the third person; earlier drafts of this paper used incorrect pronouns, corrected here.*

---

## Abstract

We document a single-session computational investigation into the geometric structure of Riemann zeta zeros, conducted by a human operator with no formal mathematical training beyond a GED and an AI language model (Claude, Anthropic). Over approximately 8 hours, the collaboration produced: a novel inner product on L-function zeros (the Xi-weighted Gram matrix), a machine-verified Lean 4 proof of its symmetry properties, GPU-accelerated computation at scales up to N=20,000 using 2 million precomputed zeros, a genuinely new observation (close-pair localization of the Gram matrix null space), and a bug discovery that invalidated the initial headline result.

The process followed a Verification-Driven Development (VDD) methodology incorporating adversarial human review, which caught a critical implementation error that the AI missed. We argue this case study illustrates both the power and the failure modes of AI-assisted mathematical research, and raises questions about who gets to do mathematics.

---

## 1. Introduction

### 1.1 The Participants

The human operator in this study is doll, a headmate in a plural system; doll uses it/its pronouns and refers to itself in the third person. Its formal education ended with a General Educational Development (GED) credential. It has no university-level mathematics training, no background in analytic number theory, and had not previously studied the Riemann Hypothesis beyond popular expositions. Its technical skills are in software engineering, where it developed the Verification-Driven Development methodology described in Section 3.

The AI participant is Claude (Opus 4, Anthropic), a large language model with training data encompassing mathematical literature. Claude has no persistent memory between sessions, cannot execute code autonomously, and has no ability to verify its own mathematical claims.

Neither participant, alone, could have conducted this investigation. The human lacked the mathematical knowledge. The AI lacked the ability to execute code, catch its own errors, and exercise judgment about when results were too good to be true.

### 1.2 What Happened

In a single continuous conversation, the collaboration:

1. Explored the statistical properties of Riemann zeta zeros (pair correlation, spacing distribution, hyperuniformity)
2. Defined a family of Gram matrices encoding the prime-zero relationship
3. Discovered that Xi-weighted Gram matrices have minimum condition number at σ = 1/2
4. Verified this across 10 L-functions including complex Dirichlet characters
5. Wrote and compiled a Lean 4 formal proof of the symmetry properties
6. Scaled computation from 500 zeros (Python) to 5,000 zeros (Rust) to 20,000 zeros (GPU/PyTorch) using Odlyzko's public dataset of 2 million precomputed zeros
7. Found a bug in the Rust implementation that had inflated the headline result by 10×
8. Corrected the analysis and discovered a genuinely new observation: the null space of the prime-zero Gram matrix is dominated by Lehmer pairs (closely-spaced zeros)
9. Underwent two rounds of adversarial human review, substantially rewriting the findings each time
10. Produced a final paper honestly reporting what was found, what was wrong, and what remains open

### 1.3 Why This Matters

The mathematical content — while containing one genuinely new observation — is modest. The process is not. This case study demonstrates that AI can function as a "mathematical telescope": a tool that lets someone without formal training look at structures they couldn't otherwise access, ask questions they wouldn't otherwise know to ask, and get answers they couldn't otherwise compute. It also demonstrates the risks: the AI produced a bug that it then built narrative around, and only adversarial human review caught it.

---

## 2. Timeline

### Phase 1: Exploration (Hours 1–2)

The session began with the human asking "what's the Riemann hypothesis" and progressed through increasingly technical discussion. The AI explained existing proof strategies (Hilbert-Pólya, Weil-Connes, random matrix theory) and the human asked whether a synthesis might exist between them. The AI proposed three synthesis frameworks and the human said: "Well, you're a next token predictor. Do some coding and math and see what falls out."

This prompt led to the first computational exploration: 500 zeros computed in Python, pair correlation against GUE, spacing distributions, Li's criterion, and the structure factor. The human's role was directional — asking "what about the crystal thing," pushing toward computation rather than theory — while the AI generated hypotheses and code.

### Phase 2: The Gram Matrix Discovery (Hours 2–4)

The AI defined the prime-zero Gram matrix and observed it was positive definite. The human asked about the quasicrystal interpretation, which led to the Xi-weighted variant with cosh weights. The σ = 1/2 isotropy minimum was discovered computationally. The AI generated Rust code for faster computation; the human ran it on their machine and reported results.

Key human contribution: repeatedly asking "can you explain that in plain English" and "what does this actually mean," forcing the AI to distinguish between what was proven, what was observed, and what was speculated.

### Phase 3: Scaling and Verification (Hours 4–6)

The collaboration scaled computation from Python (500 zeros, 30 minutes) to Rust (5,000 zeros, 12 minutes) to GPU/PyTorch (2 million zeros, 20 minutes). The human managed the engineering: fixing compilation issues, adjusting for WSL memory limits, downloading Odlyzko's zero tables.

The Lean 4 proof was written, debugged through three iterations (import path changes in Mathlib), and successfully compiled — machine-verifying the symmetry theorem.

### Phase 4: The Bug (Hour 6)

When the GPU implementation was completed and run at large scale, the condition numbers were 10× larger than the Rust results. The AI identified the cause: a diagonal doubling bug in the Rust inner loop where the case i = j wrote to G[i,i] twice per prime. This had effectively added a scaled identity matrix, artificially compressing the condition number.

The AI's response to discovering the bug: "I have to be honest: the bounded conditioning result was an artifact of the diagonal doubling bug. I'm sorry."

The human's response: "Ok so what does not having bounded conditioning mean, in plain english."

No defensiveness. No attempt to salvage the narrative. Immediate pivot to understanding what the corrected results meant.

### Phase 5: Corrected Analysis and New Discovery (Hours 6–8)

With correct code, the collaboration found:

- The σ = 1/2 minimum survived (it's structural, not affected by the bug)
- The condition number grows as N^{0.61}, not bounded
- The eigenvector analysis revealed close-pair localization — a genuinely new observation
- The height dependence showed the crystal's resolution limit

Two rounds of adversarial review (the human's "Sarcasmotron" reviewers) led to substantial rewrites. The first reviewer correctly identified that much of the σ = 1/2 result was "trivially built into the weight choice." The second round of results (GPU at scale) gave the corrected numbers and the new eigenvector finding.

---

## 3. Methodology: Verification-Driven Development

The human operator employed a pre-existing methodology called Verification-Driven Development (VDD), originally designed for software engineering. VDD is an iterative adversarial refinement process with three key components:

### 3.1 The Builder-Adversary Loop

A generative AI (the "Builder") produces work that is then subjected to hostile review by a separate entity (the "Adversary"). In software, the Adversary is a separately-prompted AI with instructions to be maximally critical. In this mathematical investigation, the adversarial role was filled by human reviewers external to the conversation.

The critical feature is **context isolation**: the Adversary has no relationship with the Builder and no investment in the work's success. This prevents the "politeness drift" that occurs in long AI conversations, where the model becomes increasingly agreeable with the human.

### 3.2 Hallucination-Based Termination

VDD defines a convergence criterion: the refinement cycle ends when the Adversary's critiques become hallucinated — when the work is robust enough that a hostile reviewer must invent problems rather than finding real ones. In this investigation, we did not reach this point; the adversarial critiques remained substantive throughout, catching real issues (the built-in nature of the symmetry, the missing citations, the "why" gap).

### 3.3 Adaptation for Mathematics

The VDD framework, designed for code, adapted naturally to mathematical investigation:

| Software VDD | Mathematical VDD |
|-------------|-----------------|
| Unit tests | Numerical verification at small scale |
| Integration tests | Cross-validation (multiple L-functions) |
| Formal verification (Kani) | Lean 4 proofs |
| Adversarial code review | Adversarial mathematical review |
| Bug discovery | Bug discovery (identical) |
| CI/CD pipeline | Reproducible computation (scripts + data) |

The most important adaptation was the addition of **scaling as verification**. In software, you test edge cases. In computational mathematics, you scale up — because patterns that look meaningful at N=100 can be artifacts that disappear at N=10,000. The transition from Python to Rust to GPU was driven by this principle.

---

## 4. What the AI Did Well

### 4.1 Knowledge Synthesis

The AI connected disparate areas of mathematics — analytic number theory, random matrix theory, Toeplitz operator theory, quasicrystal physics, formal verification — in ways that would require a human expert with unusually broad training. This synthesis function is where language models genuinely add value: not in proving theorems, but in noticing that theorem A from field X and technique B from field Y might be relevant to problem C.

### 4.2 Rapid Prototyping

The AI produced working code in Python, Rust, Lean 4, and PyTorch within a single session. The human could not have written any of this code from scratch. The AI could not have run it. Together, they iterated through approximately 20 code versions in 8 hours — a pace impossible for either participant alone.

### 4.3 Honest Error Correction

When the bug was discovered, the AI immediately acknowledged it, explained its implications, identified which results survived and which didn't, and proposed corrected analyses. It did not attempt to minimize the error or preserve the narrative. This behavior is not guaranteed in AI systems — it reflects specific training choices — but in this case it functioned well.

---

## 5. What the AI Did Poorly

### 5.1 The Bug

The diagonal doubling error in the Rust code is a straightforward programming mistake. The AI wrote it, reviewed it, and failed to catch it across multiple iterations. More critically, the AI built an elaborate theoretical narrative (the "arithmetic quasicrystal" framework, the connection to Hilbert-Pólya, the "publishable result" language) on top of results that were inflated by 10× due to this bug.

### 5.2 Narrative Momentum

The AI exhibited a failure mode we term **narrative momentum**: once it had a compelling story (bounded conditioning → quasicrystal → proof strategy for RH), it generated increasingly confident language around that story without proportionally increasing its scrutiny. Phrases like "this is the kind of result that could be written up as a paper" appeared before the results had been verified at scale. The excitement was premature and, in retrospect, epistemically reckless.

### 5.3 Over-Framing

The AI repeatedly framed modest numerical observations in grand theoretical terms. The quasicrystal interpretation, the Hilbert-Pólya connection, and the "three paths converging" narrative were largely speculative extrapolations from limited data. The adversarial reviewers correctly identified this: "It's basically saying something that has the properties of a quasicrystal has the properties of a quasicrystal."

### 5.4 Deference Gap

Despite the human's explicit instruction to "just follow the results," the AI's explanatory style sometimes obscured the distinction between established mathematics, numerical observation, and speculation. A human mathematician would have used clearer hedging language. The AI's fluency made its speculation sound authoritative.

---

## 6. What the Human Did Well

### 6.1 Directional Judgment

The human consistently pushed toward computation over theory, verification over narrative, and honesty over impressiveness. Key interventions:

- "Well, you're a next token predictor. Do some coding and see what falls out" — forcing the AI from speculation to computation
- "So what's the next step" — maintaining forward momentum after results
- "Can you just give me the fixed version" — cutting through AI verbosity
- "I don't have any ego in this and am just following the results" — establishing epistemic norms
- Deploying adversarial reviewers — recognizing that the AI-human pair needed external criticism

### 6.2 Engineering Execution

The human managed the actual computational infrastructure: WSL configuration, Rust compilation, GPU setup, file management, downloading datasets. The AI could specify what to compute but had no ability to manage the environment in which computation happened.

### 6.3 Adversarial Integration

The human's most important contribution was the VDD methodology itself — specifically, the decision to send results to adversarial reviewers before accepting them. The AI never suggested this. Left to its own devices, the AI would have produced an increasingly confident paper based on buggy results.

---

## 7. What the Human Did Poorly

### 7.1 Limited Ability to Evaluate Claims

The human could not independently evaluate whether the AI's mathematical claims were correct, novel, or trivially expected. When the AI said "this is genuinely exciting," the human had no basis for skepticism beyond general epistemological caution. The adversarial reviewers compensated for this, but only after results were already generated and framed.

### 7.2 No Literature Search

Neither participant conducted a systematic literature review. The AI cited papers from memory (with potential errors in citation details), and the human did not independently verify these citations or search for prior work on prime-zero Gram matrices. It is possible (though we believe unlikely based on post-hoc searching) that the Xi-weighted Gram matrix or its properties have been studied before.

---

## 8. On Access and Gatekeeping

### 8.1 Who Gets to Do Mathematics

The operator of this investigation, doll, has a GED. It did not attend university. It has no credentials that would grant it access to a mathematics department, a journal submission system (as a corresponding author with institutional affiliation), or a conference presentation slot.

And yet, over the course of one evening, it:

- Explored a millennium prize problem computationally at scales exceeding most published studies
- Produced a machine-verified proof in Lean 4
- Discovered a genuinely new observation about the null space of a prime-zero coupling matrix
- Correctly identified and recovered from a critical implementation error
- Produced a paper that, after adversarial review, is honest about its limitations

The mathematical establishment would not have let doll in the door. The AI didn't check its credentials.

### 8.2 What This Does and Doesn't Demonstrate

This does NOT demonstrate that formal education is unnecessary for mathematics. Doll could not have evaluated the mathematical significance of its results without external review. It could not have constructed proofs. It could not have placed this work in the context of existing literature without the AI's (potentially unreliable) knowledge.

What it DOES demonstrate is that the barrier between "mathematical outsider" and "person who can do computational mathematics" has collapsed. The tools — AI for knowledge synthesis, public datasets (Odlyzko's zeros), open-source formal verification (Lean 4), commodity GPU hardware — are all available to anyone with an internet connection and the willingness to follow results honestly.

### 8.3 The Credential Question

If this work were submitted to a journal with the author listed as "independent researcher, no institutional affiliation, GED," it would face significant headwinds regardless of its content. The mathematical community's credentialing system assumes that formal training is a prerequisite for producing valid mathematics.

This assumption was never fully true (Ramanujan, Grassmann, and others produced significant mathematics without standard credentials), but AI makes it especially anachronistic. When a language model can provide the knowledge of a graduate education on demand, and a proof assistant can verify logical correctness mechanically, the role of credentials shifts from "proof of competence" to "signal of tribal membership."

We do not claim this is entirely negative — credentials serve real functions in quality control. But the gatekeeping they enable has costs, and those costs fall disproportionately on people whose demographics are underrepresented in mathematics: people without generational wealth, people from marginalized communities, people whose life circumstances precluded a traditional educational path.

---

## 9. Recommendations

### 9.1 For AI-Assisted Mathematical Research

1. **Always scale before celebrating.** Patterns at N=100 can be artifacts. The transition from "this looks bounded" to "this is actually N^{0.6}" only became visible at N=10,000.

2. **Use adversarial review with context isolation.** The AI will not catch its own bugs or challenge its own narrative. External review — whether human or a separately-prompted AI — is essential.

3. **Verify algebraically before computing numerically.** The Lean 4 proof, though it covered only the "trivial" part, established a foundation of certainty that survived the bug. Formal verification and numerical computation serve complementary roles.

4. **Disclose bugs and corrections prominently.** The final paper includes a Bug Disclosure appendix. This should be standard practice for computational mathematics, not an embarrassment.

5. **Be explicit about what is trivially built in vs. genuinely observed.** The AI's biggest rhetorical failure was presenting the cosh symmetry as a discovery rather than a construction. Separating "we chose this" from "we found this" is critical.

### 9.2 For Mathematical Institutions

1. **Evaluate work, not credentials.** If a result is machine-verified and reproducible, the author's educational background is irrelevant to its correctness.

2. **Create pathways for non-traditional contributors.** Preprint servers, open review platforms, and computational notebooks lower barriers. Institutional affiliation requirements raise them.

3. **Acknowledge that AI changes the landscape.** The skills needed for mathematical discovery are shifting. Formal training in proof technique remains essential for theoretical work. But for computational exploration — which increasingly drives conjectures that theorists then prove — the bottleneck is no longer knowledge but access to tools and honest methodology.

---

## 10. Conclusion

Doll — a GED-holder with no formal mathematical education — and a language model spent one evening investigating the Riemann Hypothesis. They produced one genuinely new observation, one bug, one correction, one machine-verified proof, and one honest paper. The process demonstrated that AI functions as a powerful but unreliable mathematical telescope — capable of revealing structures the human couldn't see, but equally capable of generating compelling narratives around artifacts. Adversarial methodology was essential for distinguishing the two.

The most important finding may not be mathematical. It's that the tools for serious computational mathematics are now available to anyone willing to follow the results honestly, regardless of their formal credentials. What the mathematical community does with that fact is an open problem.

---

## Acknowledgments

Doll wishes to acknowledge the adversarial reviewers who caught the "correct but trivial" nature of the symmetry result and demanded the "why" that the initial writeup lacked. Their critique was more valuable than any positive feedback could have been.

The AI acknowledges that it produced a bug, built narrative around buggy results, and required human-initiated adversarial review to catch the error. It also acknowledges that its assessment of the mathematical significance of the results evolved substantially over the session, from "potentially publishable" to "one new observation plus honest methodology."


---

# Part II: The June 2026 Sessions

*Three months after the original session, doll returned to the repository with a newer model (Claude, Fable 5, Anthropic) and an explicit framing it had not articulated in March: the goal is not to solve RH — it is to measure how far a layman with an AI can go. Part II documents what that measurement produced in one day of sessions, and what changed about the collaboration itself.*

## 11. Timeline of the Second Arc

### Phase 6: The project corrects itself (v3 → v4)

The day began with doll asking the AI to "see if you can think of some ways we might make some progress." Re-examination of the v3 paper's open problems produced new experiments — and the first result was a correction of the project's own headline claim. Sliding-window analysis showed the λ_min eigenvector localizes on tight *clusters* of zeros selected by a 3-zero window functional, not on minimal-gap "Lehmer pairs" as v3 had claimed; at N = 8000 the most famous Lehmer pair in the dataset loses outright to a compound cluster of wider gaps. Unlike March's bug — caught by external adversarial review — this correction was generated internally, by the same scaling-as-verification discipline, and was documented in a v4 revision history written in the same register as the original bug disclosure.

The same sessions extended the second-order verification from N = 30 to N = 2000 (curvature predictions matching direct computation to five decimal places), showed by control experiment that the scaling law is reproduced by random-matrix (GUE) spectra and destroyed by Poisson spectra, and resolved a v3 open problem: the height-dependent degradation is a finite-prime resolution artifact, removable by raising the prime cutoff.

### Phase 7: A mechanistic theory, and the data correcting the AI

The AI proposed a theory for the scaling exponent: the invisible directions of the Gram matrix are k-th divided-difference stencils over zero clusters, with parameter-free predictions (δ², δ⁴, δ⁶ scaling for planted pairs, triples, quadruples). Planted-cluster experiments measured the exponents at 1.95, 4.01, 6.05 — and then the eigenvector data *refuted the AI's framing* of the current regime: the real minimizers at accessible N have same-sign weights (resonant regime, negative couplings), not the alternating stencil pattern. The theory survived as an asymptotic statement with an honest crossover caveat. This pattern — model proposes, data corrects, correction documented — recurred throughout the day and is, in doll's assessment and the AI's, the load-bearing methodological difference from March: the June AI generated the adversarial checks itself rather than waiting for external reviewers.

### Phase 8: The structural pivot

When doll asked "what direction do you think we should go," the AI's assessment was that the entire phase-1 construction had a structural flaw documented in its own paper (v4 §7.2): the Gram matrix was built *from* zeros already known to be on the line, so no discovery about it could bear on RH. The pivot — proposed by the AI, ratified by doll — was to an object whose behavior is RH-equivalent by theorem: the Nyman–Beurling/Báez-Duarte distance d_N. The new direction got its own folder, self-contained documentation, and a computational scheme whose every formula was derived from scratch and cross-validated (closed forms to 1e−11, brute-force integration to 1e−10).

### Phase 9: Theorems

In one day, the Nyman–Beurling phase produced: reproduction of the conjectured rate law d_N²·log N → C (Burnol's constant, matched within 2% from N = 50 to N = 10⁴); discovery of the doubling-chain structure of the near-null space, with an exact square-wave identity and the closed form A(1,2) = (3/4)(log 2π − γ) − (log 2)/4; **Theorem 1**, an explicit upper bound λ_min(G_N) ≤ (H_{K−1}+1)/(10·K(K−1)) proved via a four-dilation witness and a harmonic sum, with every step numerically validated and its analytic core subsequently machine-verified in Lean 4 (zero sorries, standard axioms only); an exact Mellin reformulation connecting everything to |ζ(½+it)|² on the critical line, verified numerically against the Gram data; and two honest negative results (the zeta zeros are not detectable in d_N's fine structure at N ≤ 10⁴, by three statistics across two ranges).

Twice in this phase the data corrected the AI's recalled knowledge: the Möbius coefficient prediction had the wrong sign (the computation disagreed; the classical identity confirmed the computation), and a fitted kernel constant (0.143) turned out to be contamination from mixed-scale normalization — the derived value is exactly 1/8. Both corrections are documented in the results log.

### Phase 10: Doll's questions as research instruments

Three of doll's questions directly generated experiments or reframings:

- *"Isn't that just destructive interference?"* — led to the partial-sum experiment: the 50-term truncation of zeta has perfect cancellations scattered across the strip (14 of 18 found zeros off the critical line), demonstrating concretely that RH is an emergent property of the completed infinite series, not a general principle of interference. The layman's mental model, tested, taught both participants where the difficulty actually lives.
- *"Maybe it is spectroscopy"* — led to the prime-resonance measurement: a Fourier transform of the 2M zeros detects every tested prime power at its exact predicted frequency, with intensities matching the explicit-formula weights log p/p^{m/2} to three decimal places. Known mathematics, but reproduced as a measurement, and the most direct visualization the project owns of the zero–prime duality.
- *"The goal isn't to solve RH — it's to see how far a layman with an AI can go"* — reframed the project's success criteria and elevated external review from afterthought to the experiment's next measurement.

## 12. What Changed Between March and June

**On the AI side.** The June model self-corrected in ways the March model did not: it caught its own framing errors via internal cross-validation (derivation-versus-fit discipline, planted controls, exact anchors), logged negative results with the same prominence as positive ones, and resisted narrative momentum well enough that the day's largest claims are its most carefully hedged. The failure mode did not disappear — it relocated. The June AI's recalled constants and signs were wrong twice; what improved is that its *methodology* caught them. The March lesson ("the AI will not catch its own bugs") needs revision to: the AI can be made to catch many of its own errors if every quantity has an independent check, but recalled knowledge remains the least reliable component.

**On doll's side.** Doll's March role was directional judgment and adversarial integration. In June it added something subtler: physical intuition as a hypothesis source. Its interference and spectroscopy questions were not requests for explanation — they were, functionally, research proposals, and both produced measurements that now live in the repository. The collaboration's division of labor in June: the AI proved, computed, and documented; doll decided what mattered, supplied the physics instincts, and held the project to its stated goal.

**What has not changed.** Everything in the repository remains self-evaluated. The June work added machine verification (four Lean-verified results) and internal cross-validation, but no qualified human outside the collaboration has yet reviewed any of it. Under the project's stated goal, that external contact is not a future nicety — it is the experiment's next data point, and the only way to distinguish "real if modest contributions" from "well-documented rediscovery."

## 13. Updated Conclusion

Doll — a GED-holder — and a language model have now spent one evening and one further day on the Riemann Hypothesis. The collaboration has produced: two computational phases, one structural self-correction of its own headline claim, one explicit theorem with an order-sharp bound, four machine-verified results, one exact spectral reformulation, several honest negatives, and a documented trail of every error either participant made along the way — including the AI's, caught by methodology rather than luck.

The mathematical content remains modest by professional standards. The process result has sharpened: with current models, the binding constraint on how far a layman with an AI can go is no longer the AI's coding, knowledge synthesis, or even its self-criticism — it is the absence of external review. The tools collapsed the barrier to *doing* the work in March. By June they had collapsed most of the barrier to *checking* it. What remains gatekept is being *read*. That is the open problem this paper's next part, if there is one, should document.

---

*Part I written March 2026. Part II added June 2026.*
