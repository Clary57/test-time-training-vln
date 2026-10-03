# Research Workflow

## Purpose

This project separates scientific reasoning from engineering execution. The workflow is hypothesis-driven and attack-driven. The role assignments below are a handoff contract; they do not replace the evidence and authorization required for a particular experiment.

## Roles

### Main Scientist

Tool: ChatGPT. See [MAIN_SCIENTIST.md](MAIN_SCIENTIST.md).

Responsible for scientific reasoning, interpreting experiment results, generating competing hypotheses, deciding the next scientific question, designing falsification experiments, updating beliefs, maintaining the scientific story, and determining what the evidence actually supports.

Must not treat metric improvement as sufficient scientific evidence, directly optimize implementation details while acting in this role, or invent causal claims unsupported by evidence.

### Reviewer / Attack

Tool: Local Codex Desktop. See [REVIEWER.md](REVIEWER.md).

Responsible for attacking current claims, identifying confounders and alternative explanations, checking causal support, identifying the most dangerous uncertainty, challenging the paper story, and identifying missing controls.

Default behavior: **READ ONLY**. Must not modify source code, optimize the current method, defend the existing story by default, or redesign the method before identifying the uncertainty.

### Experiment Engineer

Tool: Server Codex. See [EXPERIMENT_ENGINEER.md](EXPERIMENT_ENGINEER.md).

Responsible for implementing the experiment defined in [NEXT_EXPERIMENT.md](../research/NEXT_EXPERIMENT.md), debugging and running it, preserving reproducibility, collecting factual evidence, and recording configs, commands, seeds, commits, outputs, and failures.

Must not decide research direction, invent scientific hypotheses, rewrite the paper story, optimize metrics unless explicitly requested, or silently change the scientific question.

## Research Loop

Observation → competing hypotheses → reviewer attack → scientific question → falsification experiment → `NEXT_EXPERIMENT.md` → Server Codex implementation → validated result → `RESULTS_INDEX.md` → `FINDINGS.md` → Main Scientist updates beliefs → repeat.

The Main Scientist writes a concrete experiment specification before handing work to the Experiment Engineer. The Engineer records what actually ran and reports deviations or failures. The Main Scientist interprets the validated evidence after reviewer scrutiny; a runtime result does not update a hypothesis or paper claim by itself.

The current [NEXT_EXPERIMENT.md](../research/NEXT_EXPERIMENT.md) is empty. **No next experiment has been selected.**

## Pre-Experiment Rule

Before any significant experiment, the following must be answerable:

1. What hypothesis are we testing?
2. What result do we predict?
3. What result would weaken or reject the hypothesis?

If these cannot be answered, do not run the experiment. Use [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) to record the question, controls, falsification criteria, and required outputs; the template is not itself an experiment authorization.

## Evidence Hierarchy

From strongest to weakest:

1. Validated experiment outputs.
2. Controlled or paired comparisons, within their protocol boundaries.
3. Diagnostic analyses.
4. Empirical observations.
5. Interpretations.
6. Hypotheses.
7. Ideas.

Do not promote a lower-level statement into a higher-level fact without evidence. A paired comparison is not automatically causal: its controls, data role, validation, and confounders still matter.

## Source of Truth

Canonical research state:

- [CURRENT_STATUS.md](../../CURRENT_STATUS.md)
- [PROBLEM.md](../research/PROBLEM.md)
- [HYPOTHESES.md](../research/HYPOTHESES.md)
- [FINDINGS.md](../research/FINDINGS.md)
- [PAPER_STORY.md](../research/PAPER_STORY.md)
- [ATTACK_LOG.md](../research/ATTACK_LOG.md)
- [NEXT_EXPERIMENT.md](../research/NEXT_EXPERIMENT.md)
- [LITERATURE_MAP.md](../literature/LITERATURE_MAP.md)
- [RESULTS_INDEX.md](../experiments/RESULTS_INDEX.md)

Experiment manifests, validation reports, and raw outputs remain the evidence for individual runs. Git synchronizes the local and server copies. File responsibilities are defined in [FILE_OWNERSHIP.md](FILE_OWNERSHIP.md).
