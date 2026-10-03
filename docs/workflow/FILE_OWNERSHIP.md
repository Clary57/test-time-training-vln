# Research File Ownership

Ownership identifies the role responsible for a proposed change; it does not override explicit user instructions or the evidence requirements in [WORKFLOW.md](WORKFLOW.md).

## Main Scientist primarily owns

- [CURRENT_STATUS.md](../../CURRENT_STATUS.md)
- [PROBLEM.md](../research/PROBLEM.md)
- [HYPOTHESES.md](../research/HYPOTHESES.md)
- [FINDINGS.md](../research/FINDINGS.md)
- [PAPER_STORY.md](../research/PAPER_STORY.md)
- [NEXT_EXPERIMENT.md](../research/NEXT_EXPERIMENT.md)
- [LITERATURE_MAP.md](../literature/LITERATURE_MAP.md)

## Reviewer proposes changes to

- [ATTACK_LOG.md](../research/ATTACK_LOG.md)

The Reviewer normally operates read-only. Reviewer suggestions should be reviewed by the Main Scientist before being accepted into the research state.

## Server Experiment Engineer primarily owns factual updates to

- [RESULTS_INDEX.md](../experiments/RESULTS_INDEX.md)
- Experiment manifests.
- Experiment logs and validation outputs.
- Experiment-specific HANDOFF files.
- Reproducibility records.

## Source code ownership

The Server Experiment Engineer modifies source code and experiment implementation when required by a defined [NEXT_EXPERIMENT.md](../research/NEXT_EXPERIMENT.md) and the current task authorization.

## Restrictions

The Server Experiment Engineer must not independently edit `HYPOTHESES.md`, `PAPER_STORY.md`, or `PROBLEM.md`. The Reviewer must not edit implementation files. The Main Scientist should not directly modify implementation while acting in the scientific reasoning role.
