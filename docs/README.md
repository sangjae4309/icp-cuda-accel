# Documentation map

This tree retains non-agent project guides and immutable historical validation.
Local agent instructions, memory and workflows are not part of the published
project documentation. Use the guides below for source layout, design and results.

| Existing category | Purpose |
| --- | --- |
| [CPU feature ICP](feature/README.md) | Implemented feature baseline, LM equations and numerical checks |
| [Vanilla source layout](vanilla/README.md) | CPU/CUDA source routing |
| [CPU pair validation](CPU_PAIR_VALIDATION.md) | Measured CPU build and final-pose alignment on the bundled pair |
| [Existing optimizations](VANILLA_OPTIMIZATIONS.md) | Smith commit attribution, tiled target reuse and cost model |
| [Vanilla design](VANILLA_DESIGN.md) | PDF five-step mapping, proposed owners, GPU memory and thread diagrams |
| [Draft contracts](DRAFT.md) | Historical scaffold description and build interfaces; see current status below |
| [Dataflow](DATAFLOW.md) | CPU/CUDA stages, buffers and ownership |
| [Glossary](GLOSSARY.md) | Existing ICP terminology |
| [Team tasks](TASKS.md) | Assigned implementation boundaries |
| [Recorded draft validation](DRAFT_VALIDATION.md) | Historical infrastructure checks and limits |
| [Historical validation](HISTORICAL_VALIDATION.md) | Earlier verification evidence |
| [Contributing](CONTRIBUTING.md) | Branches, reviews and validation policy |


Detailed design and task contracts stay here. Historical validation records are
evidence of their original scope, not current algorithm correctness or GPU availability.
Use the measured comparison below when assessing current execution evidence.

Current source includes all five vanilla stages and final evaluation. CPU and
CUDA execution was measured in the [full-cloud comparison](CPU_CUDA_COMPARISON.md);
the later KD-tree and voxel-grid additions are unimplemented placeholders.

- [CPU accuracy review and saved-pose images](ICP_ACCURACY_REVIEW.md).

- [Full-cloud CPU/CUDA accuracy and latency](CPU_CUDA_COMPARISON.md).
