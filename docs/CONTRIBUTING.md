# Contributing

This project adapts [gem5's contribution workflow](https://www.gem5.org/contributing):
work on topic branches, submit focused pull requests, provide test evidence, and
respond to review. Our target is `main`; gem5's target is `develop`. The branch
prefixes and review policy below are local team conventions, not gem5 requirements.

## Naming

Branches use `<area>/<short-change>`: lowercase words, hyphens within the change,
and a slash after the area. Examples: `transform/add-cpu-transform`,
`correspondence/add-tiled-search`, `integration/update-draftcode`.
Validate with `git check-ref-format --branch integration/update-draftcode`.
Git forbids spaces and colons in branch names; see
[git-check-ref-format](https://git-scm.com/docs/git-check-ref-format).

Commit and PR titles use `<area>: <change>`. Start the change with an action such
as add, fix, update, or remove. Following gem5's commit convention, keep the commit
subject within 65 characters and optional body lines within 72 characters. Separate
the body with a blank line; explain the reason and link an issue when available.
Gem5 allows comma-separated component tags; this project normally uses one primary
area to keep ownership clear.

| Area | Responsibility |
|---|---|
| integration | Build, interfaces, memory ownership and execution flow |
| transform | CPU and GPU point transforms |
| correspondence | Matching, tie-breaking and threshold handling |
| statistics | Centroids, covariance and reductions |
| solver | Rigid update, pose composition and convergence |
| validation | Tests, correctness evidence and benchmarks |
| docs | Documentation |

## Workflow

Start from an up-to-date `main` in a clean working tree. Create a topic branch,
implement one coherent change, run relevant checks, and commit only intended files.
See the commands in [README](../README.md#how-to-contribute).

Collaborators may push their topic branch to this repository. Contributors without
write access should fork, branch from the upstream `main`, push to their fork, then
open a cross-fork PR against `sangjae4309/icp-cuda-accel:main`.

Use a draft PR for unfinished implementations. Review readiness is separate from
whether the project intentionally contains TODO functions: a completed draft-code
infrastructure change can be ready for review.

Before publishing, inspect `git diff --check`, `git diff --cached` and `git status`.
Exclude generated builds, results, private configuration and new bulk datasets.
Do not include unrelated changes or claim unexecuted tests passed.

## Validation and PR description

Run CPU/GPU builds and tests inside Slurm `interactive-gpu`; use one-minute jobs
for current small checks. Build commands are in [DRAFT.md](DRAFT.md).

- CPU changes: Release build, CTest, and existing Python tool tests.
- CUDA changes: inspect actual GPU/toolkit, select architecture explicitly, build
  and run GPU tests; use memcheck and synccheck where supported.
- Documentation-only changes: verify links, examples and whitespace; no need to
  rerun algorithm tests when executable code has not changed.
- Future algorithm changes: add applicable correctness cases from [TASKS.md](TASKS.md).
  Benchmark only after correctness with matching settings.

PR descriptions should state the problem, resulting behavior, affected modules,
exact test commands/results, and anything not tested. A skipped GPU test is not
passing evidence. Current TODO algorithms must continue returning
`ICP_NOT_IMPLEMENTED`; infrastructure tests cannot establish registration accuracy.

## Review and merge

Request at least one teammate's review. Address feedback by pushing further commits
to the same branch; GitHub updates the PR automatically. Do not merge your own PR
before approval. Merge after relevant checks pass and review comments are resolved;
use squash merge with a title following the convention, then delete the topic branch.
This guide does not configure GitHub branch protection or enforce these settings.

If main moves while you work, fetch and rebase your own topic branch onto
`origin/main`. Resolve conflicts and rerun affected checks. If a published branch
must be updated after a rebase, coordinate with collaborators and use
`git push --force-with-lease`; never force-push shared main.

## Coding boundaries

Keep simple structs, raw pointers, explicit sizes and separate host/device functions.
Maintain independent CPU and GPU drafts. Document buffer ownership and preserve
column-vector, left-composition and original-source conventions in
[DATAFLOW.md](DATAFLOW.md). Put documentation in `docs/`. Keep repository prose,
comments and reports in English.
