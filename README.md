# CUDA-accelerated ICP draft code

Procedural draft code for teammates to complete. ICP math remains TODO.

```text
src_cpu/             Independent CPU draft: structs, allocation, stage contracts, flow
src_gpu/             Independent CUDA draft: device allocation/copies, kernel placeholders
docs/                Draft guide, dataflow, terminology, ownership tasks and validation
scripts/             Existing data preparation and evaluation tools
tests/               Existing Python tool tests
data/kitti-sample/   Bundled scan pair
```

Each draft owns its header and build files. The root build selects `src_cpu/` by
default or `src_gpu/` with `-DICP_ENABLE_CUDA=ON`. The executable is `icp_run`.
Normal execution stops at the first `ICP_NOT_IMPLEMENTED` stage; it does not
report a registration result. There is no output writer or dry-run CLI.

See [draft guide](docs/DRAFT.md), [dataflow](docs/DATAFLOW.md),
[team tasks](docs/TASKS.md), and [validation](docs/DRAFT_VALIDATION.md).

The previous implementations remain available in Git history at
[`d46bb94`](https://github.com/sangjae4309/icp-cuda-accel/tree/d46bb9486a361bebdfdb6ab7ad84d9a561bf9b1b).

## How to contribute

We use topic branches and pull requests, inspired by
[gem5's contribution workflow](https://www.gem5.org/contributing).
Our integration branch is `main` (gem5 uses `develop`). Do not push work directly
onto `main`; submit a focused PR and obtain a teammate's review before merging.

| Item | Project convention | Example |
|---|---|---|
| Branch | `<area>/<short-change>` in lowercase kebab-case | `integration/update-draftcode` |
| Commit title | `<area>: <change>` | `integration: update draftcode` |
| PR title | Same format as commit title | `integration: update draftcode` |

The colon format comes from gem5's **commit** convention, not its branch naming.
The slash-based branch convention is our project rule: Git forbids colons and
spaces in branch names. Use areas such as `integration`, `transform`,
`correspondence`, `statistics`, `solver`, `validation`, or `docs`.

1. Pick a task from [TASKS.md](docs/TASKS.md) and coordinate its owner.
2. Update `main`, then create a branch for that task.
3. Implement and test the change; keep unrelated work in separate PRs.
4. Commit, push the branch, and open a pull request targeting `main`.
5. Describe the change and test evidence, request a teammate's review, and address
   feedback on the same branch. Merge only after approval and applicable checks.

```sh
git switch main
git pull --ff-only origin main
git switch -c integration/update-draftcode
# Edit files and run the relevant checks.
git add <changed-files>
git commit -m "integration: update draftcode"
git push -u origin integration/update-draftcode
```

On GitHub, choose **Compare & pull request**, set base `main` and compare
`integration/update-draftcode`, then use `integration: update draftcode` as the title.
With an authenticated GitHub CLI, the equivalent is:

```sh
gh pr create --base main --head integration/update-draftcode \
  --title "integration: update draftcode" --body-file /tmp/icp-pr.md
```

Write the PR description into `/tmp/icp-pr.md` first. See the full
[contribution guide](docs/CONTRIBUTING.md) for validation, review, and fork workflows.
