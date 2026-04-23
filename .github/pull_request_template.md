# What does this PR do?
<!-- One paragraph overview of the change and why it is needed. -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / code cleanup
- [ ] Documentation
- [ ] CI / tooling

## Changes summary
<!-- Bullet-point list of the concrete changes made. -->
-

## Testing
<!-- Describe how you tested this change. -->
- [ ] I ran the relevant unit tests locally (`pytest tests/unit_tests/`)
- [ ] I verified the change works end-to-end on a real training run
- [ ] No automated test covers this yet — I have verified it manually


### Pre-checks

- [ ] I have added relevant unit tests
- [ ] I have added relevant functional tests
- [ ] I have added proper typing to my code [Typing guidelines](https://docs.python.org/3/library/typing.html)
- [ ] I have added relevant documentation
- [ ] I have run the [autoformatter.sh](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/autoformat.sh) on my PR

## Testing
<!-- Describe how you tested this change. -->
- [ ] I ran the relevant unit tests locally (`pytest tests/unit_tests/`)
- [ ] I verified the change works end-to-end on a real training run
- [ ] No automated test covers this yet — I have verified it manually

## Checklist
- [ ] Code follows the project style (`ruff check megatron/` passes locally)
- [ ] New functionality includes tests, or I have explained why tests are not feasible
- [ ] Type hints are added for new public functions/classes
- [ ] I have rebased on the latest `main` and resolved any conflicts

### Code review

Feel free to message or comment the [@mcore-oncall](https://github.com/orgs/NVIDIA/teams/mcore-oncall) to help accelerate your merge into main. The less complex your PR is, the faster it will be approved and merged!

All PRs start as **draft**. If you open a non-draft PR, it will be automatically converted to draft.

#### Step 1: Mark PR as "Ready for Review"

1. When your PR is ready, click **Ready for Review**.
2. An oncall reviewer is auto-assigned and expert reviewers are notified based on your changes.
   - Some PRs may jump straight to step 2. This is determined by `.github/CODEOWNERS`.

:warning: Only mark as ready once merge-conflicts are resolved and the CI is passing.
Final Review might get declined if these requirements are not fulfilled.

#### Step 2: Final Review

For PRs that change `megatron/core`, once all expert reviewers have approved, the `Final Review` label is applied **automatically** and final reviewers are assigned.

For PRs outside `megatron/core`, this step is skipped.

#### Step 3: Approved

Once all required reviewers have approved, the `Approved` label is applied **automatically**.

### Merge

Any member of [mcore-engineers](https://github.com/orgs/NVIDIA/teams/mcore-engineers) will be able to merge your PR.

<details>
<summary>For MRs into `dev` branch</summary>
The proposed review process for `dev` branch is under active discussion.

MRs are mergable after one approval by either `eharper@nvidia.com` or `zijiey@nvidia.com`.
</details>
