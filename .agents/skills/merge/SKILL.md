---
name: merge
description: >-
  Merge changes in this project only when the user has explicitly requested
  merging, such as through Merge Change, equivalent direct language, or an
  explicit /merge invocation. Do not invoke for review, preparation, passing
  checks, or skill installation alone.
disable-model-invocation: true
metadata:
  delta-action: land
---

# Merge changes

Carry out the user's explicit request to merge the requested changes in this
repository, from its current state through preparation, review, verification,
publication, merge, and confirmation. The request that invoked this skill is
the authorization to merge; do not ask again whether the user wants to merge
or say that you are waiting for an explicit merge request. Stop only for a
genuine blocker, unclear scope, or a permission gate that cannot be satisfied.

## Scope and safety

- Determine the change the user requested and the intended destination. The
  repository's default branch is currently `main`; verify the remote's current
  default branch and destination rules before proceeding.
- Inspect Git status, branch and upstream, remotes, and the change diff before
  modifying anything. Preserve unrelated work, including uncommitted files.
  Never stage, commit, reset, stash, or publish unrelated changes. If the
  requested change cannot be isolated safely, stop and ask how to proceed.
- Confirm the request is to merge the identified change, not merely review or
  prepare it. Do not infer scope from every change in the working tree.
- Inspect the latest repository contribution guidance, open PRs, destination
  branch settings/rules, and required reviews/checks. Follow applicable
  requirements, including any signing, contributor agreement, human-authored
  submission text, review/discussion, tests, changelog, documentation, or
  submission metadata requirements. Do not invent requirements or bypass
  applicable ones. Obtain required human-authored text from the user rather
  than drafting it on their behalf.
- Use non-interactive Git and hosting commands. Never force-push, rewrite
  shared history, disable checks, bypass review, or change destination
  protections.

## Prepare and propose

1. Check for local modifications and determine which belong to the requested
   change. Keep unrelated work untouched. If the user's requested change is
   not committed, prepare a focused branch and commit containing only the
   requested files and hunks. Use the repository's existing commit conventions;
   recent history uses Conventional Commit-style subjects such as
   `refactor: ...` and `feat: ...`. Do not amend or rewrite existing shared
   commits.
2. Update the feature branch with the destination branch using a safe,
   non-destructive method. If conflicts occur, resolve them automatically
   only when the intended result is clear from the change and surrounding
   code. Otherwise leave the conflict unresolved and ask the user to decide.
   Never discard unrelated edits to make a merge succeed.
3. Create or update a pull request targeting the verified default branch,
   unless an applicable current repository policy requires a different
   process. Include a concise factual title and description; satisfy any
   human-authorship rules by obtaining required text from the user. Do not
   duplicate an existing PR for the same change.

## Verify

- Check the README's provider-independent tests locally:

  ```sh
  uv run pytest tests/test_documents.py tests/test_rag.py tests/test_cli.py -q
  ```

  This command is documented in `README.md` under Tests. The project requires
  Python `>=3.14` as declared by `pyproject.toml`; use a compatible runtime.
- The README documents storage integration tests separately. When required by
  the change or repository policy and the test database is available, create
  and use only the dedicated `rag_test` database, never the application
  database:

  ```sh
  docker compose exec -T db createdb -U rag rag_test
  RAG_TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:6024/rag_test uv run pytest tests/test_storage.py -q
  ```

  The integration command follows `README.md`'s dedicated-database test
  procedure; use the actual test database URL configured for the environment
  rather than assuming credentials from this example.
- Inspect the current CI configuration and destination rules; this repository
  currently has no checked-in GitHub Actions workflow and no branch protection
  was found during setup. Do not assume this remains true. Before merging,
  verify every check required by the destination and applicable policy has
  passed for the exact PR head being merged, and that required reviews and
  discussions are complete. Pending, failing, missing, stale, or unverifiable
  required checks are not success; stop and report the blocker.
- Do not treat local tests as a substitute for required remote checks. If an
  optional check cannot run because a service or tool is unavailable, state
  that limitation and determine whether it is required before proceeding.

## Merge and confirm

When the PR is eligible under current repository rules and all required checks
and reviews are verified for its current head, merge it using the repository's
supported GitHub PR workflow and permitted merge method. Do not assume a
particular merge strategy is mandatory; inspect current repository settings
and use an allowed method consistent with existing practice.

After merging, verify that the PR is merged into the intended destination and
that the destination contains the merge result. If any step is blocked, do
not claim the change was merged; report the specific blocker and leave
unrelated work untouched.
