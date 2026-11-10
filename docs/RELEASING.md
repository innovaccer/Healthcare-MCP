# Releasing HMCP

This guide is for maintainers of the `innovaccer-hmcp` distribution (import name
`hmcp`). Releases are published to [PyPI](https://pypi.org/project/innovaccer-hmcp/)
from GitHub Actions using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC). No API tokens are stored or rotated.

> **Important:** `pip install hmcp` installs an unrelated project that happens to own
> the name on PyPI. The HMCP distribution is `innovaccer-hmcp`; always install it by
> that name.

## One-time setup

1. **PyPI pending publisher:** Sign in to [PyPI](https://pypi.org/manage/account/publishing/)
   and add a pending trusted publisher for a project that does not exist yet:
   - Project name: `innovaccer-hmcp`
   - Owner: `innovaccer`
   - Repository: `Healthcare-MCP`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

2. **TestPyPI pending publisher:** Repeat on [TestPyPI](https://test.pypi.org/manage/account/publishing/)
   with the same values except environment name `testpypi`.

3. **GitHub environments:** In the repository, create two environments under
   *Settings → Environments*:
   - `pypi`: add required reviewers (release approval gate).
   - `testpypi`: add required reviewers (optional, but recommended).

   The publish jobs reference these environments, so the OIDC subject claims match
   the pending publishers above.

## Release checklist

1. **Bump the version:** Edit `__version__` in `src/hmcp/__init__.py` (the single
   source of truth; `pyproject.toml` reads it via `[tool.hatch.version]`).

2. **Date the changelog:** The current release section in `CHANGELOG.md` is headed
   `## [X.Y.Z] - Unreleased`. The date is set when the release is tagged: replace
   `Unreleased` with the release date, e.g. `## [0.1.0] - 2026-11-09`.

3. **Merge:** Open a pull request with those two changes, have it reviewed, and merge
   to `main`.

4. **Create the GitHub release:** Draft a new release with tag `vX.Y.Z` (matching the
   version exactly, e.g. version `0.2.0` → tag `v0.2.0`), targeting `main`. Publishing
   the release triggers `release.yml`; the build job fails if the tag does not equal
   `v` + the version read from the built wheel's `METADATA`.

5. **Approve the `pypi` environment:** When the `publish-pypi` job reaches the
   environment gate, approve it as a required reviewer. The job then uploads the
   distributions to PyPI via Trusted Publishing.

6. **Verify:** In a fresh virtual environment:

   ```sh
   python -m venv /tmp/hmcp-verify
   /tmp/hmcp-verify/bin/python -m pip install "innovaccer-hmcp==X.Y.Z"
   /tmp/hmcp-verify/bin/python -c "import hmcp; print(hmcp.__version__)"
   ```

## Dry run on TestPyPI

TestPyPI **rejects re-uploading a version that already exists on TestPyPI**, so a dry
run must use a throwaway pre-release version. A dry run is started by pushing a
pre-release tag, and it can be run from any branch before the release is merged:

1. On top of the commit to be tested, create a commit that sets `__version__` to a
   release-candidate or development version (for example `0.2.0rc1` or `0.2.0.dev1`).
   The commit does not need to be on a branch.
2. Tag that commit `v` + the version (for example `v0.2.0rc1`) and push only the tag
   (`git push origin v0.2.0rc1`). Tags that match `v*rc*` or `v*.dev*` trigger the
   **Release** workflow, and the build job fails if the tag does not equal `v` + the
   version read from the built wheel's `METADATA`.
3. Approve the `testpypi` environment when the `publish-testpypi` job reaches the
   gate. The job then uploads the distributions to TestPyPI via Trusted Publishing.
   A pre-release tag never publishes to PyPI.
4. Verify from TestPyPI:

   ```sh
   python -m venv /tmp/hmcp-testpypi
   /tmp/hmcp-testpypi/bin/python -m pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     "innovaccer-hmcp==X.Y.Zrc1"
   ```

5. Before the real release, set `__version__` to the final version, update the
   changelog, and follow the release checklist above.

Once `release.yml` is on `main`, the **Release** workflow can also be run manually
(*Actions → Release → Run workflow*) for a selected branch; that run uploads the
version on that branch to TestPyPI. GitHub offers manual runs only for workflows that
exist on the default branch.
