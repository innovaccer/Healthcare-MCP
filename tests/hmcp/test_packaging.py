"""Packaging tests: metadata, version single-sourcing, py.typed and import health.

These tests are deliberately independent of the test suite's ``pythonpath=src``
setting, so they pass both against the source tree and against an installed
wheel.

``HMCP_PYPROJECT`` may point :data:`PYPROJECT` at a copy of ``pyproject.toml``;
it exists only so the distribution-name guard can be exercised against a renamed
copy without editing the real file.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import hmcp

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = Path(os.environ.get("HMCP_PYPROJECT", str(REPO_ROOT / "pyproject.toml")))
DIST_NAME = "innovaccer-hmcp"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _hmcp_from_site_packages() -> bool:
    """True when the imported ``hmcp`` resolves to an installed distribution."""
    return "site-packages" in Path(hmcp.__file__).resolve().parts


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that can still import this ``hmcp``.

    ``-I`` isolates the subprocess from the parent's environment (PYTHONPATH,
    user site); the package's own parent directory is added explicitly so the
    same code works whether ``hmcp`` comes from ``src`` or site-packages.
    """
    package_root = Path(hmcp.__file__).resolve().parent
    bootstrap = f"import sys; sys.path.insert(0, {str(package_root.parent)!r})\n"
    return subprocess.run(
        [sys.executable, "-I", "-c", bootstrap + code],
        capture_output=True,
        text=True,
    )


def test_distribution_name() -> None:
    """The distribution is named ``innovaccer-hmcp``; PyPI's ``hmcp`` is taken."""
    assert _pyproject()["project"]["name"] == DIST_NAME


def test_version_matches_hatch_source() -> None:
    data = _pyproject()
    project = data["project"]

    # The version is single-sourced: declared dynamic and read from a file.
    assert "version" in project.get("dynamic", [])
    assert "version" not in project

    hatch_version = data["tool"]["hatch"]["version"]
    version_path = REPO_ROOT / hatch_version["path"]
    assert version_path.name == "__init__.py"

    expected = None
    for line in version_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            expected = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    assert expected is not None, "no __version__ found in hatch version source"
    assert expected == hmcp.__version__


def test_installed_distribution_version_matches() -> None:
    try:
        installed = importlib.metadata.version(DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        # A src-only dev run imports hmcp without any installed distribution, so
        # there is nothing to compare against: skip. But if hmcp *is* importable
        # from site-packages, a distribution named ``innovaccer-hmcp`` must
        # provide it -- otherwise the name has been reverted or the package is
        # misnamed, which is a failure, not a skip.
        if _hmcp_from_site_packages():
            pytest.fail(
                f"{hmcp.__file__} is importable from site-packages, but no "
                f"distribution named {DIST_NAME!r} provides it"
            )
        pytest.skip(f"{DIST_NAME} is not installed")
    assert installed == hmcp.__version__


def test_py_typed_present() -> None:
    py_typed = Path(hmcp.__file__).resolve().parent / "py.typed"
    assert py_typed.is_file()


def test_server_imports_in_fresh_interpreter() -> None:
    result = _run_isolated(
        "from hmcp.server import HMCPServer; print(HMCPServer.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "HMCPServer"


def test_patient_context_before_server_has_no_cycle() -> None:
    result = _run_isolated(
        "import hmcp.server.patient_context\n"
        "from hmcp.server import HMCPServer\n"
        "print(HMCPServer.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "HMCPServer"
