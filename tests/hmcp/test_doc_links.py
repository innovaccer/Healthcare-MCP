"""Documentation link guards.

README.md is the PyPI long description. PyPI resolves relative links against
pypi.org, so a repository-relative link in README.md would break there. The
repository keeps relative links (they work on GitHub); the
``hatch-fancy-pypi-readme`` metadata hook rewrites them to absolute GitHub URLs
in the published long description only.

The helpers below take text or a path, so they can be exercised against mutated
copies without touching the tracked files.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DIST_NAME = "innovaccer-hmcp"

# Markdown files in scope: the top-level docs plus everything under these trees.
# ``.github/``, ``tasks/``, ``venv/`` and ``dist/`` are intentionally excluded.
_TOP_LEVEL_MD = ("README.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md")
_MD_TREES = ("docs", "examples", "src/hmcp")

# The sdist include list in pyproject.toml deliberately omits ``images/`` (the
# multi-megabyte screenshots) and ``.github/``. A relative target whose first
# path component is one of these is skipped *only* when that directory is absent
# -- i.e. from an unpacked sdist. In a git checkout the directories exist and the
# targets are checked as usual.
SDIST_OMITTED_TOP_DIRS = ("images", ".github")

_BLOB_PREFIX = "https://github.com/innovaccer/Healthcare-MCP/blob/main/"
_RAW_PREFIX = "https://raw.githubusercontent.com/innovaccer/Healthcare-MCP/main/"

_INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`+([^`\n]*)`+")
_EXTERNAL_RE = re.compile(r"^(?:https?://|mailto:)", re.IGNORECASE)
_SELF_LINK_RE = re.compile(
    r"(?:github\.com/innovaccer/Healthcare-MCP/(?:blob|tree)/"
    r"|raw\.githubusercontent\.com/innovaccer/Healthcare-MCP/)"
)
# A relative target the hook rewrites must be a plain path (optionally with a
# ``#fragment``): no angle brackets, titles, leading ``/`` or spaces.
_SAFE_TARGET_RE = re.compile(r"^[A-Za-z0-9._-][A-Za-z0-9._/-]*(#[A-Za-z0-9._-]+)?$")


def strip_fenced_code(text: str) -> str:
    """Return ``text`` with fenced code blocks (``` / ~~~) removed."""
    kept: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)[0]
                continue
            kept.append(line)
        elif match and match.group(1)[0] == fence:
            fence = None
    return "\n".join(kept)


def fenced_code_blocks(text: str) -> list[str]:
    """Return the contents of fenced code blocks (``` / ~~~)."""
    blocks: list[str] = []
    fence: str | None = None
    current: list[str] = []
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)[0]
                current = []
            continue
        if match and match.group(1)[0] == fence:
            blocks.append("\n".join(current))
            fence = None
            continue
        current.append(line)
    if fence is not None:
        blocks.append("\n".join(current))
    return blocks


def inline_code_spans(text: str) -> list[str]:
    """Return the contents of inline code spans outside fenced blocks."""
    return _INLINE_CODE_RE.findall(strip_fenced_code(text))


def code_regions(text: str) -> list[str]:
    """Return code-block and inline-code contents, where ``](`` is unsafe.

    The metadata hook rewrites the whole README text, including code, so a
    ``](`` inside code would be corrupted in the PyPI description.
    """
    return fenced_code_blocks(text) + inline_code_spans(text)


def raw_link_targets(text: str) -> list[tuple[bool, str]]:
    """Return ``(is_image, raw_target)`` with no normalization.

    Fenced code blocks are ignored. Unlike :func:`link_targets` the target is
    returned exactly as written (angle brackets, titles and spaces included).
    """
    return [
        (match.group(0).startswith("!"), match.group(1))
        for match in _INLINE_LINK_RE.finditer(strip_fenced_code(text))
    ]


def link_targets(text: str) -> list[tuple[bool, str]]:
    """Return ``(is_image, target)`` for every inline link/image target.

    Fenced code blocks are ignored. Reference-style links are not used in this
    repository and are not matched. Angle-bracket wrapping and a trailing title
    are removed so the target can be resolved.
    """
    targets: list[tuple[bool, str]] = []
    for match in _INLINE_LINK_RE.finditer(strip_fenced_code(text)):
        raw = match.group(1).strip()
        if raw.startswith("<") and ">" in raw:
            raw = raw[1 : raw.index(">")]
        elif " " in raw:  # a trailing title: ``[x](url "title")``
            raw = raw.split(None, 1)[0]
        if raw:
            targets.append((match.group(0).startswith("!"), raw))
    return targets


def is_external(target: str) -> bool:
    """True for ``http(s)://`` and ``mailto:`` targets."""
    return bool(_EXTERNAL_RE.match(target))


def is_omitted_sdist_path(repo_root: Path, candidate: Path) -> bool:
    """True for a target under a directory the sdist deliberately omits.

    ``candidate`` is the target resolved against ``repo_root``. It is omitted
    only when its first path component is in :data:`SDIST_OMITTED_TOP_DIRS` and
    that top-level directory does not exist under ``repo_root`` -- i.e. from an
    unpacked sdist, not from a git checkout.
    """
    try:
        rel = candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    if not rel.parts:
        return False
    first = rel.parts[0]
    return first in SDIST_OMITTED_TOP_DIRS and not (repo_root / first).exists()


def missing_relative_targets(md_path: Path, text: str, repo_root: Path) -> list[str]:
    """Return relative link/image targets that resolve to no file or directory.

    External targets, bare ``#anchors`` and empty targets are skipped. A
    ``#fragment`` is stripped before resolution; anchors themselves are not
    checked. A leading ``/`` is resolved against the repository root. Targets
    under a directory the sdist omits (and that is absent here) are skipped.
    """
    missing: list[str] = []
    for _is_image, target in link_targets(text):
        if is_external(target) or target.startswith("#"):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        if path_part.startswith("/"):
            candidate = repo_root / path_part.lstrip("/")
        else:
            candidate = md_path.parent / path_part
        if is_omitted_sdist_path(repo_root, candidate):
            continue
        if not candidate.exists():
            missing.append(target)
    return missing


def forbidden_self_links(text: str) -> list[str]:
    """Return targets that point at files in this repo via an absolute URL.

    The CI badge (``actions/workflows/...``) is not a file link and is allowed.
    """
    return [
        target
        for _is_image, target in link_targets(text)
        if _SELF_LINK_RE.search(target)
    ]


def non_http_targets(text: str) -> list[str]:
    """Return link/image targets that are neither external nor a bare anchor.

    External means exactly what :func:`is_external` allows (``http(s)://`` and
    ``mailto:``); bare ``#anchors`` are allowed too.
    """
    return [
        target
        for _is_image, target in link_targets(text)
        if not is_external(target) and not target.startswith("#")
    ]


def rewritten_targets_not_in_repo(text: str, repo_root: Path) -> list[str]:
    """Return rewritten GitHub/raw URLs whose path does not exist in the repo.

    Each rewritten URL is stripped of its prefix and any ``#fragment`` and the
    remainder must name an existing path (a target under an omitted-and-absent
    sdist directory is skipped). This catches a malformed substitution such as
    ``<``, a ``#`` image or a doubled ``//``.
    """
    bad: list[str] = []
    for _is_image, target in link_targets(text):
        if target.startswith(_BLOB_PREFIX):
            path_part = target[len(_BLOB_PREFIX) :]
        elif target.startswith(_RAW_PREFIX):
            path_part = target[len(_RAW_PREFIX) :]
        else:
            continue
        path_part = path_part.split("#", 1)[0]
        if not path_part:
            bad.append(target)
            continue
        candidate = repo_root / path_part
        if is_omitted_sdist_path(repo_root, candidate):
            continue
        if not candidate.exists():
            bad.append(target)
    return bad


def apply_readme_substitutions(text: str, pyproject_path: Path) -> str:
    """Apply the ``fancy-pypi-readme`` substitutions from ``pyproject_path``.

    Mirrors ``hatch_fancy_pypi_readme._substitutions``: each substitution
    compiles its pattern (case-insensitively when ``ignore-case`` is set) and is
    applied in the declared order.
    """
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    hook = data["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    for cfg in hook.get("substitutions", []):
        flags = re.IGNORECASE if cfg.get("ignore-case") else 0
        text = re.compile(cfg["pattern"], flags).sub(cfg["replacement"], text)
    return text


def markdown_files() -> list[Path]:
    """Return the in-scope markdown files under the repository root."""
    files = [REPO_ROOT / name for name in _TOP_LEVEL_MD if (REPO_ROOT / name).is_file()]
    for tree in _MD_TREES:
        root = REPO_ROOT / tree
        if root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
    return files


def test_relative_links_resolve() -> None:
    """Every relative link/image target resolves to a real file or directory."""
    failures: dict[str, list[str]] = {}
    for md in markdown_files():
        missing = missing_relative_targets(
            md, md.read_text(encoding="utf-8"), REPO_ROOT
        )
        if missing:
            failures[str(md.relative_to(REPO_ROOT))] = missing
    assert not failures, f"relative link targets do not exist: {failures}"


def test_no_absolute_self_links() -> None:
    """Links to files in this repo are relative, never absolute GitHub URLs."""
    offenders: dict[str, list[str]] = {}
    for md in markdown_files():
        found = forbidden_self_links(md.read_text(encoding="utf-8"))
        if found:
            offenders[str(md.relative_to(REPO_ROOT))] = found
    assert not offenders, f"links to files in this repo must be relative: {offenders}"


def test_readme_link_forms_are_rewrite_safe() -> None:
    """README targets the hook rewrites must be plain, resolvable paths."""
    text = README.read_text(encoding="utf-8")
    for _is_image, target in raw_link_targets(text):
        if is_external(target) or target.startswith("#"):
            continue
        assert _SAFE_TARGET_RE.match(
            target
        ), f"unsafe link target in README.md: {target!r}"
    for region in code_regions(text):
        assert "](" not in region, (
            "'](' inside a code block or inline span would be rewritten in the "
            f"PyPI description: {region!r}"
        )


def test_readme_substitutions_produce_absolute_links() -> None:
    """The metadata hook rewrites every README target to an absolute URL."""
    text = apply_readme_substitutions(README.read_text(encoding="utf-8"), PYPROJECT)
    leftovers = non_http_targets(text)
    assert not leftovers, f"substitutions left relative targets: {leftovers}"

    targets = [target for _is_image, target in link_targets(text)]
    assert any(
        t.startswith(_BLOB_PREFIX) for t in targets
    ), "no rewritten blob/main link found; the substitution is a no-op"
    images = [t for is_image, t in link_targets(text) if is_image]
    assert any(
        t.startswith(_RAW_PREFIX) for t in images
    ), "no rewritten raw.githubusercontent image found; the substitution is a no-op"

    unresolved = rewritten_targets_not_in_repo(text, REPO_ROOT)
    assert not unresolved, f"rewritten URLs do not map back to repo paths: {unresolved}"


def test_published_long_description_is_absolute() -> None:
    """The installed distribution's long description carries absolute links."""
    try:
        metadata = importlib.metadata.metadata(DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{DIST_NAME} is not installed")

    long_description = metadata.get_payload()
    assert long_description.strip(), "the long description is empty"
    assert (
        "github.com/innovaccer/Healthcare-MCP/blob/main/" in long_description
    ), "the long description has no rewritten blob/main link"
    leftovers = non_http_targets(long_description)
    assert not leftovers, f"relative link targets survived into metadata: {leftovers}"
