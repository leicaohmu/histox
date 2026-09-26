#!/usr/bin/env python3
"""Reject stale Slideflow product identities in maintained HistoX sources."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ROOT / "histox",
    ROOT / "scripts",
    ROOT / "docs" / "source",
    ROOT / "docs-source" / "source",
    ROOT / ".github",
)
SCAN_FILES = (
    ROOT / ".gitignore",
    ROOT / ".gitmodules",
    ROOT / "docs" / "CNAME",
    ROOT / "README.md",
    ROOT / "ROADMAP.md",
    ROOT / "setup.py",
)
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".ini",
    ".jinja",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# These identifiers name real upstream distributions that are still used as
# optional compatibility dependencies. They must not be silently renamed to
# packages that do not exist.
ALLOWED_IDENTIFIERS = (
    "slideflow-noncommercial",
    "slideflow_noncommercial",
    "slideflow-gpl",
    "slideflow_gpl",
)
ALLOWED_UPSTREAM_MARKERS = (
    "github.com/slideflow/",
    "media.githubusercontent.com/media/slideflow/slideflow",
    "github.com/jamesdolezal/stylegan2-slideflow",
    "github.com/jamesdolezal/stylegan3-slideflow",
    "hub.docker.com/repository/docker/jamesdolezal/slideflow",
)
ALLOWED_PATHS = {
    Path("histox/assets/branding/README.md"),
}
PRODUCT_PATTERN = re.compile(r"slideflow", re.IGNORECASE)
LEGACY_ALIAS_PATTERNS = (
    re.compile(r"\bSF_[A-Z0-9_]+\b"),
    re.compile(r"\bimport\s+histox\s+as\s+sf\b"),
    re.compile(r"\bsf\."),
    re.compile(r"\bsf_[A-Za-z0-9_]*\b"),
)


def iter_text_files():
    """Yield maintained repository text files in deterministic order."""
    files = set(SCAN_FILES)
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        files.update(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES
            and "__pycache__" not in path.parts
        )
    yield from sorted(files)


def strip_allowed_identifiers(line: str) -> str:
    """Remove approved compatibility package identifiers before matching."""
    if any(marker in line.lower() for marker in ALLOWED_UPSTREAM_MARKERS):
        return ""
    for identifier in ALLOWED_IDENTIFIERS:
        line = re.sub(re.escape(identifier), "", line, flags=re.IGNORECASE)
    return line


def main() -> int:
    """Return non-zero when a stale product identity or path is found."""
    violations = []
    for path in iter_text_files():
        if not path.exists():
            continue
        relative = path.relative_to(ROOT)
        if relative in ALLOWED_PATHS or relative == Path("scripts/check_identity.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            checked_line = strip_allowed_identifiers(line)
            if PRODUCT_PATTERN.search(checked_line) or any(
                pattern.search(checked_line) for pattern in LEGACY_ALIAS_PATTERNS
            ):
                violations.append(f"{relative}:{line_number}: {line.strip()}")

    named_paths = [
        path.relative_to(ROOT)
        for root in SCAN_ROOTS
        if root.exists()
        for path in root.rglob("*")
        if "slideflow" in path.name.lower() and "__pycache__" not in path.parts
    ]

    if violations or named_paths:
        print("Stale Slideflow product identity detected:")
        for violation in violations:
            print(f"  {violation}")
        for path in named_paths:
            print(f"  stale path: {path}")
        return 1

    print("HistoX identity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
