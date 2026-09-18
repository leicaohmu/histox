#!/usr/bin/env python3
"""Validate HistoX distribution contents before publication."""

from pathlib import Path, PurePosixPath
import tarfile
import zipfile


MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
FORBIDDEN_SDIST_PREFIXES = (
    "build/",
    "datasets/",
    "dist/",
    "docs/",
    "docs-source/",
)
REQUIRED_SDIST_PATHS = {
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "histox/norm/norm_tile.jpg",
    "licenses/MIT-HistoBistro.txt",
    "licenses/MIT-StainTools.txt",
    "licenses/MIT-Transformer-Explainability.txt",
    "pyproject.toml",
    "setup.py",
}
REQUIRED_WHEEL_SUFFIXES = {
    "/LICENSE",
    "/MIT-HistoBistro.txt",
    "/MIT-StainTools.txt",
    "/MIT-Transformer-Explainability.txt",
    "/THIRD_PARTY_NOTICES.md",
}


def _single_artifact(dist_dir, pattern):
    artifacts = sorted(dist_dir.glob(pattern))
    if len(artifacts) != 1:
        raise RuntimeError(
            "Expected exactly one {!r} artifact in {}, found {}".format(
                pattern, dist_dir, len(artifacts)
            )
        )
    return artifacts[0]


def _check_size(path):
    size = path.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise RuntimeError(
            "{} is {:.2f} MiB; limit is {:.2f} MiB".format(
                path.name,
                size / 1024**2,
                MAX_ARTIFACT_BYTES / 1024**2,
            )
        )
    print("{}: {:.2f} MiB".format(path.name, size / 1024**2))


def _sdist_paths(path):
    with tarfile.open(path, "r:gz") as archive:
        members = [PurePosixPath(member.name) for member in archive.getmembers()]
    roots = {member.parts[0] for member in members if member.parts}
    if len(roots) != 1:
        raise RuntimeError("Expected one top-level sdist directory, found {}".format(roots))
    return {
        PurePosixPath(*member.parts[1:]).as_posix()
        for member in members
        if len(member.parts) > 1
    }


def _check_sdist(path):
    paths = _sdist_paths(path)
    forbidden = sorted(
        item
        for item in paths
        if any(item == prefix[:-1] or item.startswith(prefix)
               for prefix in FORBIDDEN_SDIST_PREFIXES)
    )
    if forbidden:
        raise RuntimeError(
            "Source distribution contains forbidden paths: {}".format(
                ", ".join(forbidden[:10])
            )
        )
    missing = sorted(REQUIRED_SDIST_PATHS - paths)
    if missing:
        raise RuntimeError(
            "Source distribution is missing required paths: {}".format(
                ", ".join(missing)
            )
        )


def _check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    if "histox/norm/norm_tile.jpg" not in names:
        raise RuntimeError("Wheel is missing histox/norm/norm_tile.jpg")
    missing = sorted(
        suffix
        for suffix in REQUIRED_WHEEL_SUFFIXES
        if not any(name.endswith(suffix) for name in names)
    )
    if missing:
        raise RuntimeError(
            "Wheel is missing required license files: {}".format(
                ", ".join(missing)
            )
        )


def main():
    dist_dir = Path("dist")
    wheel = _single_artifact(dist_dir, "*.whl")
    sdist = _single_artifact(dist_dir, "*.tar.gz")
    for artifact in (wheel, sdist):
        _check_size(artifact)
    _check_wheel(wheel)
    _check_sdist(sdist)
    print("Distribution content checks passed.")


if __name__ == "__main__":
    main()
