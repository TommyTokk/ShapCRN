"""Validate the contents of built ShapCRN wheel and source archives."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def _members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"Unsupported distribution archive: {path}")


def validate(path: Path) -> None:
    members = _members(path)
    parts = [PurePosixPath(member).parts for member in members]
    if any("models" in member_parts or "results" in member_parts for member_parts in parts):
        raise AssertionError(f"Repository models/results leaked into {path.name}")
    if not any(member.endswith("shapcrn/api.py") for member in members):
        raise AssertionError(f"Public API is missing from {path.name}")
    if not any(member.endswith("LICENSE") for member in members):
        raise AssertionError(f"MIT license is missing from {path.name}")


def main(arguments: list[str]) -> int:
    paths = [Path(argument) for argument in arguments]
    if not paths:
        raise SystemExit("Pass wheel and sdist paths")
    for path in paths:
        validate(path)
        print(f"Validated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
