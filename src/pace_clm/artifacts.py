"""Small provenance helpers used by every future run artifact."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: str | Path) -> dict[str, object]:
    root = Path(root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}
