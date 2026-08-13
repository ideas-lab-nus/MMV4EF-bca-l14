#!/usr/bin/env python3
"""Verify the SHA-256 digest of every released model artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    checked = 0
    for line_number, line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            failures.append(f"line {line_number}: malformed checksum record")
            continue
        path = ROOT / relative.strip()
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected.lower():
            failures.append(f"digest mismatch: {relative}")
        checked += 1

    if failures:
        print("Model verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"Verified {checked} released model artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

