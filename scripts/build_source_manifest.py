"""Build a source-only SHA-256 manifest on the approved server.

This script deliberately rejects generated research directories and secret-like
paths. Its output is a runtime receipt and should not be committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a_share_research.experiments.source_evidence import source_files, source_manifest_payload

__all__ = ["source_files", "source_manifest_payload"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = source_manifest_payload(root)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
