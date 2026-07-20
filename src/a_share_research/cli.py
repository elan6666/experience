"""Minimal CLI that exposes package metadata without starting research work."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from a_share_research import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the metadata-only command parser."""
    parser = argparse.ArgumentParser(prog="a-share-research")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse metadata options; research commands are intentionally absent."""
    build_parser().parse_args(argv)
    return 0
