"""Foundation checks; execute only on the approved server."""

from pathlib import Path

import pytest
from scripts.build_source_manifest import source_files

import a_share_research
from a_share_research.contracts import ContractError


def test_package_has_version() -> None:
    assert a_share_research.__version__ == "0.1.0"


def test_generated_research_directories_are_absent_from_source() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = {
        "data",
        "artifacts",
        "runs",
        "results",
        "logs",
        "checkpoints",
        "weights",
        "predictions",
    }
    assert forbidden.isdisjoint(path.name for path in root.iterdir())


def test_spec_snapshots_are_present() -> None:
    root = Path(__file__).resolve().parents[2]
    specs = root / "docs" / "specs"
    assert (specs / "PRODUCT_SPEC.snapshot.md").is_file()
    assert (specs / "TECH_SPEC.snapshot.md").is_file()
    assert (specs / "SNAPSHOT_MANIFEST.md").is_file()


def test_source_manifest_fails_when_forbidden_directory_is_present(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("source\n", encoding="utf-8")
    forbidden = tmp_path / "data"
    forbidden.mkdir()
    (forbidden / "leak.csv").write_text("secret research output\n", encoding="utf-8")
    with pytest.raises(ContractError, match="forbidden source paths"):
        source_files(tmp_path)


def test_source_manifest_ignores_interpreter_cache(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("source\n", encoding="utf-8")
    cache = tmp_path / "src" / "package" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.cpython-312.pyc").write_bytes(b"runtime cache")
    assert [path.relative_to(tmp_path).as_posix() for path in source_files(tmp_path)] == [
        "README.md"
    ]
