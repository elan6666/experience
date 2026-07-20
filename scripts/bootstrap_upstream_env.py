#!/usr/bin/env python3
"""Build exact or CUDA-12.8 compatibility-smoke environments on server."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from a_share_research.provenance import (
    EMPTY_STDERR_SHA256,
    RECEIPT_SCHEMA_VERSION,
    error_digest,
    write_receipt,
)

APPROVED_PREFIX = Path("/data/yilangliu/a_share_research")
ALLOWED = {"itransformer", "fact", "timexer", "timepro", "s4m"}


def _approved(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != APPROVED_PREFIX and APPROVED_PREFIX not in resolved.parents:
        raise ValueError(f"refusing non-server path: {resolved}")
    return resolved


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _run(*argv: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _base(model: str, mode: str, stage: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "environment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "status": "ENV_FAIL",
        "stage": stage,
        "stderr_digest": EMPTY_STDERR_SHA256,
        "command": _command(),
        "provenance_status": "SERVER_ENVIRONMENT",
        "environment_mode": mode,
        "requirements_sha256": None,
        "resolved_lock_sha256": None,
        "resolved_lock_path": None,
        "python_version": None,
        "torch_version": None,
        "cuda_version": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(ALLOWED))
    parser.add_argument("--mode", choices=["exact", "compat-smoke"], required=True)
    parser.add_argument("--python-bin", required=True)
    environment = parser.add_mutually_exclusive_group(required=True)
    environment.add_argument("--env-root", type=Path)
    environment.add_argument(
        "--existing-environment",
        type=Path,
        help="Audit and lock an already-built server environment without reinstalling.",
    )
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = _approved(args.receipt)
    stage = "select_requirements"
    receipt = _base(args.model, args.mode, stage)
    try:
        project_root = Path(__file__).resolve().parents[1]
        filename = (
            "upstream-requirements.txt"
            if args.mode == "exact"
            else "compat-smoke-requirements.txt"
        )
        requirements = project_root / "envs" / args.model / filename
        requirements_bytes = requirements.read_bytes()
        receipt["requirements_sha256"] = hashlib.sha256(requirements_bytes).hexdigest()

        stage = "select_environment"
        lock_root = _approved(args.lock_root)
        if args.existing_environment is not None:
            destination = _approved(args.existing_environment)
            if not (destination / "bin" / "python").is_file():
                raise FileNotFoundError(
                    f"existing environment has no Python executable: {destination}"
                )
        else:
            stage = "create_venv"
            assert args.env_root is not None
            env_root = _approved(args.env_root)
            destination = env_root / f"{args.model}-{args.mode}"
            if destination.exists():
                raise FileExistsError(f"environment already exists: {destination}")
            env_root.mkdir(parents=True, exist_ok=True)
            _run(args.python_bin, "-m", "venv", str(destination))
            python = destination / "bin" / "python"
            stage = "upgrade_pip"
            _run(str(python), "-m", "pip", "install", "--upgrade", "pip")
            stage = "install_requirements"
            _run(str(python), "-m", "pip", "install", "--requirement", str(requirements))
        lock_root.mkdir(parents=True, exist_ok=True)
        python = destination / "bin" / "python"

        stage = "capture_resolved_lock"
        freeze = _run(str(python), "-m", "pip", "freeze", "--all")
        index_lines = [
            line
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("--")
        ]
        lock_text = "\n".join([*index_lines, *freeze.splitlines()]) + "\n"
        lock_path = lock_root / f"{args.model}-{args.mode}.resolved.txt"
        lock_path.write_text(lock_text, encoding="utf-8")
        lock_hash = hashlib.sha256(lock_text.encode("utf-8")).hexdigest()

        stage = "verify_runtime"
        probe_code = (
            "import json,platform,torch;"
            "print(json.dumps({'python':platform.python_version(),"
            "'torch':torch.__version__,'cuda':torch.version.cuda}))"
        )
        runtime = json.loads(_run(str(python), "-c", probe_code))
        if args.mode == "compat-smoke" and not str(runtime["cuda"]).startswith("12.8"):
            raise RuntimeError(f"compat-smoke requires cu128, got {runtime['cuda']}")

        receipt.update(
            {
                "status": "PASS" if args.mode == "exact" else "PASS_WITH_WARNING",
                "stage": "complete",
                "stderr_digest": EMPTY_STDERR_SHA256,
                "resolved_lock_sha256": lock_hash,
                "resolved_lock_path": str(lock_path),
                "python_version": runtime["python"],
                "torch_version": runtime["torch"],
                "cuda_version": runtime["cuda"],
                "environment_path": str(destination),
                "semantic_scope": (
                    "author exact dependency declaration"
                    if args.mode == "exact"
                    else "CUDA-12.8 import/shape smoke only; not reproduction training"
                ),
            }
        )
        stage = "write_receipt"
        write_receipt(receipt_path, receipt)
        return 0
    except Exception as error:
        receipt["status"] = "ENV_FAIL"
        receipt["stage"] = stage
        receipt["stderr_digest"] = error_digest(error)
        write_receipt(receipt_path, receipt)
        print(f"environment bootstrap failed at {stage}; see receipt digest", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
