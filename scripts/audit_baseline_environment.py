#!/usr/bin/env python3
"""Audit the locked Ridge or LightGBM baseline environment on the server."""

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

import yaml

from a_share_research.provenance import (
    EMPTY_STDERR_SHA256,
    RECEIPT_SCHEMA_VERSION,
    assert_registry,
    error_digest,
    write_receipt,
)

APPROVED_PREFIX = Path("/data/yilangliu/a_share_research")
ALLOWED = {"ridge", "lightgbm"}


def _approved(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != APPROVED_PREFIX and APPROVED_PREFIX not in resolved.parents:
        raise ValueError(f"refusing non-server path: {resolved}")
    return resolved


def _run(*argv: str) -> str:
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _base(model: str, stage: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "environment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "status": "ENV_FAIL",
        "stage": stage,
        "stderr_digest": EMPTY_STDERR_SHA256,
        "command": _command(),
        "provenance_status": "BASELINE_ENVIRONMENT",
        "environment_mode": "baseline-smoke",
        "requirements_sha256": None,
        "resolved_lock_sha256": None,
        "resolved_lock_path": None,
        "python_version": None,
        "torch_version": None,
        "cuda_version": None,
    }


def _probe_code(model: str) -> str:
    if model == "ridge":
        return (
            "import json,platform,sklearn;"
            "from sklearn.linear_model import Ridge;"
            "m=Ridge(alpha=1.0).fit([[0.,0.],[1.,1.],[2.,1.]],[0.,1.,2.]);"
            "p=m.predict([[1.,0.]]);"
            "print(json.dumps({'python':platform.python_version(),"
            "'package':'scikit-learn','version':sklearn.__version__,"
            "'output_shape':list(p.shape)}))"
        )
    return (
        "import json,platform,lightgbm;"
        "from lightgbm import LGBMRegressor;"
        "m=LGBMRegressor(n_estimators=2,verbosity=-1,random_state=20260719)"
        ".fit([[0.,0.],[1.,1.],[2.,1.],[3.,2.]],[0.,1.,2.,3.]);"
        "p=m.predict([[1.,0.]]);"
        "print(json.dumps({'python':platform.python_version(),"
        "'package':'lightgbm','version':lightgbm.__version__,"
        "'output_shape':list(p.shape)}))"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(ALLOWED))
    parser.add_argument("--registry", type=Path, default=Path("configs/upstreams.lock.yaml"))
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = _approved(args.receipt)
    stage = "load_registry"
    receipt = _base(args.model, stage)
    try:
        document: dict[str, Any] = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
        assert_registry(document)
        entry = document["baselines"][args.model]
        expected_version = entry["package"].split("==", 1)[1]

        stage = "select_environment"
        environment = _approved(args.environment)
        python = environment / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"baseline environment has no Python: {environment}")
        requirements = Path(__file__).resolve().parents[1] / "envs/baselines/requirements.txt"
        requirements_bytes = requirements.read_bytes()
        receipt["requirements_sha256"] = hashlib.sha256(requirements_bytes).hexdigest()

        stage = "capture_resolved_lock"
        lock_root = _approved(args.lock_root)
        lock_root.mkdir(parents=True, exist_ok=True)
        freeze = _run(str(python), "-m", "pip", "freeze", "--all") + "\n"
        lock_path = lock_root / "baselines.resolved.txt"
        lock_path.write_text(freeze, encoding="utf-8")
        lock_hash = hashlib.sha256(freeze.encode("utf-8")).hexdigest()

        stage = "fit_predict_smoke"
        probe = json.loads(_run(str(python), "-c", _probe_code(args.model)))
        if probe["version"] != expected_version:
            raise RuntimeError(
                f"{probe['package']} version mismatch: {probe['version']} != {expected_version}"
            )
        if probe["output_shape"] != [1]:
            raise RuntimeError(f"unexpected prediction shape: {probe['output_shape']}")

        receipt.update(
            {
                "status": "PASS",
                "stage": "complete",
                "stderr_digest": EMPTY_STDERR_SHA256,
                "resolved_lock_sha256": lock_hash,
                "resolved_lock_path": str(lock_path),
                "python_version": probe["python"],
                "torch_version": "not-applicable",
                "cuda_version": "not-applicable",
                "environment_path": str(environment),
                "package": probe["package"],
                "package_version": probe["version"],
                "output_shape": probe["output_shape"],
                "scope": "synthetic package fit/predict smoke; not A-share training",
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
        print(f"baseline audit failed at {stage}; see receipt digest", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
