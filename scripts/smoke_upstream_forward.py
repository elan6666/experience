#!/usr/bin/env python3
"""Minimal forward/backward and physical-GPU identity smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from a_share_research.provenance import (
    EMPTY_STDERR_SHA256,
    RECEIPT_SCHEMA_VERSION,
    error_digest,
    git_status_porcelain,
    validate_receipt,
    worktree_content_sha256,
    write_receipt,
)

APPROVED_PREFIX = Path("/data/yilangliu/a_share_research")
ALLOWED = {"itransformer", "fact"}


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


def _git_state(upstream: Path) -> tuple[str, str]:
    return (
        _run("git", "rev-parse", "HEAD", cwd=upstream),
        _run("git", "rev-parse", "HEAD^{tree}", cwd=upstream),
    )


def _gpu_inventory() -> list[dict[str, Any]]:
    output = _run(
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name",
        "--format=csv,noheader,nounits",
    )
    rows: list[dict[str, Any]] = []
    for fields in csv.reader(io.StringIO(output)):
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {fields}")
        rows.append(
            {
                "physical_index": int(fields[0].strip()),
                "uuid": fields[1].strip(),
                "pci_bus_id": fields[2].strip(),
                "name": fields[3].strip(),
            }
        )
    return rows


def _visible_indices(cuda_visible: str, inventory: list[dict[str, Any]]) -> list[int]:
    tokens = [token.strip() for token in cuda_visible.split(",") if token.strip()]
    if not tokens:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name at least one physical GPU")
    by_uuid = {row["uuid"]: row["physical_index"] for row in inventory}
    physical: list[int] = []
    for token in tokens:
        if token.isdigit():
            physical.append(int(token))
        elif token in by_uuid:
            physical.append(int(by_uuid[token]))
        else:
            raise RuntimeError(f"unsupported CUDA_VISIBLE_DEVICES token: {token}")
    return physical


def _process_gpu_uuids(pid: int) -> list[str]:
    output = _run(
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    )
    matches: list[str] = []
    for fields in csv.reader(io.StringIO(output)):
        if len(fields) >= 2 and fields[0].strip() == str(pid):
            matches.append(fields[1].strip())
    return matches


def _itransformer(torch: Any, device: Any) -> tuple[Any, tuple[int, ...]]:
    module = importlib.import_module("model.iTransformer")
    config = SimpleNamespace(
        seq_len=8,
        pred_len=2,
        output_attention=False,
        use_norm=1,
        embed="timeF",
        freq="h",
        dropout=0.0,
        class_strategy="projection",
        d_model=16,
        factor=1,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        activation="gelu",
    )
    model = module.Model(config).to(device)
    # Match ordinary author training: model parameters require gradients, while
    # input batches do not.  Requiring input gradients trips iTransformer's
    # in-place normalization even though that path is irrelevant to training.
    x = torch.randn(2, 8, 9, device=device)
    return model(x, None, None, None), (2, 2, 9)


def _fact(torch: Any, device: Any) -> tuple[Any, tuple[int, ...]]:
    module = importlib.import_module("models.FACT")
    config = SimpleNamespace(
        task_name="long_term_forecast",
        pred_len=2,
        seq_len=8,
        enc_in=9,
        use_norm=1,
        freq="n",
        d_model=8,
        dilation=[1],
        num_kernels=2,
        d_ff=16,
        core=0.5,
        dropout=0.0,
    )
    model = module.Model(config).to(device)
    x = torch.randn(2, 8, 9, device=device)
    return model(x, None, None, None), (2, 2, 9)


def _base(model: str, status: str, stage: str, digest: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_type": "smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "status": status,
        "stage": stage,
        "stderr_digest": digest,
        "command": _command(),
        "provenance_status": None,
        "commit": None,
        "license_status": None,
        "source_tree_hash_before": None,
        "source_tree_hash_after": None,
        "worktree_content_sha256_before": None,
        "worktree_content_sha256_after": None,
        "git_status_before": None,
        "git_status_after": None,
        "python_version": sys.version,
        "torch_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_current_device": None,
        "physical_gpu_requested": None,
        "physical_gpu_evidence": None,
        "environment_receipt_sha256": None,
        "resolved_lock_sha256": None,
        "sys_executable": os.path.abspath(sys.executable),
        "output_shape": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(ALLOWED))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkout-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    receipt_path = _approved(args.receipt)
    stage = "validate_checkout_receipt"
    receipt = _base(args.model, "SMOKE_FAIL", stage, EMPTY_STDERR_SHA256)
    receipt["physical_gpu_requested"] = args.physical_gpu
    try:
        upstream = _approved(args.upstream)
        checkout_receipt_path = _approved(args.checkout_receipt)
        environment_receipt_path = _approved(args.environment_receipt)
        if upstream == receipt_path or upstream in receipt_path.parents:
            raise ValueError("smoke receipt must be outside the read-only checkout")
        if upstream == checkout_receipt_path or upstream in checkout_receipt_path.parents:
            raise ValueError("checkout receipt must be outside the read-only checkout")
        checkout_receipt = json.loads(checkout_receipt_path.read_text(encoding="utf-8"))
        errors = validate_receipt(checkout_receipt)
        if errors:
            raise RuntimeError(f"invalid checkout receipt: {errors}")
        if checkout_receipt["model"] != args.model:
            raise RuntimeError("checkout receipt model mismatch")
        if checkout_receipt["status"] not in {"PASS", "PASS_WITH_WARNING"}:
            raise RuntimeError("checkout receipt is not successful")
        if Path(checkout_receipt["checkout"]).resolve() != upstream:
            raise RuntimeError("checkout receipt path mismatch")
        receipt.update(
            {
                "commit": checkout_receipt["commit"],
                "license_status": checkout_receipt["license_status"],
                "provenance_status": checkout_receipt["provenance_status"],
            }
        )

        stage = "validate_environment_receipt"
        environment_bytes = environment_receipt_path.read_bytes()
        environment_receipt = json.loads(environment_bytes)
        errors = validate_receipt(environment_receipt)
        if errors:
            raise RuntimeError(f"invalid environment receipt: {errors}")
        if environment_receipt.get("receipt_type") != "environment":
            raise RuntimeError("environment receipt has wrong type")
        if environment_receipt.get("model") != args.model:
            raise RuntimeError("environment receipt model mismatch")
        if environment_receipt.get("status") not in {"PASS", "PASS_WITH_WARNING"}:
            raise RuntimeError("environment receipt is not successful")
        if environment_receipt.get("environment_mode") != "compat-smoke":
            raise RuntimeError("smoke requires a compat-smoke environment receipt")
        expected_python = os.path.abspath(
            str(Path(environment_receipt["environment_path"]) / "bin" / "python")
        )
        if os.path.abspath(sys.executable) != expected_python:
            raise RuntimeError("running interpreter does not match environment receipt")
        resolved_lock = _approved(Path(environment_receipt["resolved_lock_path"]))
        resolved_lock_sha = hashlib.sha256(resolved_lock.read_bytes()).hexdigest()
        if resolved_lock_sha != environment_receipt["resolved_lock_sha256"]:
            raise RuntimeError("resolved environment lock hash mismatch")
        receipt.update(
            {
                "environment_receipt_sha256": hashlib.sha256(environment_bytes).hexdigest(),
                "resolved_lock_sha256": resolved_lock_sha,
                "sys_executable": os.path.abspath(sys.executable),
            }
        )

        stage = "capture_integrity_before"
        commit_before, tree_before = _git_state(upstream)
        status_before = git_status_porcelain(upstream)
        worktree_before = worktree_content_sha256(upstream)
        if commit_before != checkout_receipt["commit"] or status_before:
            raise RuntimeError("checkout commit/status no longer matches clean receipt")
        receipt.update(
            {
                "source_tree_hash_before": tree_before,
                "worktree_content_sha256_before": worktree_before,
                "git_status_before": list(status_before),
            }
        )

        stage = "verify_physical_gpu_mapping"
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        inventory = _gpu_inventory()
        visible_physical = _visible_indices(cuda_visible, inventory)
        if args.physical_gpu not in visible_physical:
            raise RuntimeError("requested physical GPU is not visible")
        logical_device = visible_physical.index(args.physical_gpu)
        selected = next(
            row for row in inventory if row["physical_index"] == args.physical_gpu
        )

        stage = "import_torch"
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch reports CUDA unavailable")
        if logical_device >= torch.cuda.device_count():
            raise RuntimeError("logical CUDA device is outside torch device_count")
        torch.cuda.set_device(logical_device)
        torch.manual_seed(20260719)
        torch.cuda.manual_seed_all(20260719)
        device = torch.device(f"cuda:{logical_device}")
        receipt.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(logical_device),
                "cuda_visible_devices": cuda_visible,
                "torch_current_device": torch.cuda.current_device(),
            }
        )

        stage = "forward_backward"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(upstream))
        output, expected_shape = (
            _itransformer(torch, device)
            if args.model == "itransformer"
            else _fact(torch, device)
        )
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"unexpected shape: {tuple(output.shape)} != {expected_shape}")
        output.square().mean().backward()
        torch.cuda.synchronize(device)
        output_digest = hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest()

        stage = "verify_process_gpu_uuid"
        process_uuids = _process_gpu_uuids(os.getpid())
        if selected["uuid"] not in process_uuids:
            raise RuntimeError("process UUID does not prove requested physical GPU")
        receipt["physical_gpu_evidence"] = {
            "inventory": inventory,
            "visible_physical_indices": visible_physical,
            "selected": selected,
            "process_gpu_uuids": process_uuids,
        }

        stage = "capture_integrity_after"
        commit_after, tree_after = _git_state(upstream)
        status_after = git_status_porcelain(upstream)
        worktree_after = worktree_content_sha256(upstream)
        if (commit_after, tree_after, worktree_after, status_after) != (
            commit_before,
            tree_before,
            worktree_before,
            status_before,
        ):
            raise RuntimeError("upstream worktree changed during smoke")

        receipt.update(
            {
                "status": (
                    "PASS_WITH_WARNING"
                    if checkout_receipt.get("license_review_required")
                    else "PASS"
                ),
                "stage": "complete",
                "stderr_digest": EMPTY_STDERR_SHA256,
                "source_tree_hash_after": tree_after,
                "worktree_content_sha256_after": worktree_after,
                "git_status_after": list(status_after),
                "output_shape": list(output.shape),
                "output_sha256": output_digest,
                "scope": "minimal synthetic forward/backward; not benchmark reproduction",
            }
        )
        stage = "write_receipt"
        write_receipt(receipt_path, receipt)
        return 0
    except Exception as error:
        receipt["status"] = "SMOKE_FAIL"
        receipt["stage"] = stage
        receipt["stderr_digest"] = error_digest(error)
        write_receipt(receipt_path, receipt)
        print(f"smoke failed at {stage}; see receipt digest", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
