from __future__ import annotations

from a_share_research.provenance import EMPTY_STDERR_SHA256, validate_receipt


def _smoke_receipt() -> dict[str, object]:
    return {
        "schema_version": 2,
        "receipt_type": "smoke",
        "created_at_utc": "2026-07-19T00:00:00+00:00",
        "model": "itransformer",
        "status": "PASS",
        "stage": "complete",
        "stderr_digest": EMPTY_STDERR_SHA256,
        "command": "smoke",
        "provenance_status": "READY_FOR_SERVER_SMOKE",
        "commit": "a" * 40,
        "license_status": "MIT_CLEAR",
        "source_tree_hash_before": "b" * 40,
        "source_tree_hash_after": "b" * 40,
        "worktree_content_sha256_before": "c" * 64,
        "worktree_content_sha256_after": "c" * 64,
        "git_status_before": [],
        "git_status_after": [],
        "python_version": "3.11",
        "torch_version": "2.7.1+cu128",
        "cuda_version": "12.8",
        "gpu_name": "gpu",
        "cuda_visible_devices": "0",
        "torch_current_device": 0,
        "physical_gpu_requested": 0,
        "physical_gpu_evidence": {
            "selected": {
                "physical_index": 0,
                "uuid": "GPU-test",
                "pci_bus_id": "00000000:01:00.0",
                "name": "gpu",
            }
        },
        "environment_receipt_sha256": "d" * 64,
        "resolved_lock_sha256": "e" * 64,
        "sys_executable": "/approved/env/bin/python",
        "output_shape": [2, 2, 9],
    }


def test_success_receipt_requires_physical_gpu_evidence() -> None:
    receipt = _smoke_receipt()
    assert validate_receipt(receipt) == []
    del receipt["physical_gpu_evidence"]
    assert any("physical_gpu_evidence" in error for error in validate_receipt(receipt))


def test_failure_receipt_may_keep_unreached_fields_null() -> None:
    receipt = _smoke_receipt()
    receipt["status"] = "SMOKE_FAIL"
    receipt["stage"] = "import_torch"
    receipt["torch_version"] = None
    assert validate_receipt(receipt) == []
