#!/usr/bin/env python3
"""Server-only GPU smoke for the projected iTransformer/FACT/TimeXer runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPECTED_COMMITS = {
    "itransformer": "c2426e68ca13f74aaec08045c5c724d8ad328124",
    "fact": "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4",
    "timexer": "76011909357972bd55a27adba2e1be994d81b327",
    "timepro": "70a20e5a257b30eb026ee4316293cf4feeb92a1f",
    "s4m": "a718823addd3606e763dfc261174e0135b2535f4",
}


def _git(upstream_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(upstream_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _author_model(model_name: str, torch: Any, assets: int, horizon: int) -> Any:
    if model_name == "itransformer":
        module = importlib.import_module("model.iTransformer")
        config = SimpleNamespace(
            seq_len=8,
            pred_len=horizon,
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
            learning_rate=1e-3,
            lradj="type1",
        )
    elif model_name == "fact":
        module = importlib.import_module("models.FACT")
        config = SimpleNamespace(
            task_name="long_term_forecast",
            pred_len=horizon,
            seq_len=8,
            enc_in=assets,
            use_norm=1,
            freq="n",
            d_model=8,
            dilation=[1],
            num_kernels=2,
            d_ff=16,
            core=0.5,
            dropout=0.0,
            learning_rate=1e-3,
            lradj="type1",
        )
    elif model_name == "timexer":
        module = importlib.import_module("models.TimeXer")
        config = SimpleNamespace(
            task_name="long_term_forecast",
            features="M",
            seq_len=8,
            pred_len=horizon,
            enc_in=assets,
            dec_in=assets,
            c_out=assets,
            use_norm=1,
            patch_len=4,
            d_model=16,
            n_heads=4,
            e_layers=2,
            d_ff=32,
            factor=1,
            dropout=0.0,
            embed="timeF",
            freq="h",
            activation="gelu",
            learning_rate=1e-3,
            lradj="type1",
        )
    elif model_name == "timepro":
        module = importlib.import_module("model.TimePro")
        config = SimpleNamespace(
            seq_len=8,
            pred_len=horizon,
            use_norm=1,
            patch_len=4,
            stride=4,
            d_model=64,
            dropout=0.0,
            enc_in=assets,
            e_layers=1,
            learning_rate=1e-3,
            lradj="type1",
        )
    elif model_name == "s4m":
        module = importlib.import_module("model.S4M")
        config = SimpleNamespace(
            seq_len=50,
            pred_len=horizon,
            enc_in=assets,
            d_var=assets,
            dec_in=assets,
            c_out=assets,
            use_norm=1,
            freq="h",
            d_model=64,
            d_ff=64,
            e_layers=2,
            n_heads=8,
            factor=1,
            dropout=0.0,
            output_attention=False,
            lradj="type1",
            mask=True,
            classification=False,
            plot=0,
            num_class=10,
            short_len=50,
            n=10,
            W=6,
            en_conv_hidden_size=256,
            en_rnn_hidden_sizes=[20, 32],
            output_keep_prob=0.9,
            input_keep_prob=0.9,
            K=10,
            topK=10,
            topM=100,
            thres1=0.6,
            thres2=0.3,
            M=30,
            momentum=0.99,
            memory_size=256,
            per_mem_size=50,
            is_training=1,
            learning_rate=1e-3,
        )
    return module.Model(config), config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(EXPECTED_COMMITS), required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "projected_deep_runtime_gpu_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "status": "FAIL",
        "commit": None,
        "physical_gpu": args.physical_gpu,
        "logical_gpu": None,
        "torch_version": None,
        "input_shape": None,
        "projected_shape": None,
        "output_shape": None,
        "best_epoch": None,
        "best_validation_mse": None,
        "selected_target_count": None,
        "native_normalization": (
            "decay_imputation_v1"
            if args.model == "s4m"
            else "out_of_place_math_equivalent_v1"
        ),
        "source_clean_before": None,
        "source_clean_after": None,
        "error_type": None,
        "error_digest": None,
    }
    try:
        if not args.upstream_root.is_absolute():
            raise RuntimeError("upstream root must be absolute")
        if not args.checkpoint.is_absolute() or not args.receipt.is_absolute():
            raise RuntimeError("checkpoint and receipt paths must be absolute")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(args.physical_gpu):
            raise RuntimeError("CUDA_VISIBLE_DEVICES must expose only the requested physical GPU")
        commit = _git(args.upstream_root, "rev-parse", "HEAD")
        if commit != EXPECTED_COMMITS[args.model]:
            raise RuntimeError("upstream checkout does not match the frozen commit")
        payload["commit"] = commit
        before = _git(args.upstream_root, "status", "--porcelain", "--untracked-files=all")
        payload["source_clean_before"] = not before
        if before:
            raise RuntimeError("upstream checkout is not clean before smoke")

        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.upstream_root))
        if args.model == "s4m":
            sys.path.insert(0, str(args.upstream_root / "model"))
        import torch

        from a_share_research.adapters.common.torch_runtime import (
            DeepForecastBatch,
            ProjectedForecastModule,
            S4MForecastModule,
            SharedPerAssetProjector,
            fit_protocol_safe,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("smoke requires exactly one visible CUDA device")
        torch.cuda.set_device(0)
        torch.manual_seed(20260719)
        torch.cuda.manual_seed_all(20260719)
        device = torch.device("cuda:0")
        if args.model == "s4m":
            assets, channels, lookback, horizon, n_samples = 9, 5, 50, 2, 32
        else:
            assets, channels, lookback, horizon, n_samples = 9, 5, 8, 2, 2
        backbone, config = _author_model(args.model, torch, assets, horizon)
        if args.model == "s4m":
            model = S4MForecastModule(
                SharedPerAssetProjector(channels), backbone, pred_len=horizon
            ).to(device)
        else:
            model = ProjectedForecastModule(
                SharedPerAssetProjector(channels), backbone
            ).to(device)
        values = torch.randn(n_samples, lookback, assets, channels, device=device)
        observed = torch.ones(n_samples, lookback, assets, dtype=torch.bool, device=device)
        observed[:, :, -1] = False
        target = torch.zeros(n_samples, horizon, assets, device=device)
        target_observed = torch.ones_like(target, dtype=torch.bool)
        label_available = torch.ones_like(target, dtype=torch.bool)
        label_available[:, :, -1] = False
        batch = DeepForecastBatch(
            x_enc=values,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None,
            observed_mask=observed,
            target=target,
            target_observed=target_observed,
            label_available=label_available,
        )
        tools = importlib.import_module("utils.tools")

        def adjust(optimizer: Any, epoch: int) -> None:
            tools.adjust_learning_rate(optimizer, epoch, config)

        if args.model == "s4m":
            def _s4m_sgd(parameters, lr):
                return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=1e-5)
            optimizer_factory = _s4m_sgd
        else:
            optimizer_factory = None
        summary = fit_protocol_safe(
            model,
            (batch,),
            (batch,),
            learning_rate=config.learning_rate,
            maximum_epochs=2,
            patience=2,
            adjust_learning_rate=adjust,
            checkpoint_path=args.checkpoint,
            optimizer_factory=optimizer_factory,
        )
        model.eval()
        with torch.no_grad():
            projected = model.projector(values, observed)
            output = model(values, None, None, None, observed)
            if isinstance(output, tuple):
                output = output[0]
        torch.cuda.synchronize(device)
        after = _git(args.upstream_root, "status", "--porcelain", "--untracked-files=all")
        payload.update(
            {
                "status": "PASS" if not after else "FAIL",
                "logical_gpu": torch.cuda.current_device(),
                "torch_version": torch.__version__,
                "input_shape": list(values.shape),
                "projected_shape": list(projected.shape),
                "output_shape": list(output.shape),
                "best_epoch": summary.best_epoch,
                "best_validation_mse": summary.best_validation_mse,
                "selected_target_count": summary.selected_target_count,
                "source_clean_after": not after,
            }
        )
        if after:
            raise RuntimeError("upstream checkout changed during smoke")
    except Exception as error:  # receipt must survive typed upstream/runtime failures
        message = f"{type(error).__name__}: {error}"
        payload["status"] = "FAIL"
        payload["error_type"] = type(error).__name__
        payload["error_digest"] = hashlib.sha256(message.encode("utf-8")).hexdigest()
    _write_json(args.receipt, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
