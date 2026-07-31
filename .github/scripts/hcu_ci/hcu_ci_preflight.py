# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fail-closed HCU CI resource and dependency preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence


class PreflightError(RuntimeError):
    """Raised when the selected runner cannot execute its assigned job."""


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_requirement(item: dict[str, Any], model_root: Path | None) -> Path:
    env_name = item.get("env")
    relative = item.get("relative")
    if not isinstance(env_name, str) or not env_name:
        raise PreflightError(f"invalid requirement env: {item!r}")
    if not isinstance(relative, str) or not relative:
        raise PreflightError(f"invalid requirement relative path: {item!r}")
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    if model_root is None:
        raise PreflightError(
            f"{env_name} is unset and VLLM_HCU_TEST_MODEL_ROOT is unavailable"
        )
    return (model_root / relative).resolve()


def _check_requirements(
    requirements: list[dict[str, Any]],
    model_root: Path | None,
) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for item in requirements:
        path = _resolve_requirement(item, model_root)
        if not path.exists():
            raise PreflightError(f"required resource is unavailable: {path}")
        kind = item.get("kind")
        if kind == "model":
            if not path.is_dir() or not (
                (path / "config.json").is_file()
                or (path / "params.json").is_file()
            ):
                raise PreflightError(
                    f"model resource is not loadable: {path}; expected "
                    "config.json or params.json"
                )
        elif kind != "path":
            raise PreflightError(f"unsupported requirement kind: {kind!r}")
        resolved.append(
            {
                "env": str(item["env"]),
                "kind": str(kind),
                "path": str(path),
            }
        )
    return resolved


def run_preflight(
    *,
    expected_arch: str,
    required_cards: int,
    requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import torch
    except Exception:
        raise PreflightError(
            "HCU runtime dependency initialization failed."
        ) from None

    try:
        if not torch.cuda.is_available():
            raise PreflightError("torch reports no live HCU/ROCm device")
        actual_cards = int(torch.cuda.device_count())
        if actual_cards < required_cards:
            raise PreflightError(
                f"requires {required_cards} visible HCU devices, "
                f"got {actual_cards}"
            )

        device_arches: list[str] = []
        for index in range(required_cards):
            properties = torch.cuda.get_device_properties(index)
            raw_arch = getattr(properties, "gcnArchName", None)
            if not isinstance(raw_arch, str):
                raise PreflightError(
                    f"device {index} does not expose gcnArchName"
                )
            device_arches.append(raw_arch.split(":", 1)[0])
        wrong = [
            f"{index}:{arch}"
            for index, arch in enumerate(device_arches)
            if arch != expected_arch
        ]
        if wrong:
            raise PreflightError(
                f"requires {expected_arch}, incompatible devices: "
                f"{', '.join(wrong)}"
            )
    except PreflightError:
        raise
    except Exception:
        raise PreflightError("HCU device inspection failed.") from None

    model_root_text = os.environ.get("VLLM_HCU_TEST_MODEL_ROOT")
    model_root = (
        Path(model_root_text).expanduser().resolve()
        if model_root_text
        else None
    )
    resolved_requirements = _check_requirements(requirements, model_root)
    versions = {
        name: _distribution_version(name)
        for name in ("torch", "vllm", "vllm-hcu", "aiter", "evalscope")
    }
    for mandatory in ("torch", "vllm"):
        if versions[mandatory] is None:
            raise PreflightError(f"required distribution is missing: {mandatory}")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "expected_arch": expected_arch,
        "device_arches": device_arches,
        "required_cards": required_cards,
        "visible_cards": actual_cards,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
        "resources": resolved_requirements,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("gfx936", "gfx938"))
    parser.add_argument("--cards", required=True, type=int)
    parser.add_argument("--requirements-json", default="[]")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raw_requirements = json.loads(args.requirements_json)
        if not isinstance(raw_requirements, list) or not all(
            isinstance(item, dict) for item in raw_requirements
        ):
            raise PreflightError("requirements JSON must be a list of mappings")
        report = run_preflight(
            expected_arch=args.arch,
            required_cards=args.cards,
            requirements=raw_requirements,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, PreflightError) as exc:
        print(f"HCU CI preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
