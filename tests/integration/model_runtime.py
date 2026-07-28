# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared helpers for real-model integration tests.

This module is intentionally safe to import on CPU-only hosts. vLLM is imported
only inside the subprocess cases after pytest has checked model and HCU
availability.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources


DEFAULT_MODEL_ROOT = Path("/models/llm-models")
DEFAULT_LOG_DIR = Path("/tmp/vllm-hcu-integration/logs")
RESULT_PREFIX = "VLLM_HCU_RESULT="
UNIFIED_ATTENTION_HEAD_DIMS = {128, 192, 256, 512}


def available_hcu_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def resolve_model_path(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    if resources.model_root is not None:
        return resources.resolve_model(relative_path)
    default_path = DEFAULT_MODEL_ROOT / relative_path
    if default_path.exists():
        return default_path.resolve()
    return Path(relative_path)


def require_model_runtime(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
    label: str,
    hcu_count: int = 1,
) -> Path:
    model_path = resolve_model_path(
        resources,
        env_name=env_name,
        relative_path=relative_path,
    )
    if not model_path.exists():
        message = f"{label} model path is unavailable: {model_path}"
        if resources.strict:
            pytest.fail(message)
        pytest.skip(message)
    _require_unified_attention_compatible(resources, model_path, label)
    actual_hcu_count = available_hcu_count()
    if actual_hcu_count < hcu_count:
        pytest.skip(
            f"{label} test requires {hcu_count} HCU devices, got {actual_hcu_count}"
        )
    return model_path


def require_non_hybrid_model(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
    label: str,
    hcu_count: int = 1,
) -> Path:
    model_path = resolve_model_path(
        resources,
        env_name=env_name,
        relative_path=relative_path,
    )
    if not model_path.exists():
        message = f"{label} model path is unavailable: {model_path}"
        if resources.strict:
            pytest.fail(message)
        pytest.skip(message)
    _require_unified_attention_compatible(resources, model_path, label)
    if _is_hybrid_kv_model(model_path):
        message = (
            f"{label} model {model_path} is a hybrid KV-cache model and is "
            "incompatible with ExampleConnector, which does not support HMA"
        )
        if resources.strict:
            pytest.fail(message)
        pytest.skip(message)
    actual_hcu_count = available_hcu_count()
    if actual_hcu_count < hcu_count:
        pytest.skip(
            f"{label} test requires {hcu_count} HCU devices, got {actual_hcu_count}"
        )
    return model_path


def require_resource_path(
    resources: HcuTestResources,
    *,
    env_name: str,
    relative_path: str,
    label: str,
) -> Path:
    path = resolve_model_path(
        resources,
        env_name=env_name,
        relative_path=relative_path,
    )
    if path.exists():
        return path
    message = f"{label} path is unavailable: {path}"
    if resources.strict:
        pytest.fail(message)
    pytest.skip(message)


def require_model_architecture(
    resources: HcuTestResources,
    model_path: Path,
    *,
    label: str,
    supported_architectures: set[str],
) -> None:
    architectures = _model_architectures(model_path)
    if architectures and any(item in supported_architectures for item in architectures):
        return
    rendered = ", ".join(architectures) if architectures else "unknown"
    message = (
        f"{label} architectures [{rendered}] are unsupported for this "
        f"integration test; expected one of {sorted(supported_architectures)}"
    )
    if resources.strict:
        pytest.fail(message)
    pytest.skip(message)


def _require_unified_attention_compatible(
    resources: HcuTestResources,
    model_path: Path,
    label: str,
) -> None:
    head_dim = _model_attention_head_dim(model_path)
    if head_dim is None or head_dim in UNIFIED_ATTENTION_HEAD_DIMS:
        return
    message = (
        f"{label} attention head_dim={head_dim} is incompatible with "
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1"
    )
    if resources.strict:
        pytest.fail(message)
    pytest.skip(message)


def _model_attention_head_dim(model_path: Path) -> int | None:
    config = _read_model_config(model_path)
    if config is None:
        return None
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        head_dim = _attention_head_dim_from_config(text_config)
        if head_dim is not None:
            return head_dim
    return _attention_head_dim_from_config(config)


def _model_architectures(model_path: Path) -> list[str]:
    config = _read_model_config(model_path)
    if config is None:
        return []
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        return []
    return [str(item) for item in architectures]


def _read_model_config(model_path: Path) -> dict[str, Any] | None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        return None
    return config


def _attention_head_dim_from_config(config: dict[str, Any]) -> int | None:
    explicit_head_dim = config.get("head_dim")
    if isinstance(explicit_head_dim, int) and not isinstance(
        explicit_head_dim, bool
    ):
        return explicit_head_dim
    hidden_size = config.get("hidden_size")
    attention_heads = config.get("num_attention_heads")
    if (
        isinstance(hidden_size, int)
        and isinstance(attention_heads, int)
        and not isinstance(hidden_size, bool)
        and not isinstance(attention_heads, bool)
        and attention_heads > 0
    ):
        return hidden_size // attention_heads
    return None


def _is_hybrid_kv_model(model_path: Path) -> bool:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        return False
    return _config_has_multiple_kv_layer_types(config)


def _config_has_multiple_kv_layer_types(config: dict[str, Any]) -> bool:
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and len({str(item) for item in layer_types}) > 1:
        return True
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return _config_has_multiple_kv_layer_types(text_config)
    return False


def run_vllm_case(
    case: str,
    model_path: Path,
    *,
    timeout_s: int = 900,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("VLLM_PLUGINS", None)
    env["VLLM_HCU_USE_FLASH_ATTN_UNIFIED"] = "1"
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    log_path = _case_log_path(case, model_path)
    command = [
        sys.executable,
        "-m",
        "tests.integration.model_runtime",
        case,
        "--model",
        str(model_path),
    ]
    if extra_args:
        command.extend(extra_args)
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    _write_case_log(log_path, command, result.stdout)
    if result.returncode != 0:
        raise AssertionError(
            f"vLLM integration case {case!r} failed with rc={result.returncode}\n"
            f"command={' '.join(command)}\n"
            f"log={log_path}\n"
            f"{result.stdout}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = line.removeprefix(RESULT_PREFIX)
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise AssertionError(f"invalid result payload for {case!r}: {payload}")
            return parsed
    raise AssertionError(
        f"vLLM integration case {case!r} did not emit {RESULT_PREFIX!r}\n"
        f"log={log_path}\n"
        f"{result.stdout}"
    )


def _case_log_path(case: str, model_path: Path) -> Path:
    log_dir = Path(os.environ.get("VLLM_HCU_INTEGRATION_LOG_DIR", DEFAULT_LOG_DIR))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_path.name)
    case_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case)
    return log_dir / f"{timestamp}_{model_name}_{case_name}.log"


def _write_case_log(log_path: Path, command: list[str], output: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "command: " + " ".join(command) + "\n"
    header += "environment: VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1\n"
    log_path.write_text(header + output, encoding="utf-8")


def _llm_kwargs(
    model_path: Path,
    *,
    enforce_eager: bool,
    **overrides: Any,
) -> dict[str, Any]:
    kwargs = {
        "model": str(model_path),
        "trust_remote_code": True,
        "enforce_eager": enforce_eager,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 4,
        "gpu_memory_utilization": 0.35,
        "seed": 0,
    }
    kwargs.update(overrides)
    return kwargs


def _shutdown_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=30)
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _single_completion(record: Any) -> dict[str, Any]:
    output = record.outputs[0]
    token_ids = list(output.token_ids)
    cumulative_logprob = output.cumulative_logprob
    if cumulative_logprob is not None and not math.isfinite(cumulative_logprob):
        raise AssertionError(f"non-finite cumulative logprob: {cumulative_logprob}")
    lora_request = getattr(record, "lora_request", None)
    return {
        "prompt_token_count": len(record.prompt_token_ids or []),
        "token_ids": token_ids,
        "text": output.text,
        "finish_reason": output.finish_reason,
        "cumulative_logprob": cumulative_logprob,
        "lora_name": getattr(lora_request, "lora_name", None),
        "lora_int_id": getattr(lora_request, "lora_int_id", None),
    }


def _generate_with_llm(
    llm: Any,
    *,
    prompts: list[str] | None = None,
    lora_request: Any = None,
    max_tokens: int = 8,
) -> list[dict[str, Any]]:
    from vllm.sampling_params import SamplingParams

    if prompts is None:
        prompts = [
            "The capital of France is",
            "Answer with one number: 2 + 2 =",
        ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=1,
        seed=0,
    )
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    return [_single_completion(record) for record in outputs]


def _generate(
    model_path: Path,
    *,
    enforce_eager: bool,
    **llm_overrides: Any,
) -> list[dict[str, Any]]:
    from vllm import LLM

    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=enforce_eager,
            **llm_overrides,
        )
    )
    try:
        return _generate_with_llm(llm)
    finally:
        _shutdown_llm(llm)


def _case_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM

    llm = LLM(**_llm_kwargs(model_path, enforce_eager=True))
    try:
        first = _generate_with_llm(llm)
        second = _generate_with_llm(llm)
    finally:
        _shutdown_llm(llm)
    return {
        "first": first,
        "second": second,
    }


def _case_graph_parity(model_path: Path) -> dict[str, Any]:
    eager = _generate(model_path, enforce_eager=True)
    graph = _generate(model_path, enforce_eager=False)
    return {
        "eager": eager,
        "graph": graph,
    }


def _case_lora_switching(
    model_path: Path,
    *,
    lora_a: Path,
    lora_b: Path,
) -> dict[str, Any]:
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    prompt = ["用一句话写一个武侠小说开头："]
    adapter_a = LoRARequest("adapter-a", 1, str(lora_a))
    adapter_b = LoRARequest("adapter-b", 2, str(lora_b))
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            enable_lora=True,
            max_loras=2,
            max_cpu_loras=2,
            max_lora_rank=16,
        )
    )
    try:
        base = _generate_with_llm(llm, prompts=prompt)
        first_a = _generate_with_llm(llm, prompts=prompt, lora_request=adapter_a)
        first_b = _generate_with_llm(llm, prompts=prompt, lora_request=adapter_b)
        second_a = _generate_with_llm(llm, prompts=prompt, lora_request=adapter_a)
    finally:
        _shutdown_llm(llm)
    return {
        "base": base,
        "adapter_a": first_a,
        "adapter_b": first_b,
        "adapter_a_again": second_a,
    }


def _case_spec_decode_parity(
    model_path: Path,
    *,
    draft_model: Path,
) -> dict[str, Any]:
    baseline = _generate(model_path, enforce_eager=True)
    speculative = _generate(
        model_path,
        enforce_eager=True,
        spec_model=str(draft_model),
        spec_tokens=2,
    )
    return {
        "baseline": baseline,
        "speculative": speculative,
    }


def _case_kv_transfer_smoke(model_path: Path) -> dict[str, Any]:
    from vllm import LLM
    from vllm.config.kv_transfer import KVTransferConfig

    storage_path = Path(
        os.environ.get(
            "VLLM_HCU_KV_TRANSFER_STORAGE",
            "/tmp/vllm-hcu-integration/kv-transfer",
        )
    )
    storage_path.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        **_llm_kwargs(
            model_path,
            enforce_eager=True,
            disable_hybrid_kv_cache_manager=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "shared_storage_path": str(storage_path),
                },
            ),
        )
    )
    try:
        output = _generate_with_llm(
            llm,
            prompts=["KV transfer smoke test prompt:"],
            max_tokens=4,
        )
    finally:
        _shutdown_llm(llm)
    return {
        "connector": "ExampleConnector",
        "storage_path": str(storage_path),
        "output": output,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=(
            "smoke",
            "graph-parity",
            "lora-switching",
            "spec-decode-parity",
            "kv-transfer-smoke",
        ),
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--lora-a", type=Path)
    parser.add_argument("--lora-b", type=Path)
    parser.add_argument("--draft-model", type=Path)
    args = parser.parse_args(argv)

    if args.case == "smoke":
        payload = _case_smoke(args.model)
    elif args.case == "graph-parity":
        payload = _case_graph_parity(args.model)
    elif args.case == "lora-switching":
        if args.lora_a is None or args.lora_b is None:
            raise SystemExit("lora-switching requires --lora-a and --lora-b")
        payload = _case_lora_switching(
            args.model,
            lora_a=args.lora_a,
            lora_b=args.lora_b,
        )
    elif args.case == "spec-decode-parity":
        if args.draft_model is None:
            raise SystemExit("spec-decode-parity requires --draft-model")
        payload = _case_spec_decode_parity(args.model, draft_model=args.draft_model)
    else:
        payload = _case_kv_transfer_smoke(args.model)
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
