# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Behavior contracts for HCU model-runner speculative-token dispatch."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_hcu"
    / "v1"
    / "hcu_model_runner.py"
)


class _NgramProposer:
    pass


class _NgramProposerGPU:
    pass


class _SuffixDecodingProposer:
    pass


class _MedusaProposer:
    pass


class _ExtractHiddenStatesProposer:
    pass


class _EagleProposer:
    pass


class _DFlashProposer:
    pass


class _DraftModelProposer:
    pass


class _Gemma4Proposer:
    pass


def _load_propose_draft_token_ids():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "propose_draft_token_ids"
    )
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)
    namespace = {
        "torch": torch,
        "NgramProposerGPU": _NgramProposerGPU,
        "SuffixDecodingProposer": _SuffixDecodingProposer,
        "MedusaProposer": _MedusaProposer,
        "ExtractHiddenStatesProposer": _ExtractHiddenStatesProposer,
        "EagleProposer": _EagleProposer,
        "DFlashProposer": _DFlashProposer,
        "DraftModelProposer": _DraftModelProposer,
        "Gemma4Proposer": _Gemma4Proposer,
        "copy_num_valid_draft_tokens": lambda *args: None,
    }
    exec(compile(extracted, str(SOURCE_PATH), "exec"), namespace)
    return namespace["propose_draft_token_ids"]


def _spec_config(branch: str) -> SimpleNamespace:
    return SimpleNamespace(
        method={
            "ngram_cpu": "ngram",
            "suffix": "suffix",
            "medusa": "medusa",
        }.get(branch, "other"),
        use_ngram_gpu=lambda: branch == "ngram_gpu",
        uses_extract_hidden_states=lambda: branch == "extract_hidden_states",
        use_eagle=lambda: False,
        use_dflash=lambda: False,
        uses_draft_model=lambda: branch == "draft_model",
        disable_padded_drafter_batch=True,
    )


def _runner(branch: str, drafter: object) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=_spec_config(branch),
        drafter=drafter,
        input_batch=SimpleNamespace(
            num_tokens_no_spec=[1],
            token_ids_cpu=[[1]],
            num_reqs=1,
        ),
        _copy_valid_sampled_token_count=lambda *args: None,
    )


@pytest.mark.parametrize(
    "branch",
    [
        "ngram_cpu",
        "ngram_gpu",
        "suffix",
        "medusa",
        "extract_hidden_states",
        "draft_model",
    ],
)
def test_runner_forwards_scheduler_dynamic_speculative_token_count(
    branch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    propose_draft_token_ids = _load_propose_draft_token_ids()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    ngram_module = ModuleType("vllm.v1.spec_decode.ngram_proposer")
    ngram_module.NgramProposer = _NgramProposer
    monkeypatch.setitem(sys.modules, ngram_module.__name__, ngram_module)

    if branch == "ngram_cpu":
        class Drafter(_NgramProposer):
            def propose(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "drafts"

        drafter = Drafter()
        sampled_token_ids = [[1]]
        aux_hidden_states = None
    elif branch == "ngram_gpu":
        class Drafter(_NgramProposerGPU):
            def update_token_ids_ngram(self, *args):
                return torch.tensor([1]), torch.tensor([1]), torch.tensor([1])

            def propose(self, *args, **kwargs):
                calls.append((args, kwargs))
                return torch.tensor([[11]]), torch.tensor([1])

        drafter = Drafter()
        sampled_token_ids = torch.tensor([[1]])
        aux_hidden_states = None
    elif branch == "suffix":
        class Drafter(_SuffixDecodingProposer):
            def propose(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "drafts"

        drafter = Drafter()
        sampled_token_ids = [[1]]
        aux_hidden_states = None
    elif branch == "medusa":
        class Drafter(_MedusaProposer):
            def propose(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "drafts"

        drafter = Drafter()
        sampled_token_ids = [[1], [2]]
        aux_hidden_states = None
    elif branch == "extract_hidden_states":
        class Drafter(_ExtractHiddenStatesProposer):
            def propose(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "drafts"

            def prepare_next_token_ids_padded(self, *args):
                return torch.tensor([1]), torch.tensor([1])

        drafter = Drafter()
        sampled_token_ids = torch.tensor([[1]])
        aux_hidden_states = [torch.ones(2, 2)]
    else:
        class Drafter(_DraftModelProposer):
            supports_mm_inputs = False

            def prepare_next_token_ids_cpu(self, *args):
                return torch.tensor([1])

            def propose(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "drafts"

        drafter = Drafter()
        sampled_token_ids = [[1]]
        aux_hidden_states = None

    runner = _runner(branch, drafter)
    if branch == "ngram_gpu":
        runner.token_ids_gpu_tensor = torch.tensor([[1]])
        runner.num_tokens_no_spec_gpu = torch.tensor([1])
        runner.discard_request_mask = SimpleNamespace(gpu=torch.tensor([False]))
        runner._num_valid_draft_tokens_cpu = object()
        runner._num_valid_draft_tokens_copy_stream = object()
        runner._num_valid_draft_tokens_event = object()
    elif branch == "extract_hidden_states":
        runner.use_aux_hidden_state_outputs = True
        runner.requests = {}
        runner.discard_request_mask = SimpleNamespace(gpu=torch.tensor([False]))
    elif branch == "draft_model":
        runner.requests = {}
        runner.get_model = lambda: SimpleNamespace(
            get_mtp_target_hidden_states=lambda: None
        )
        runner.input_ids = SimpleNamespace(gpu=torch.tensor([1, 2]))
        runner._get_positions = lambda size: torch.arange(size)
        runner.use_aux_hidden_state_outputs = False
        runner.supports_mm_inputs = False

    dynamic_k = 7
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_spec_tokens_to_schedule=dynamic_k,
        num_scheduled_tokens={},
    )
    propose_draft_token_ids(
        runner,
        scheduler_output,
        sampled_token_ids,
        object(),
        torch.ones(2, 3),
        torch.ones(2, 3),
        aux_hidden_states,
        None,
        object(),
        None,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    if branch in {"ngram_cpu", "ngram_gpu", "suffix"}:
        assert args[0] == dynamic_k
    else:
        assert kwargs["num_speculative_tokens"] == dynamic_k
