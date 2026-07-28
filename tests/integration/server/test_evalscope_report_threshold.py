# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable tests for EvalScope report pass criteria."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.server.evalscope_server import _assert_pass_criteria


def _config(score: float) -> dict:
    return {
        "model": "/models/Qwen3-8B",
        "evalscope": {
            "pass_criteria": {
                "dataset": "gsm8k",
                "metric": "mean_acc",
                "display_name": "Pass@1",
                "minimum_score": score,
            }
        },
    }


def _write_report(work_dir: Path, score: float) -> Path:
    report_path = work_dir / "reports/Qwen3-8B/gsm8k.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "name": "Qwen3-8B@gsm8k",
                "dataset_name": "gsm8k",
                "model_name": "Qwen3-8B",
                "metrics": [
                    {
                        "name": "mean_acc",
                        "score": score,
                        "num": 100,
                        "categories": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_pass_at_one_accepts_score_at_threshold(tmp_path: Path) -> None:
    _write_report(tmp_path, 0.95)

    _assert_pass_criteria(
        _config(0.95),
        model_env="VLLM_HCU_TEST_UNUSED_MODEL",
        work_dir=tmp_path,
        eval_log_path=tmp_path / "logs/evalscope.log",
    )

    assert "Pass@1=0.9500" in (tmp_path / "logs/evalscope.log").read_text()


def test_pass_at_one_rejects_score_below_threshold(tmp_path: Path) -> None:
    _write_report(tmp_path, 0.9499)

    with pytest.raises(AssertionError, match=r"Pass@1=0\.9499"):
        _assert_pass_criteria(
            _config(0.95),
            model_env="VLLM_HCU_TEST_UNUSED_MODEL",
            work_dir=tmp_path,
            eval_log_path=tmp_path / "logs/evalscope.log",
        )


def test_pass_at_one_requires_report_metric(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="missing EvalScope report metric"):
        _assert_pass_criteria(
            _config(0.95),
            model_env="VLLM_HCU_TEST_UNUSED_MODEL",
            work_dir=tmp_path,
            eval_log_path=tmp_path / "logs/evalscope.log",
        )
