# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light KV-cache layout helpers shared by runtime and tests."""

from __future__ import annotations

import torch


def split_kv_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Return key and value tensors from split or axis-zero stacked storage."""
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            raise ValueError(f"expected two split KV cache tensors, got {len(kv_cache)}")
        return kv_cache[0], kv_cache[1]
    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")
    if kv_cache.ndim >= 1 and kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)
    raise ValueError(
        "expected stacked KV cache dimension of size 2 at axis 0, "
        f"got shape {tuple(kv_cache.shape)}"
    )


__all__ = ["split_kv_cache"]
