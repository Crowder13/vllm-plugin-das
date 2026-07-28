# Model integration

Real checkpoint loading and normalized generation comparisons.

The first smoke test uses `qwen3.5/Qwen3.5-9B` because its attention head
dimension is compatible with `VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1` while still
fitting a single-HCU sanity pass. Resolve it with one of:

- `--model-root /models/llm-models`
- `VLLM_HCU_TEST_MODEL_ROOT=/models/llm-models`
- `VLLM_HCU_QWEN35_9B_MODEL=/absolute/path/to/Qwen3.5-9B`

On the shared HCU hosts, the helper also accepts `/models/llm-models` as a
last-resort local default so ad-hoc runs work without extra flags.
