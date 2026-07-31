# Server integration

This layer contains real OpenAI-compatible server tests.

Current coverage:

- `test_qwen3_protocol_features.py` starts one `Qwen3-4B` server and verifies
  completions, chat/Jinja rendering, reasoning and logprobs, JSON mode,
  JSON Schema, EBNF grammar, named function calls, Anthropic Messages,
  request-length rejection, truncation, and `ignore_eos`.
- `test_qwen25_server_smoke.py` verifies that `Qwen2.5-1.5B-Instruct`
  starts as an OpenAI-compatible chat service.
- `test_evalscope_qwen3_8b_gsm8k.py` starts a single-HCU, eager-mode
  `Qwen3-8B` server and runs a 10-sample EvalScope GSM8K smoke test.
- `test_evalscope_qwen35_9b_gsm8k.py` starts a single-HCU, eager-mode
  `Qwen3.5-9B` server and checks EvalScope GSM8K Pass@1.
- `test_evalscope_qwen3_vl_8b_mmmu.py` starts a single-HCU, eager-mode
  `Qwen3-VL-8B-Instruct` server and checks EvalScope MMMU multimodal accuracy.
- `test_evalscope_deepseek_r1_gsm8k.py` starts `vllm serve` for
  DeepSeek-R1 Channel-FP8 W8A8 with TP=8, waits for `/health`, then runs
  EvalScope on GSM8K through the OpenAI API.

The Qwen3-8B smoke test needs one local HCU device, the checkpoint at
`/models/llm-models/qwen3/Qwen3-8B`, `vllm`, and `evalscope`. Select it with
`-k qwen3_8b_gsm8k_evalscope_server`.

The Qwen3.5-9B and Qwen3-VL-8B accuracy tests are selected with
`-k qwen35_9b_gsm8k_evalscope_server` and
`-k qwen3_vl_8b_mmmu_evalscope_server`.

Run the protocol coverage with:

```bash
python tools/run_patch_tests.py --suite model -- -k qwen3_protocol
```

Set `VLLM_HCU_PROTOCOL_MODEL` to override the default `qwen3/Qwen3-4B`
checkpoint. Server output is stored under
`/tmp/vllm-hcu-integration/logs` unless
`VLLM_HCU_INTEGRATION_LOG_DIR` is set.

The DeepSeek-R1 test needs eight local HCU devices, the local model path,
`vllm`, and `evalscope`. It is marked `hcu`, `model`, `multi_hcu`,
`hcu_count(8)`, `slow`, `nightly`, and `external_service("evalscope")`, so it
is excluded from `integration-smoke`.
