# Speculative-decoding integration

Real speculative-decoding tests live here. Contract-only proposer and scheduler
state tests remain under `runtime_patch/`.

The initial coverage compares greedy output tokens between baseline generation
and speculative decoding with:

- target: `vllm-optest-models/TheBloke/Llama-2-7B-fp16`
- draft: `vllm-optest-models/yuhuili/EAGLE-llama2-chat-7B`

It also compares baseline greedy tokens with the in-checkpoint MTP path of
`qwen3.5/Qwen3.5-4B`. The MTP test uses `method="mtp"` with one draft token;
its checkpoint has the required `mtp.layers.*` weights.

The local Qwen2 EAGLE checkpoint resolves to `EagleQwen2ForCausalLM`, which is
not registered in the current vLLM 0.25 runtime. The local NVIDIA Llama 3.1
FP8 target fails baseline weight loading on this HCU stack, so the default
smoke uses the plain Llama2 fp16 checkpoint. The local
`eagle/sglang-EAGLE-llama2-chat-7B` draft advertises
`LlamaForCausalLMEagle`, which is also not registered in this runtime; the
default yuhuili draft keeps the standard Llama architecture in its config and
is compatible with vLLM's EAGLE draft-model path.

Override paths with:

- `VLLM_HCU_SPEC_TARGET_MODEL`
- `VLLM_HCU_SPEC_DRAFT_MODEL`
- `VLLM_HCU_MTP_MODEL`

Run only this coverage with:

```bash
python tools/run_patch_tests.py --suite model -k llama2_7b_eagle
```
