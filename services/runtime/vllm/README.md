# vLLM Runtime

This directory holds local runtime inputs for the OpenAI-compatible vLLM services.

- `llm` loads the official Qwen3-30B-A3B FP8 checkpoint but serves the stable
  logical name `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- `vlm` loads the official Qwen3-VL-8B FP8 checkpoint but serves the stable
  logical name `Qwen/Qwen3-VL-8B-Instruct`.
- Checkpoint revisions are pinned in the environment files. The two runtimes
  share the Hugging Face cache but have independent memory budgets.

- `loras/`: optional LoRA adapters mounted read-only into the `llm` container at `/models/loras`.
  See `LORA.md` for the adapter contract and orchestrator API.
- Hugging Face model files are stored in the Docker named volume `huggingface_cache`.
