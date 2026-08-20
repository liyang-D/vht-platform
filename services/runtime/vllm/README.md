# vLLM Runtime

This directory holds local runtime inputs for the OpenAI-compatible vLLM service.

- `loras/`: optional LoRA adapters mounted read-only into the `llm` container at `/models/loras`.
  See `LORA.md` for the adapter contract and orchestrator API.
- Hugging Face model files are stored in the Docker named volume `huggingface_cache`.
