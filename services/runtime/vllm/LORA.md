# LoRA adapters

## Directory contract

Copy a deployable Hugging Face PEFT export into one direct child directory.
Use the training project's `artifacts/adapter` export, not a Trainer
`checkpoint-*` directory.

```text
loras/
└── <adapter-name>/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── training_summary.json
```

`training_summary.json` is strongly recommended because it records the exact
base model revision. The platform rejects adapters for a different base model
or, when both revisions are available, a different base revision.

Adapter names may contain letters, numbers, `.`, `_`, and `-`. API clients
cannot supply filesystem paths; adapters must be direct children of `loras/`.

## Internal orchestrator API

- `GET /loras` lists local adapters, validation results, and load state.
- `POST /loras/{adapter-name}/load` validates and loads an adapter into vLLM.
- `DELETE /loras/{adapter-name}` unloads it.

Set `task_config.lora_adapter` when creating a session to use that adapter for
every LLM call in the session. Omit it to use the base model.

Runtime updates are enabled only on the internal vLLM service. Production
Compose does not publish that service's port; development binds it to loopback.
