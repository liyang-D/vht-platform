import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


LORA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LORA_ROOT = Path(os.getenv("LLM_LORA_ROOT", "/models/loras"))
LLM_RUNTIME_URL = os.getenv("LLM_RUNTIME_URL", "http://llm:8000").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_MODEL_REVISION = os.getenv("LLM_MODEL_REVISION", "").strip() or None
LLM_MAX_LORA_RANK = int(os.getenv("LLM_MAX_LORA_RANK", "16"))
LORA_RUNTIME_TIMEOUT_SECONDS = float(
    os.getenv("LORA_RUNTIME_TIMEOUT_SECONDS", "30")
)
LORA_OPERATION_LOCK = threading.Lock()


class LoraError(Exception):
    pass


class LoraNotFoundError(LoraError):
    pass


class LoraValidationError(LoraError):
    pass


class LoraRuntimeError(LoraError):
    pass


@dataclass(frozen=True)
class LoraAdapter:
    name: str
    path: str
    base_model: str
    base_revision: str | None
    rank: int
    peft_type: str
    weights_file: str
    loaded: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise LoraValidationError(f"Cannot read {path.name}: {e}") from e
    except json.JSONDecodeError as e:
        raise LoraValidationError(f"Invalid JSON in {path.name}: {e}") from e

    if not isinstance(payload, dict):
        raise LoraValidationError(f"{path.name} must contain a JSON object.")

    return payload


def _adapter_path(name: str) -> Path:
    if not LORA_NAME_PATTERN.fullmatch(name):
        raise LoraValidationError(
            "Adapter name must use only letters, numbers, '.', '_' or '-'."
        )

    root = LORA_ROOT.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.parent != root:
        raise LoraValidationError("Adapter path must be a direct child of the LoRA root.")

    if not path.is_dir():
        raise LoraNotFoundError(f"LoRA adapter '{name}' was not found.")

    return path


def inspect_adapter(name: str, loaded: bool | None = None) -> LoraAdapter:
    path = _adapter_path(name)
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise LoraValidationError("adapter_config.json is required.")

    config = _read_json(config_path)
    peft_type = str(config.get("peft_type") or "").upper()
    if peft_type != "LORA":
        raise LoraValidationError("Only PEFT LoRA adapters are supported.")

    try:
        rank = int(config["r"])
    except (KeyError, TypeError, ValueError) as e:
        raise LoraValidationError("adapter_config.json must contain a valid rank 'r'.") from e
    if rank <= 0:
        raise LoraValidationError("LoRA rank must be positive.")
    if rank > LLM_MAX_LORA_RANK:
        raise LoraValidationError(
            f"LoRA rank {rank} exceeds runtime maximum {LLM_MAX_LORA_RANK}."
        )

    weights_path = next(
        (
            candidate
            for candidate in (
                path / "adapter_model.safetensors",
                path / "adapter_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )
    if weights_path is None:
        raise LoraValidationError(
            "adapter_model.safetensors or adapter_model.bin is required."
        )

    base_model = str(config.get("base_model_name_or_path") or "").strip()
    summary_path = path / "training_summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    summary_base_model = str(summary.get("base_model") or "").strip()
    if base_model and summary_base_model and base_model != summary_base_model:
        raise LoraValidationError(
            "Base model differs between adapter_config.json and training_summary.json."
        )
    base_model = summary_base_model or base_model
    if not base_model:
        raise LoraValidationError("Adapter base model metadata is missing.")
    if base_model != LLM_MODEL:
        raise LoraValidationError(
            f"Adapter requires base model '{base_model}', but runtime uses '{LLM_MODEL}'."
        )

    base_revision = summary.get("base_revision") or config.get("revision")
    base_revision = str(base_revision).strip() if base_revision else None
    if LLM_MODEL_REVISION and base_revision and base_revision != LLM_MODEL_REVISION:
        raise LoraValidationError(
            f"Adapter requires base revision '{base_revision}', but runtime uses "
            f"'{LLM_MODEL_REVISION}'."
        )

    return LoraAdapter(
        name=name,
        path=str(path),
        base_model=base_model,
        base_revision=base_revision,
        rank=rank,
        peft_type=peft_type,
        weights_file=weights_path.name,
        loaded=loaded,
    )


def get_loaded_model_ids() -> set[str]:
    try:
        response = httpx.get(
            f"{LLM_RUNTIME_URL}/v1/models",
            timeout=LORA_RUNTIME_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as e:
        raise LoraRuntimeError(f"Cannot query the LLM runtime: {e}") from e

    models = payload.get("data", []) if isinstance(payload, dict) else []
    return {
        str(model["id"])
        for model in models
        if isinstance(model, dict) and model.get("id")
    }


def list_adapters() -> list[dict[str, Any]]:
    try:
        loaded_models: set[str] | None = get_loaded_model_ids()
    except LoraRuntimeError:
        loaded_models = None

    if not LORA_ROOT.is_dir():
        return []

    adapters: list[dict[str, Any]] = []
    for path in sorted(LORA_ROOT.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or not LORA_NAME_PATTERN.fullmatch(path.name):
            continue
        try:
            loaded = path.name in loaded_models if loaded_models is not None else None
            adapter = inspect_adapter(path.name, loaded=loaded)
            record = adapter.to_dict()
            record.update({"valid": True, "error": None})
        except LoraError as e:
            record = {
                "name": path.name,
                "path": str(path),
                "loaded": False if loaded_models is not None else None,
                "valid": False,
                "error": str(e),
            }
        adapters.append(record)

    return adapters


def _load_adapter(name: str) -> tuple[LoraAdapter, bool]:
    adapter = inspect_adapter(name)
    if name in get_loaded_model_ids():
        return LoraAdapter(**{**adapter.to_dict(), "loaded": True}), False

    try:
        response = httpx.post(
            f"{LLM_RUNTIME_URL}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": adapter.path},
            timeout=LORA_RUNTIME_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        detail = e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
        raise LoraRuntimeError(f"vLLM could not load adapter '{name}': {detail}") from e

    return LoraAdapter(**{**adapter.to_dict(), "loaded": True}), True


def _unload_adapter(name: str) -> bool:
    _adapter_path(name)
    if name not in get_loaded_model_ids():
        return False

    try:
        response = httpx.post(
            f"{LLM_RUNTIME_URL}/v1/unload_lora_adapter",
            json={"lora_name": name},
            timeout=LORA_RUNTIME_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        detail = e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
        raise LoraRuntimeError(f"vLLM could not unload adapter '{name}': {detail}") from e

    return True


def load_adapter(name: str) -> tuple[LoraAdapter, bool]:
    with LORA_OPERATION_LOCK:
        return _load_adapter(name)


def unload_adapter(name: str) -> bool:
    with LORA_OPERATION_LOCK:
        return _unload_adapter(name)


def resolve_model(adapter_name: str | None) -> str:
    if not adapter_name:
        return LLM_MODEL

    adapter, _ = load_adapter(adapter_name)
    return adapter.name
