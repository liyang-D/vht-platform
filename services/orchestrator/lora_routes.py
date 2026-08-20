from fastapi import APIRouter, HTTPException

from . import lora
from .schemas import LoraAdapterActionResponse, LoraAdapterListResponse


router = APIRouter(prefix="/loras", tags=["loras"])


@router.get("", response_model=LoraAdapterListResponse)
def list_lora_adapters():
    return {"adapters": lora.list_adapters()}


@router.post(
    "/{adapter_name}/load",
    response_model=LoraAdapterActionResponse,
)
def load_lora_adapter(adapter_name: str):
    try:
        adapter, changed = lora.load_adapter(adapter_name)
        return {"adapter": adapter.to_dict(), "changed": changed}
    except lora.LoraNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except lora.LoraValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except lora.LoraRuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.delete(
    "/{adapter_name}",
    response_model=LoraAdapterActionResponse,
)
def unload_lora_adapter(adapter_name: str):
    try:
        adapter = lora.inspect_adapter(adapter_name, loaded=False)
        changed = lora.unload_adapter(adapter_name)
        return {"adapter": adapter.to_dict(), "changed": changed}
    except lora.LoraNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except lora.LoraValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except lora.LoraRuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
