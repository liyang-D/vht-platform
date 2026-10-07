from fastapi import APIRouter, HTTPException
from openai import APIError

from . import vision
from .schemas import VisionAnalyzeRequest, VisionAnalyzeResponse


router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/analyze", response_model=VisionAnalyzeResponse)
def analyze_images(request: VisionAnalyzeRequest):
    """Run one event-triggered VLM analysis after an app task completes."""
    try:
        return vision.analyze(request)
    except vision.VisionInputError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except APIError as e:
        raise HTTPException(status_code=502, detail="VLM runtime request failed.") from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
