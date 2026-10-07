import base64
import binascii
import io
import math
import os
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from . import clients
from .schemas import VisionAnalyzeRequest


VLM_MAX_IMAGES = int(os.getenv("VLM_MAX_IMAGES", "9"))
VLM_MAX_IMAGE_BYTES = int(os.getenv("VLM_MAX_IMAGE_BYTES", "8388608"))
VLM_MAX_TOTAL_IMAGE_BYTES = int(
    os.getenv("VLM_MAX_TOTAL_IMAGE_BYTES", "33554432")
)
VLM_MAX_SOURCE_PIXELS = int(os.getenv("VLM_MAX_SOURCE_PIXELS", "40000000"))
VLM_MAX_PIXELS_PER_IMAGE = int(
    os.getenv("VLM_MAX_PIXELS_PER_IMAGE", "589824")
)
VLM_MAX_TOTAL_PIXELS = int(os.getenv("VLM_MAX_TOTAL_PIXELS", "5308416"))
VLM_MAX_OUTPUT_TOKENS = int(os.getenv("VLM_MAX_OUTPUT_TOKENS", "512"))

ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class VisionInputError(ValueError):
    pass


@dataclass
class PreparedImage:
    image: Image.Image
    mime_type: str
    source_width: int
    source_height: int


def _resize_to_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image

    scale = math.sqrt(max_pixels / pixels)
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _decode_image(data_base64: str, declared_mime_type: str) -> PreparedImage:
    try:
        image_bytes = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise VisionInputError("Image data is not valid base64.") from e

    if not image_bytes:
        raise VisionInputError("Image data is empty.")
    if len(image_bytes) > VLM_MAX_IMAGE_BYTES:
        raise VisionInputError("An image exceeds the decoded byte limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as opened:
                detected_mime_type = ALLOWED_FORMATS.get(opened.format or "")
                if detected_mime_type is None:
                    raise VisionInputError("Unsupported image format.")
                if detected_mime_type != declared_mime_type:
                    raise VisionInputError(
                        "Declared MIME type does not match the image content."
                    )
                if getattr(opened, "n_frames", 1) != 1:
                    raise VisionInputError("Animated or multi-frame images are not allowed.")

                source_width, source_height = opened.size
                if source_width <= 0 or source_height <= 0:
                    raise VisionInputError("Image dimensions are invalid.")
                if source_width * source_height > VLM_MAX_SOURCE_PIXELS:
                    raise VisionInputError("Image exceeds the source pixel limit.")

                opened.seek(0)
                opened.load()
                normalized = ImageOps.exif_transpose(opened)
                if normalized.mode not in {"RGB", "RGBA"}:
                    normalized = normalized.convert("RGBA" if "A" in normalized.mode else "RGB")
                normalized = normalized.copy()
    except VisionInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise VisionInputError("Image exceeds the safe decode limit.") from e
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise VisionInputError("Image cannot be decoded safely.") from e

    return PreparedImage(
        image=_resize_to_pixels(normalized, VLM_MAX_PIXELS_PER_IMAGE),
        mime_type=detected_mime_type,
        source_width=source_width,
        source_height=source_height,
    )


def _prepare_images(request: VisionAnalyzeRequest) -> tuple[list[str], list[dict[str, object]]]:
    if len(request.images) > VLM_MAX_IMAGES:
        raise VisionInputError(f"At most {VLM_MAX_IMAGES} images are allowed.")

    decoded_bytes = 0
    prepared: list[PreparedImage] = []
    for item in request.images:
        approximate_bytes = (len(item.data_base64) * 3) // 4
        decoded_bytes += approximate_bytes
        if decoded_bytes > VLM_MAX_TOTAL_IMAGE_BYTES:
            raise VisionInputError("Combined image data exceeds the decoded byte limit.")
        prepared.append(_decode_image(item.data_base64, item.mime_type))

    total_pixels = sum(item.image.width * item.image.height for item in prepared)
    if total_pixels > VLM_MAX_TOTAL_PIXELS:
        scale = math.sqrt(VLM_MAX_TOTAL_PIXELS / total_pixels)
        for item in prepared:
            target_pixels = max(1, int(item.image.width * item.image.height * scale * scale))
            item.image = _resize_to_pixels(item.image, target_pixels)

    data_urls: list[str] = []
    metadata: list[dict[str, object]] = []
    for item in prepared:
        output = io.BytesIO()
        item.image.save(output, format="PNG", optimize=True)
        output_bytes = output.getvalue()
        if len(output_bytes) > VLM_MAX_IMAGE_BYTES:
            raise VisionInputError("A normalized image exceeds the byte limit.")
        encoded = base64.b64encode(output_bytes).decode("ascii")
        data_urls.append(f"data:image/png;base64,{encoded}")
        metadata.append(
            {
                "mime_type": "image/png",
                "source_width": item.source_width,
                "source_height": item.source_height,
                "width": item.image.width,
                "height": item.image.height,
            }
        )
        item.image.close()

    return data_urls, metadata


def analyze(request: VisionAnalyzeRequest) -> dict[str, object]:
    if request.max_tokens > VLM_MAX_OUTPUT_TOKENS:
        raise VisionInputError(
            f"max_tokens cannot exceed the configured limit of {VLM_MAX_OUTPUT_TOKENS}."
        )

    image_data_urls, image_metadata = _prepare_images(request)
    text, usage, structured_output = clients.call_vlm(
        request.prompt.strip(),
        image_data_urls,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        priority=request.priority,
        response_schema=request.response_schema,
    )
    return {
        "model": clients.VLM_MODEL,
        "text": text,
        "structured_output": structured_output,
        "usage": usage,
        "images": image_metadata,
    }
