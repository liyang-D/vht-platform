import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from services.orchestrator import clients, vision
from services.orchestrator.schemas import VisionAnalyzeRequest


def encoded_image(
    image_format: str = "JPEG",
    size: tuple[int, int] = (64, 64),
) -> str:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, image_format)
    return base64.b64encode(output.getvalue()).decode("ascii")


class VisionInputTests(unittest.TestCase):
    def test_resizes_large_image_and_normalizes_to_png(self):
        request = VisionAnalyzeRequest(
            prompt="Inspect the drawing.",
            images=[
                {
                    "mime_type": "image/jpeg",
                    "data_base64": encoded_image(size=(4000, 2000)),
                }
            ],
        )

        data_urls, metadata = vision._prepare_images(request)

        self.assertTrue(data_urls[0].startswith("data:image/png;base64,"))
        self.assertLessEqual(
            metadata[0]["width"] * metadata[0]["height"],
            vision.VLM_MAX_PIXELS_PER_IMAGE,
        )

    def test_rejects_mime_type_mismatch(self):
        request = VisionAnalyzeRequest(
            prompt="Inspect.",
            images=[
                {
                    "mime_type": "image/png",
                    "data_base64": encoded_image("JPEG"),
                }
            ],
        )

        with self.assertRaises(vision.VisionInputError):
            vision._prepare_images(request)

    def test_rejects_invalid_base64(self):
        request = VisionAnalyzeRequest(
            prompt="Inspect.",
            images=[{"mime_type": "image/png", "data_base64": "not-base64"}],
        )

        with self.assertRaises(vision.VisionInputError):
            vision._prepare_images(request)

    def test_schema_rejects_more_than_nine_images(self):
        image = {"mime_type": "image/jpeg", "data_base64": encoded_image()}

        with self.assertRaises(ValidationError):
            VisionAnalyzeRequest(prompt="Inspect.", images=[image] * 10)

    def test_schema_rejects_blank_prompt_and_unknown_fields(self):
        image = {"mime_type": "image/jpeg", "data_base64": encoded_image()}

        with self.assertRaises(ValidationError):
            VisionAnalyzeRequest(prompt="   ", images=[image])
        with self.assertRaises(ValidationError):
            VisionAnalyzeRequest(
                prompt="Inspect.",
                images=[{**image, "image_url": "https://example.invalid/image.jpg"}],
            )

    @patch("services.orchestrator.vision.clients.call_vlm")
    def test_response_exposes_logical_model_name(self, mock_call_vlm):
        mock_call_vlm.return_value = ("ok", {"total_tokens": 1}, None)
        request = VisionAnalyzeRequest(
            prompt="Inspect.",
            images=[
                {
                    "mime_type": "image/jpeg",
                    "data_base64": encoded_image(),
                }
            ],
        )

        result = vision.analyze(request)

        self.assertEqual(result["model"], clients.VLM_MODEL)
        self.assertNotIn("FP8", result["model"])
        mock_call_vlm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
