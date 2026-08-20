import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.orchestrator import lora


class LoraRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.adapter_dir = self.root / "clinical-v1"
        self.adapter_dir.mkdir()
        (self.adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": lora.LLM_MODEL,
                    "peft_type": "LORA",
                    "r": 8,
                }
            ),
            encoding="utf-8",
        )
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

        self.root_patch = patch.object(lora, "LORA_ROOT", self.root)
        self.revision_patch = patch.object(lora, "LLM_MODEL_REVISION", "revision-1")
        self.root_patch.start()
        self.revision_patch.start()

    def tearDown(self) -> None:
        self.revision_patch.stop()
        self.root_patch.stop()
        self.temp_dir.cleanup()

    def test_inspects_peft_adapter(self) -> None:
        (self.adapter_dir / "training_summary.json").write_text(
            json.dumps(
                {"base_model": lora.LLM_MODEL, "base_revision": "revision-1"}
            ),
            encoding="utf-8",
        )

        adapter = lora.inspect_adapter("clinical-v1")

        self.assertEqual(adapter.rank, 8)
        self.assertEqual(adapter.base_revision, "revision-1")
        self.assertEqual(adapter.weights_file, "adapter_model.safetensors")

    def test_rejects_rank_above_runtime_limit(self) -> None:
        config_path = self.adapter_dir / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["r"] = 32
        config_path.write_text(json.dumps(config), encoding="utf-8")

        with patch.object(lora, "LLM_MAX_LORA_RANK", 16):
            with self.assertRaises(lora.LoraValidationError):
                lora.inspect_adapter("clinical-v1")

    def test_rejects_path_traversal(self) -> None:
        with self.assertRaises(lora.LoraValidationError):
            lora.inspect_adapter("../outside")

    def test_rejects_wrong_base_revision(self) -> None:
        (self.adapter_dir / "training_summary.json").write_text(
            json.dumps(
                {"base_model": lora.LLM_MODEL, "base_revision": "revision-2"}
            ),
            encoding="utf-8",
        )

        with self.assertRaises(lora.LoraValidationError):
            lora.inspect_adapter("clinical-v1")

    @patch("services.orchestrator.lora.get_loaded_model_ids", return_value=set())
    @patch("services.orchestrator.lora.httpx.post")
    def test_loads_adapter_by_name_only(self, post, _loaded_models) -> None:
        post.return_value.raise_for_status.return_value = None

        adapter, changed = lora.load_adapter("clinical-v1")

        self.assertTrue(changed)
        self.assertTrue(adapter.loaded)
        post.assert_called_once_with(
            f"{lora.LLM_RUNTIME_URL}/v1/load_lora_adapter",
            json={
                "lora_name": "clinical-v1",
                "lora_path": str(self.adapter_dir.resolve()),
            },
            timeout=lora.LORA_RUNTIME_TIMEOUT_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
