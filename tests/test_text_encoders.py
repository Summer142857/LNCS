"""Offline integration checks using tiny randomly initialized HF backbones.

These fixtures exercise real tokenizer/model APIs without downloading weights.
They do not evaluate pretrained-model quality or reproduce paper metrics.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import TextEncoder as encoders

try:
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@unittest.skipUnless(
    HAS_TRANSFORMERS, "Install the full requirements for text-encoder integration tests"
)
class TextEncoderIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_device("cpu")
        torch.manual_seed(13)
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, family):
        vocab = {
            word: i
            for i, word in enumerate(
                [
                    "[PAD]",
                    "[UNK]",
                    "[BOS]",
                    "[EOS]",
                    "visit",
                    "all",
                    "cities",
                    "with",
                    "demand",
                    "select",
                    "items",
                    "weight",
                    "capacity",
                ]
            )
        }
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast = transformers.PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            model_input_names=["input_ids", "attention_mask"],
        )
        common = dict(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            max_position_embeddings=64,
        )
        if family == "e5":
            model = transformers.BertModel(transformers.BertConfig(**common))
        elif family == "qwen3":
            model = transformers.Qwen3Model(
                transformers.Qwen3Config(
                    **common, num_key_value_heads=2, head_dim=8, attention_dropout=0.0
                )
            )
        else:
            model = transformers.LlamaModel(
                transformers.LlamaConfig(
                    **common, num_key_value_heads=2, attention_dropout=0.0
                )
            )
        directory = self.root / family
        model.save_pretrained(directory)
        fast.save_pretrained(directory)
        return directory

    def encoder(self, name, path, pooling="legacy_mean", dtype="float32"):
        self.enterContext(patch.dict(encoders.MODEL_IDS, {name: str(path)}))
        self.enterContext(patch.dict(encoders.ENCODER_DIM_DICT, {name: 16}))
        encoder = encoders.TextEncoder(
            name,
            device="cpu",
            batch_size=2,
            pooling=pooling,
            dtype=dtype,
            local_files_only=True,
        )
        encoder.get_model()
        return encoder

    def test_all_causal_backends_match_legacy_pooling_and_are_frozen(self):
        texts = ["visit cities", "visit all cities with demand"]
        for name in ("llama2_7b", "llama3_8b", "qwen3"):
            with self.subTest(encoder=name):
                encoder = self.encoder(name, self.fixture(name))
                actual = encoder.encode(texts)
                tokens = encoder.tokenizer(
                    texts, padding="longest", return_tensors="pt"
                )
                with torch.no_grad():
                    reference = encoder.model(
                        input_ids=tokens.input_ids, return_dict=True
                    ).last_hidden_state.mean(1)
                torch.testing.assert_close(actual, reference)
                self.assertEqual(actual.shape, (2, 16))
                self.assertFalse(actual.requires_grad)
                self.assertFalse(encoder.model.training)
                self.assertTrue(
                    all(not p.requires_grad for p in encoder.model.parameters())
                )

    def test_masked_pooling_is_independent_of_padding(self):
        texts = ["visit cities", "visit all cities with demand"]
        for name in ("e5", "llama2_7b", "qwen3"):
            with self.subTest(encoder=name):
                encoder = self.encoder(name, self.fixture(name), pooling="masked_mean")
                together = encoder.encode(texts)
                alone = encoder.encode(texts[:1])
                torch.testing.assert_close(together[0], alone[0], rtol=1e-4, atol=1e-5)
                self.assertEqual(encoder.encode([]).shape, (0, 16))
                self.assertEqual(encoder.encode(texts, to_tensor=False).shape, (2, 16))

    def test_bfloat16_encoder_can_return_numpy(self):
        encoder = self.encoder("llama2_7b", self.fixture("llama2_7b"), dtype="bfloat16")
        result = encoder.encode(["visit cities"], to_tensor=False)
        self.assertEqual(result.shape, (1, 16))
        self.assertTrue(np.isfinite(result).all())

    def test_sentence_transformer_embeddings_allow_connector_backward(self):
        try:
            from sentence_transformers import SentenceTransformer, models as st_models
        except ImportError:
            self.skipTest("Install sentence-transformers for this integration test")
        base = self.fixture("e5")
        local = self.root / "tiny-st"
        backbone = st_models.Transformer(str(base), max_seq_length=64)
        pooling = st_models.Pooling(backbone.get_word_embedding_dimension())
        SentenceTransformer(modules=[backbone, pooling], device="cpu").save(str(local))
        encoder = self.encoder("ST", local)
        embedding = encoder.encode(["visit cities", "select items"])
        self.assertFalse(torch.is_inference(embedding))
        connector = torch.nn.Linear(16, 4)
        connector(embedding).square().mean().backward()
        self.assertIsNotNone(connector.weight.grad)
        self.assertTrue(torch.isfinite(connector.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in encoder.model.parameters()))

    def test_missing_local_model_fails_without_network(self):
        with patch.dict(encoders.MODEL_IDS, {"e5": str(self.root / "not-present")}):
            encoder = encoders.TextEncoder("e5", local_files_only=True)
            with self.assertRaises((OSError, ValueError)):
                encoder.get_model()


if __name__ == "__main__":
    unittest.main()
