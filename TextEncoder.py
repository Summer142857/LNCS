"""Frozen text encoders. Credentials are read by Hugging Face from HF_TOKEN/login."""

import hashlib
import os

import torch

ENCODER_DIM_DICT = {
    "ST": 768,
    "e5": 1024,
    "llama2_7b": 4096,
    "llama3_8b": 4096,
    "qwen3": 2560,
    "mock": 64,
}
MODEL_IDS = {
    "ST": "sentence-transformers/multi-qa-distilbert-cos-v1",
    "e5": "intfloat/e5-large-v2",
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B",
    "qwen3": "Qwen/Qwen3-4B",
}


class TextEncoder:
    def __init__(
        self,
        name,
        root=None,
        batch_size=1,
        device="cpu",
        local_files_only=False,
        pooling="legacy_mean",
        dtype="float32",
    ):
        if name not in ENCODER_DIM_DICT:
            raise ValueError(f"Unsupported encoder: {name}")
        if pooling not in ("legacy_mean", "masked_mean"):
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.name, self.root = name, root
        self.batch_size, self.device = batch_size, torch.device(device)
        self.local_files_only, self.pooling = local_files_only, pooling
        self.dtype = getattr(torch, dtype)
        self.model = self.tokenizer = None

    def get_model(self):
        if self.name == "mock":
            return
        options = dict(
            cache_dir=self.root,
            local_files_only=self.local_files_only,
            token=os.environ.get("HF_TOKEN"),
        )
        if self.name == "ST":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                MODEL_IDS[self.name],
                device=str(self.device),
                cache_folder=self.root,
                local_files_only=self.local_files_only,
                token=os.environ.get("HF_TOKEN"),
            )
        else:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_IDS[self.name], **options
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.tokenizer.truncation_side = "right"
            self.model = AutoModel.from_pretrained(
                MODEL_IDS[self.name], torch_dtype=self.dtype, **options
            ).to(self.device)
            if self.model.config.hidden_size != ENCODER_DIM_DICT[self.name]:
                raise ValueError(
                    "The encoder dimension does not match the configured model."
                )
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts, to_tensor=True):
        if not texts:
            result = torch.empty(0, ENCODER_DIM_DICT[self.name], device=self.device)
        elif self.name == "mock":
            # For software tests only: no language understanding or paper performance.
            rows = []
            for text in texts:
                seed = int.from_bytes(
                    hashlib.sha256(text.encode()).digest()[:8], "little"
                )
                generator = torch.Generator(device="cpu").manual_seed(seed)
                rows.append(
                    torch.randn(
                        ENCODER_DIM_DICT["mock"], generator=generator, device="cpu"
                    )
                )
            result = torch.stack(rows).to(self.device)
        elif self.name == "ST":
            result = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
            ).to(self.device)
        else:
            result = []
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding="longest",
                    truncation=True,
                    max_length=512 if self.name == "e5" else 500,
                    return_tensors="pt",
                ).to(self.device)
                # Original causal encoders averaged all positions, including padding,
                # without an attention mask. Keep this mode explicit for old weights.
                legacy = self.pooling == "legacy_mean" and self.name != "e5"
                inputs = {"input_ids": tokens.input_ids} if legacy else dict(tokens)
                hidden = self.model(**inputs, return_dict=True).last_hidden_state
                if legacy:
                    row = hidden.mean(dim=1)
                else:
                    mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)
                    row = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                result.append(row.detach())
            result = torch.cat(result)
        # SentenceTransformer.encode can return inference-mode tensors. Although
        # the encoder is frozen, the downstream trainable connector must save
        # its input for backward; give it an ordinary tensor with identical data.
        if torch.is_inference(result):
            result = result.clone()
        if to_tensor:
            return result
        # NumPy has no native bfloat16 dtype.
        if result.dtype == torch.bfloat16:
            result = result.float()
        return result.cpu().numpy()
