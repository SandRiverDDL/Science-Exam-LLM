import os
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModel


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class HFTextEmbedding:
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        max_length: int = 512,
        trust_remote_code: bool = True,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        if not model_id:
            raise ValueError("model_id is required for HFTextEmbedding")
        hf_token = os.environ.get("huggingface_token", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_token,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 解析 dtype：支持字符串或 torch.dtype；默认在 CUDA 上用 float16，否则 float32
        if isinstance(dtype, str):
            dtype = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(dtype.lower(), None)
        if dtype is None:
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_id,
            use_auth_token=hf_token,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        self.model.eval()
        self.model.to(self.device)
        self.max_length = max_length
        self.dtype = dtype

    def _embed_batch(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("texts list is empty")
        with torch.inference_mode():
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            # 尽量异步搬运到 GPU
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            if self.device.startswith("cuda"):
                # 在 GPU 上使用 AMP 提升吞吐
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)
            last_hidden = out.last_hidden_state
            pooled = _mean_pool(last_hidden, enc["attention_mask"])  # [B, D]
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return normed.cpu()

    def _embed_batch_ids(self, ids_list: List[List[int]]) -> torch.Tensor:
        """直接使用 token ids 进行嵌入，避免重复分词。
        会根据可用的特殊符号添加起止符（优先 CLS/SEP，其次 BOS/EOS），并安全填充。
        """
        if not ids_list:
            raise ValueError("ids_list is empty")
        # 选择起止特殊符：优先 CLS/SEP；否则 BOS/EOS；否则不加
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        start_id = cls_id if isinstance(cls_id, int) and cls_id >= 0 else (bos_id if isinstance(bos_id, int) and bos_id >= 0 else None)
        end_id = sep_id if isinstance(sep_id, int) and sep_id >= 0 else (eos_id if isinstance(eos_id, int) and eos_id >= 0 else None)
        add_start = 1 if isinstance(start_id, int) else 0
        add_end = 1 if isinstance(end_id, int) else 0
        # 内容部分的最大长度，确保加上特殊符号后不超过 max_length
        content_max = max(len(x) for x in ids_list) if ids_list else 0
        max_len = min(content_max, self.max_length - (add_start + add_end))
        # 构造 input_ids 与 attention_mask
        input_ids = []
        attn_mask = []
        for ids in ids_list:
            s = ids[:max_len]
            if add_start:
                s = [start_id] + s
            if add_end:
                s = s + [end_id]
            # pad 到 self.max_length（考虑特殊符号已加入）
            total_len = len(s)
            pad_len = self.max_length - total_len
            input_ids.append(s + [pad_id] * pad_len)
            attn_mask.append([1] * total_len + [0] * pad_len)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)
        # 对仅 BERT/类似模型提供 token_type_ids；Decoder-only 模型不需要
        has_type_ids = hasattr(getattr(self.model, "config", None), "type_vocab_size") and getattr(self.model.config, "type_vocab_size") and getattr(self.model.config, "type_vocab_size") > 1
        enc = {
            "input_ids": input_ids.to(self.device, non_blocking=True),
            "attention_mask": attn_mask.to(self.device, non_blocking=True),
        }
        if has_type_ids:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            enc["token_type_ids"] = token_type_ids.to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)
            last_hidden = out.last_hidden_state
            pooled = _mean_pool(last_hidden, enc["attention_mask"])  # [B, D]
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return normed.cpu()

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Embedding texts list is empty.")
        vecs = self._embed_batch(texts)
        return vecs.tolist()


class E5EmbeddingHF(HFTextEmbedding):
    """E5 instruct-style embeddings. For passages, prefix with 'passage: '."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Embedding texts list is empty.")
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self._embed_batch(prefixed)
        return vecs.tolist()


class GTEEmbeddingHF(HFTextEmbedding):
    """GTE multilingual base embeddings. Standard mean pooling + L2 norm."""
    def embed_from_ids(self, ids_list: List[List[int]]) -> List[List[float]]:
        vecs = self._embed_batch_ids(ids_list)
        return vecs.tolist()


class BGEEmbeddingHF(HFTextEmbedding):
    """BGE-m3 embeddings for documents (no special prefixes)."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Embedding texts list is empty.")
        vecs = self._embed_batch(texts)
        return vecs.tolist()

    def embed_from_ids(self, ids_list: List[List[int]]) -> List[List[float]]:
        vecs = self._embed_batch_ids(ids_list)
        return vecs.tolist()


class QwenEmbeddingHF(HFTextEmbedding):
    """Qwen/Qwen3-Embedding-0.6B embeddings. Mean pooling + L2 norm.
    使用 AutoModel + AutoTokenizer，支持 FP16 在 CUDA 上加速。
    """
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Embedding texts list is empty.")
        vecs = self._embed_batch(texts)
        return vecs.tolist()

    def embed_from_ids(self, ids_list: List[List[int]]) -> List[List[float]]:
        vecs = self._embed_batch_ids(ids_list)
        return vecs.tolist()
