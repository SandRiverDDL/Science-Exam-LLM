"""嵌入模型初始化与管理模块"""
from typing import Optional, Dict, Any
from core.config import Config
from retrieval.embedding_jina_gguf import JinaEmbeddingGGUF
from retrieval.embedding_hf import E5EmbeddingHF, GTEEmbeddingHF, BGEEmbeddingHF, QwenEmbeddingHF


class EmbeddingManager:
    """管理多个嵌入模型的初始化和调用"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models = {}
        self.enabled = {}
        self._init_models()
    
    def _init_models(self):
        """初始化所有启用的嵌入模型"""
        # Jina GGUF
        self.enabled["jina"] = bool(self.cfg.get("indices", "jina", "enabled", default=False))
        if self.enabled["jina"]:
            self.models["jina"] = JinaEmbeddingGGUF(
                repo_id=self.cfg.embedding["repo_id"],
                filename=self.cfg.embedding["filename"],
                n_ctx=self.cfg.embedding.get("n_ctx", 8192),
                n_threads=self.cfg.embedding.get("n_threads", 8),
                n_batch=self.cfg.embedding.get("n_batch", 32),
                n_seq_max=self.cfg.embedding.get("n_seq_max", 128),
            )
        
        # E5
        self.enabled["e5"] = bool(self.cfg.get("indices", "e5", "enabled", default=False))
        e5_model_id = self.cfg.get("embedding_e5", "model_id", default=None)
        if self.enabled["e5"]:
            if not e5_model_id:
                print("[warn] E5 enabled but missing embedding_e5.model_id in config; disabling E5.")
                self.enabled["e5"] = False
            else:
                self.models["e5"] = E5EmbeddingHF(
                    model_id=e5_model_id,
                    device=self.cfg.get("embedding_e5", "device", default=None),
                    max_length=self.cfg.get("embedding_e5", "max_length", default=512),
                    dtype=self.cfg.get("embedding_e5", "dtype", default=None),
                )
        
        # GTE
        self.enabled["gte"] = bool(self.cfg.get("indices", "gte", "enabled", default=False))
        gte_model_id = self.cfg.get("embedding_gte", "model_id", default=None)
        if self.enabled["gte"]:
            if not gte_model_id:
                print("[warn] GTE enabled but missing embedding_gte.model_id in config; disabling GTE.")
                self.enabled["gte"] = False
            else:
                self.models["gte"] = GTEEmbeddingHF(
                    model_id=gte_model_id,
                    device=self.cfg.get("embedding_gte", "device", default=None),
                    max_length=self.cfg.get("embedding_gte", "max_length", default=512),
                    dtype=self.cfg.get("embedding_gte", "dtype", default=None),
                )
        
        # BGE
        self.enabled["bge"] = bool(self.cfg.get("indices", "bge", "enabled", default=False))
        bge_model_id = self.cfg.get("embedding_bge", "model_id", default=None)
        if self.enabled["bge"]:
            if not bge_model_id:
                print("[warn] BGE enabled but missing embedding_bge.model_id in config; disabling BGE.")
                self.enabled["bge"] = False
            else:
                self.models["bge"] = BGEEmbeddingHF(
                    model_id=bge_model_id,
                    device=self.cfg.get("embedding_bge", "device", default=None),
                    max_length=self.cfg.get("embedding_bge", "max_length", default=512),
                    dtype=self.cfg.get("embedding_bge", "dtype", default=None),
                )
        
        # BGE-small
        self.enabled["bge_small"] = bool(self.cfg.get("indices", "bge_small", "enabled", default=False))
        bge_small_model_id = self.cfg.get("embedding_bge_small", "model_id", default=None)
        if self.enabled["bge_small"]:
            if not bge_small_model_id:
                print("[warn] BGE-small enabled but missing embedding_bge_small.model_id in config; disabling BGE-small.")
                self.enabled["bge_small"] = False
            else:
                self.models["bge_small"] = BGEEmbeddingHF(
                    model_id=bge_small_model_id,
                    device=self.cfg.get("embedding_bge_small", "device", default=None),
                    max_length=self.cfg.get("embedding_bge_small", "max_length", default=512),
                    dtype=self.cfg.get("embedding_bge_small", "dtype", default=None),
                )
        
        # Qwen3
        self.enabled["qwen3"] = bool(self.cfg.get("indices", "qwen3", "enabled", default=False))
        qwen3_model_id = self.cfg.get("embedding_qwen3", "model_id", default=None)
        if self.enabled["qwen3"]:
            if not qwen3_model_id:
                print("[warn] Qwen3 enabled but missing embedding_qwen3.model_id in config; disabling Qwen3.")
                self.enabled["qwen3"] = False
            else:
                self.models["qwen3"] = QwenEmbeddingHF(
                    model_id=qwen3_model_id,
                    device=self.cfg.get("embedding_qwen3", "device", default=None),
                    max_length=self.cfg.get("embedding_qwen3", "max_length", default=512),
                    dtype=self.cfg.get("embedding_qwen3", "dtype", default=None),
                )
        
        # 打印设备信息
        for name, model in self.models.items():
            device = getattr(model, "device", "unknown")
            print(f"[init] {name.upper()} device: {device}")
    
    def get_model(self, name: str):
        """获取指定的嵌入模型"""
        return self.models.get(name)
    
    def is_enabled(self, name: str) -> bool:
        """检查模型是否启用"""
        return self.enabled.get(name, False)
    
    def get_chunking_tokenizer(self, fallback_tokenizer):
        """获取用于分块的 tokenizer（根据 embed_from_tokens 配置）"""
        use_token_ids = bool(self.cfg.get("pipeline", "embed_from_tokens", default=True))
        embed_tokens_model = str(self.cfg.get("pipeline", "embed_tokens_model", default="bge_small")).lower()
        
        if not use_token_ids:
            return fallback_tokenizer, False, None
        
        model_map = {
            "bge_small": "bge_small",
            "bge": "bge",
            "gte": "gte",
            "qwen3": "qwen3",
        }
        
        target_model = model_map.get(embed_tokens_model)
        if target_model and self.is_enabled(target_model):
            model = self.get_model(target_model)
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer:
                print(f"[tokenize] 使用 {target_model.upper()} 的 tokenizer 进行分块（直通 token ids）")
                # 提高 model_max_length 避免警告
                try:
                    if hasattr(tokenizer, "model_max_length"):
                        chunk_size = self.cfg.data["chunking"]["chunk_size_tokens"]
                        tokenizer.model_max_length = max(
                            getattr(tokenizer, "model_max_length", 512) or 512,
                            max(chunk_size, 1_000_000_000)
                        )
                except Exception:
                    pass
                return tokenizer, True, embed_tokens_model
        
        print(f"[tokenize] 直通 token ids 需要 {embed_tokens_model} 的 tokenizer，但模型未启用或不可用，回退为文本路径")
        return fallback_tokenizer, False, None
