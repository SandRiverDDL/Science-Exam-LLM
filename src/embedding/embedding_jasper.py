"""Jasper Token Compression Embedding模型实现

基于sentence-transformers的Jasper-Token-Compression-600M模型，支持flash-attn优化
"""
from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

from retrieval.embedding_base import BaseEmbeddingModel


class JasperEmbeddingModel(BaseEmbeddingModel):
    """Jasper Token Compression Embedding模型"""
    
    def __init__(
        self,
        model_id: str = "infgrad/Jasper-Token-Compression-600M",
        device: str = "cuda",
        max_length: int = 512,
        dtype: str = "float16",
        trust_remote_code: bool = True,
        use_flash_attn: bool = False,
        enable_tokenizer_cache: bool = True
    ):
        """
        Args:
            model_id: 模型ID（默认使用Jasper-Token-Compression-600M）
            device: 设备
            max_length: 最大序列长度
            dtype: 数据类型
            trust_remote_code: 是否信任远程代码
            use_flash_attn: 是否使用flash-attn优化
            enable_tokenizer_cache: 是否启用性能监控
        """
        super().__init__(model_id, device, max_length, dtype)
        
        print(f"[Jasper] 加载模型: {model_id}")
        print(f"  Device: {device}, Dtype: {dtype}, Max length: {max_length}")
        if use_flash_attn:
            print(f"  启用 flash-attn 优化")
        
        # 加载SentenceTransformer模型
        model_kwargs = {
            "trust_remote_code": trust_remote_code
        }
        
        # flash-attn只支持float16/bfloat16，需要提前指定dtype
        if use_flash_attn and dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = SentenceTransformer(
            model_id,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
            device=device
        )
        
        # 如果没有在加载时设置dtype，事后转换
        if dtype == "float16" and device == "cuda" and not use_flash_attn:
            self.model = self.model.half()
        
        # 统计信息
        self.encode_times = []
        self.enable_tokenizer_cache = enable_tokenizer_cache
        if enable_tokenizer_cache:
            print(f"  启用 encode 性能监控")
        
        # 获取向量维度
        try:
            self._dim = self.model.get_sentence_embedding_dimension()
            if self._dim is None:
                # 如果get_sentence_embedding_dimension返回None，尝试编码一个样本获取维度
                print(f"[warn] get_sentence_embedding_dimension() 返回 None，尝试编码样本获取维度...")
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                self._dim = test_embedding.shape[1]
        except Exception as e:
            print(f"[error] 获取向量维度失败: {e}，尝试编码样本获取维度...")
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self._dim = test_embedding.shape[1]
        
        print(f"[Jasper] 模型加载完成")
        print(f"  向量维度: {self._dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """编码文本为向量（仅encode，不压缩）
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批量大小
            normalize: 是否L2归一化
            show_progress: 是否显示进度条
        
        Returns:
            shape为(N, D)的numpy数组
        """
        # 处理单个文本
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用SentenceTransformer的encode方法（不使用compression_ratio）
        encode_start = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        encode_time = time.time() - encode_start
        
        # 记录性能统计
        if self.enable_tokenizer_cache and len(texts) > 0:
            tokens_per_sec = len(texts) / (encode_time + 1e-9)
            self.encode_times.append((encode_time, tokens_per_sec))
        
        return embeddings
    
    def get_encode_stats(self) -> Optional[dict]:
        """获取encode性能统计"""
        if not self.encode_times:
            return None
        
        times = [t[0] for t in self.encode_times]
        speeds = [t[1] for t in self.encode_times]
        
        return {
            'avg_batch_time_ms': np.mean(times) * 1000,
            'total_encode_time_s': np.sum(times),
            'avg_texts_per_sec': np.mean(speeds),
            'num_batches': len(self.encode_times)
        }
    
    def get_dim(self) -> int:
        """获取向量维度"""
        return self._dim
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model_id.split('/')[-1]
    
    def reset_stats(self):
        """重置性能统计"""
        self.encode_times = []
