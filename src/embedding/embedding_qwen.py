"""Qwen3 Embedding模型实现

基于transformers的Qwen3 0.6B embedding模型，支持flash-attn优化
"""
from typing import List, Union, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
from .embedding_base import BaseEmbeddingModel



class Qwen3EmbeddingModel(BaseEmbeddingModel):
    """Qwen3 Embedding模型"""
    
    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        device: str = "cuda",
        max_length: int = 512,
        dtype: str = "float16",
        trust_remote_code: bool = True,
        use_flash_attn: bool = False,
        enable_tokenizer_cache: bool = True
    ):
        """
        Args:
            model_id: 模型 ID（默认使用Qwen2-1.5B，也可以用更小的版本）
            device: 设备
            max_length: 最大序列长度
            dtype: 数据类型
            trust_remote_code: 是否信任远程代码
            use_flash_attn: 是否使用flash-attn优化（需要CUDA病务器）
            enable_tokenizer_cache: 是否启用tokenizer缓存
        """
        super().__init__(model_id, device, max_length, dtype)
        
        print(f"[Qwen3] 加载模型: {model_id}")
        print(f"  Device: {device}, Dtype: {dtype}, Max length: {max_length}")
        if use_flash_attn:
            print(f"  启用 flash-attn 优化")
        
        # 加载tokenizer（使用fast=True获得Rust版本）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,  # Rust版本比Python快50倍
        )
        
        # tokenizer统计信息
        self.tokenize_times = []
        self.enable_tokenizer_cache = enable_tokenizer_cache
        if enable_tokenizer_cache:
            print(f"  启用 tokenizer 性能监控")
        
        # 加载模型
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": trust_remote_code
        }
        
        # 如果启用flash-attn，添加attn_implementation参数
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModel.from_pretrained(
            model_id,
            **model_kwargs
        ).to(device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 获取向量维度
        try:
            self._dim = self.model.config.hidden_size
            if self._dim is None or self._dim == 0:
                raise ValueError(f"hidden_size 无效: {self._dim}")
        except Exception as e:
            print(f"[error] 获取 hidden_size 失败: {e}，尝试编码样本获取维度...")
            with torch.no_grad():
                test_input = self.tokenizer(["test"], return_tensors="pt").to(device)
                test_output = self.model(**test_input)
                self._dim = test_output.last_hidden_state.shape[-1]
        
        print(f"[Qwen3] 模型加载完成")
        print(f"  向量维度: {self._dim}")
        print(f"  Tokenizer: {type(self.tokenizer).__name__} (use_fast=True)")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """编码文本为向量
        
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
        
        all_embeddings = []
        
        # 批量处理
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=len(texts)//batch_size + 1)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize（计时）
                tokenize_start = time.time()
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                tokenize_time = time.time() - tokenize_start
                
                if self.enable_tokenizer_cache and len(batch_texts) > 0:
                    tokens_per_sec = sum(len(t.split()) for t in batch_texts) / (tokenize_time + 1e-9)
                    self.tokenize_times.append((tokenize_time, tokens_per_sec))
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # 使用mean pooling提取embedding
                # outputs.last_hidden_state: (batch_size, seq_len, hidden_size)
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )
                
                # 转换为numpy
                embeddings_np = embeddings.cpu().numpy()
                
                # L2归一化
                if normalize:
                    embeddings_np = self._normalize(embeddings_np)
                
                all_embeddings.append(embeddings_np)
        
        # 合并所有批次
        return np.vstack(all_embeddings)
    
    def get_tokenizer_stats(self) -> Optional[dict]:
        """获取tokenizer性能统计"""
        if not self.tokenize_times:
            return None
        
        times = [t[0] for t in self.tokenize_times]
        speeds = [t[1] for t in self.tokenize_times]
        
        return {
            'avg_batch_time_ms': np.mean(times) * 1000,
            'total_tokenize_time_s': np.sum(times),
            'avg_tokens_per_sec': np.mean(speeds),
            'num_batches': len(self.tokenize_times)
        }
    
    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            (batch_size, hidden_size)
        """
        # 扩展attention_mask维度以匹配hidden_states
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # 求和
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        
        # 计算有效token数量
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # 平均
        return sum_embeddings / sum_mask
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2归一化
        
        Args:
            embeddings: (N, D)
        
        Returns:
            归一化后的embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # 避免除零
        return embeddings / norms
    
    def get_dim(self) -> int:
        """获取向量维度"""
        return self._dim
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model_id.split('/')[-1]
    
    def reset_stats(self):
        """重置性能统计"""
        self.tokenize_times = []
