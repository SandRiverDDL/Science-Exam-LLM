"""Qwen3 Embedding模型实现

基于transformers的Qwen3 0.6B embedding模型
"""
from typing import List, Union
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from retrieval.embedding_base import BaseEmbeddingModel


class Qwen3EmbeddingModel(BaseEmbeddingModel):
    """Qwen3 Embedding模型"""
    
    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        device: str = "cuda",
        max_length: int = 512,
        dtype: str = "float16",
        trust_remote_code: bool = True
    ):
        """
        Args:
            model_id: 模型ID（默认使用Qwen2-1.5B，也可以用更小的版本）
            device: 设备
            max_length: 最大序列长度
            dtype: 数据类型
            trust_remote_code: 是否信任远程代码
        """
        super().__init__(model_id, device, max_length, dtype)
        
        print(f"[Qwen3] 加载模型: {model_id}")
        print(f"  Device: {device}, Dtype: {dtype}, Max length: {max_length}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code
        ).to(device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 获取向量维度
        self._dim = self.model.config.hidden_size
        
        print(f"[Qwen3] 模型加载完成")
        print(f"  向量维度: {self._dim}")
    
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
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
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
