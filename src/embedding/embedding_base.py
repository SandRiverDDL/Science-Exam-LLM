"""Embedding模型基类 - 实现模型解耦

提供统一的接口，方便切换不同的embedding模型
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch


class BaseEmbeddingModel(ABC):
    """Embedding模型抽象基类"""
    
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        max_length: int = 512,
        dtype: str = "float16"
    ):
        """
        Args:
            model_id: 模型ID或路径
            device: 设备（cuda/cpu）
            max_length: 最大序列长度
            dtype: 数据类型（float16/float32）
        """
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.dtype = dtype
        self.torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    
    @abstractmethod
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
            normalize: 是否归一化向量
            show_progress: 是否显示进度
        
        Returns:
            shape为(N, D)的numpy数组，N为文本数量，D为向量维度
        """
        pass
    
    @abstractmethod
    def get_dim(self) -> int:
        """获取向量维度
        
        Returns:
            向量维度
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称
        
        Returns:
            模型名称
        """
        pass
