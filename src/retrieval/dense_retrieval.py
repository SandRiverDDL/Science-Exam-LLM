"""密集检索 (Dense Embedding Retrieval)

使用Qwen3-0.6B embedding和FAISS索引进行密集检索
"""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss

try:
    from ..embedding.embedding_qwen import Qwen3EmbeddingModel
except ImportError:
    # 绝对导入方式（用于直接脚本运行）
    from embedding.embedding_qwen import Qwen3EmbeddingModel


class DenseRetriever:
    """密集检索器"""
    
    def __init__(
        self,
        index_path: str,
        chunk_ids_path: str,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        max_length: int = 168,
        dtype: str = "float16"
    ):
        """初始化密集检索器
        
        Args:
            index_path: FAISS索引路径
            chunk_ids_path: chunk_id映射文件路径
            model_id: embedding模型ID（默认Qwen/Qwen3-Embedding-0.6B）
            device: 计算设备
            max_length: 最大序列长度
            dtype: 数据类型
        """
        self.index_path = Path(index_path)
        self.chunk_ids_path = Path(chunk_ids_path)
        
        self.index = None
        self.chunk_ids = None
        self.embedding_model = None
        
        self._load_index()
        self._load_chunk_ids()
        self._init_embedding_model(model_id, device, max_length, dtype)
    
    def _load_index(self):
        """加载FAISS索引"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        
        self.index = faiss.read_index(str(self.index_path))
    
    def _load_chunk_ids(self):
        """加载chunk_id映射"""
        if not self.chunk_ids_path.exists():
            raise FileNotFoundError(f"映射文件不存在: {self.chunk_ids_path}")
        
        with open(self.chunk_ids_path, 'r', encoding='utf-8') as f:
            self.chunk_ids = json.load(f)
    
    def _init_embedding_model(self, model_id: str, device: str, max_length: int, dtype: str):
        """初始化embedding模型"""
        self.embedding_model = Qwen3EmbeddingModel(
            model_id=model_id,
            device=device,
            max_length=max_length,
            dtype=dtype
        )
    
    def retrieve(self, query: str, top_k: int = 600) -> List[Tuple[str, float]]:
        """检索最相似的chunks
        
        Args:
            query: 查询文本
            top_k: 返回top_k个结果
        
        Returns:
            [(chunk_id, distance), ...] 的列表，distance越小越相似
        """
        # Embedding查询
        query_embedding = self.embedding_model.encode([query], batch_size=1)[0]
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # 搜索FAISS索引（距离模型）
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 转换为chunk_id和距离
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk_id = self.chunk_ids[int(idx)]
            # 内积距离转换为相似度（内积越大越相似）
            similarity = float(dist)
            results.append((chunk_id, similarity))
        
        return results
