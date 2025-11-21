import os
import faiss
import numpy as np
from typing import List, Optional

class FaissIndex:
    def __init__(self, index_type: str, save_path: str, dim: Optional[int] = None):
        self.index_type = index_type
        self.save_path = save_path
        self.dim = dim
        self.index = None

    def _ensure_index(self, dim: int):
        if self.index is not None:
            return
        if self.index_type == "flat_ip_fp16":
            # 使用 ScalarQuantizer + QT_fp16，保持内积度量，同时 FP16 存储
            self.index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
            # QT_fp16 不需要训练，但调用 train 也可（安全起见）
            self.index.train(np.zeros((1, dim), dtype=np.float32))
        elif self.index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")

    def build(self, vectors: List[List[float]]):
        if not vectors:
            return
        # 过滤不合法或维度不一致的向量
        filtered = []
        expected_dim = None
        for v in vectors:
            if not isinstance(v, (list, tuple)) or not v:
                continue
            if expected_dim is None:
                expected_dim = len(v)
            if len(v) != expected_dim:
                continue
            filtered.append(v)
        if not filtered:
            raise ValueError("No valid vectors to build FAISS index (empty or inconsistent dimensions).")

        mat = np.asarray(filtered, dtype=np.float32)
        # 统一为二维矩阵形状 (n, d)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        elif mat.ndim > 2:
            mat = mat.reshape(mat.shape[0], -1)
        mat = np.ascontiguousarray(mat, dtype=np.float32)  # 保证内存连续
        dim = mat.shape[1]
        self._ensure_index(dim)
        self.index.add(mat)

    def search(self, queries: List[List[float]], top_k: int):
        mat = np.array(queries, dtype=np.float32)
        D, I = self.index.search(mat, top_k)
        return D, I

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        faiss.write_index(self.index, self.save_path)

    def load(self):
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(self.save_path)
        self.index = faiss.read_index(self.save_path)
        return self.index