"""FAISS 索引管理模块"""
import os
from typing import Dict, Optional
from core.config import Config
from index.faiss_store import FaissIndex


class IndexManager:
    """管理多个 FAISS 索引的初始化、加载和保存"""
    
    def __init__(self, cfg: Config, enabled_models: Dict[str, bool]):
        self.cfg = cfg
        self.enabled = enabled_models
        self.indices = {}
        self._init_indices()
    
    def _init_indices(self):
        """初始化所有启用的索引"""
        if self.enabled.get("jina", False):
            self.indices["jina"] = FaissIndex(
                index_type="flat_ip_fp16",
                save_path=self.cfg.get("indices", "jina", "index_path")
            )
        
        if self.enabled.get("e5", False):
            self.indices["e5"] = FaissIndex(
                index_type="flat_ip_fp16",
                save_path=self.cfg.get("indices", "e5", "index_path")
            )
        
        if self.enabled.get("gte", False):
            self.indices["gte"] = FaissIndex(
                index_type="flat_ip_fp16",
                save_path=self.cfg.get("indices", "gte", "index_path")
            )
        
        if self.enabled.get("bge", False):
            self.indices["bge"] = FaissIndex(
                index_type="flat_ip_fp16",
                save_path=self.cfg.get("indices", "bge", "index_path")
            )
        
        if self.enabled.get("bge_small", False):
            self.indices["bge_small"] = FaissIndex(
                index_type="flat_ip_fp16",
                save_path=self.cfg.get("indices", "bge_small", "index_path")
            )
        
        if self.enabled.get("qwen3", False):
            self.indices["qwen3"] = FaissIndex(
                index_type="flat_ip_fp16",
                save_path=self.cfg.get("indices", "qwen3", "index_path")
            )
    
    def load_existing(self):
        """加载已存在的索引文件（断点续跑）"""
        for name, idx in self.indices.items():
            path = self.cfg.get("indices", name, "index_path")
            if path and os.path.exists(path):
                try:
                    idx.load()
                    print(f"[resume] Loaded {name.upper()} index: {path}")
                except Exception as e:
                    print(f"[resume] Skip loading {name.upper()} index: {e}")
    
    def get_index(self, name: str) -> Optional[FaissIndex]:
        """获取指定的索引"""
        return self.indices.get(name)
    
    def save_all(self):
        """保存所有索引"""
        for name, idx in self.indices.items():
            try:
                idx.save()
            except Exception as e:
                print(f"[save] Error saving {name.upper()} index: {e}")
