"""检查 FAISS 索引实际包含的向量数量"""
import os
from core.config import Config
from index.faiss_store import FaissIndex

def check_index_size():
    cfg = Config()
    
    # 检查所有启用的索引
    indices_to_check = []
    
    if cfg.get("indices", "bge_small", "enabled", default=False):
        indices_to_check.append(("BGE-small", cfg.get("indices", "bge_small", "index_path")))
    if cfg.get("indices", "bge", "enabled", default=False):
        indices_to_check.append(("BGE", cfg.get("indices", "bge", "index_path")))
    if cfg.get("indices", "jina", "enabled", default=False):
        indices_to_check.append(("Jina", cfg.get("indices", "jina", "index_path")))
    
    for name, path in indices_to_check:
        if not os.path.exists(path):
            print(f"[{name}] 索引文件不存在: {path}")
            continue
        
        # 获取文件大小
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # 加载索引
        idx = FaissIndex(index_type="flat_ip_fp16", save_path=path)
        idx.load()
        
        ntotal = idx.index.ntotal if hasattr(idx.index, 'ntotal') else 0
        dim = idx.index.d if hasattr(idx.index, 'd') else 0
        
        print(f"[{name}]")
        print(f"  文件路径: {path}")
        print(f"  文件大小: {size_mb:.2f} MB")
        print(f"  向量数量: {ntotal:,}")
        print(f"  向量维度: {dim}")
        
        if ntotal > 0 and dim > 0:
            # FP16: 2 bytes per dimension
            expected_mb = (ntotal * dim * 2) / (1024 * 1024)
            print(f"  预期大小: {expected_mb:.2f} MB (FP16)")
        print()

if __name__ == "__main__":
    check_index_size()
