"""
从 Parquet 文件读取 chunk IDs，使用 BGE-small 生成向量并构建FAISS索引

特性：
1. 直接从 Parquet 读取 child_ids（token IDs），避免重复 tokenize
2. 使用 embed_from_ids 直通路径，获得最高速度
3. float16 + CUDA 加速
4. 大 batch_size 充分利用 GPU（1024~4096）
5. 按批写入 FAISS，支持断点续跑
6. 归一化向量 + IndexFlatIP（内积检索）
7. 不使用IndexIDMap2以节省内存（400万向量时ID映射表会占用30+GB内存）
8. Metadata 从 chunks Parquet 读取（不存储在索引中）

关键优化：
- 直接使用 IndexFlatIP.add() 而不是 IndexIDMap2.add_with_ids()
- 使用外部 chunk_id_map.json 存储 faiss_id -> chunk_id 映射
- 定期垃圾回收，释放内存
- 时间戳显示，方便监控进度
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import faiss

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from core.config import Config
from retrieval.embedding_hf import HFTextEmbedding


class EmbeddingBuilder:
    """从Parquet构建FAISS向量索引"""
    
    def __init__(
        self,
        parquet_dir: str,
        model_id: str,
        index_path: str,
        chunk_id_map_path: str,
        batch_size: int = 2048,
        device: str = "cuda",
        dtype: str = "float16",
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
    ):
        """
        Args:
            parquet_dir: chunks Parquet目录
            model_id: HF模型ID
            index_path: FAISS索引保存路径
            chunk_id_map_path: chunk_id映射保存路径（faiss_id -> chunk_id）
            batch_size: 嵌入批量大小（建议1024~4096）
            device: cuda或cpu
            dtype: float16或float32
            checkpoint_path: 断点文件路径
            resume: 是否启用断点续跑
        """
        self.parquet_dir = parquet_dir
        self.model_id = model_id
        self.index_path = index_path
        self.chunk_id_map_path = chunk_id_map_path
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path or "data/faiss/checkpoints/embedding_progress.json"
        self.resume = resume
        
        # 初始化模型
        print(f"[init] 加载模型: {model_id}")
        print(f"  Device: {device}, Dtype: {dtype}")
        self.embedding_model = HFTextEmbedding(
            model_id=model_id,
            device=device,
            max_length=256,  # BGE-small max_length
            dtype=dtype,
        )
        
        # 初始化FAISS索引（延迟创建，等待第一个向量确定维度）
        self.index = None
        self.dim = None
        
        # chunk_id映射：faiss_id -> chunk_id
        self.chunk_id_map = {}
        
        # 已处理的chunk_id集合
        self.processed_chunks = set()
        self.current_id = 0  # FAISS内部ID（连续递增）
        
        # 加载断点
        if resume and os.path.exists(self.checkpoint_path):
            self._load_checkpoint()
    
    def _init_index(self, dim: int):
        """初始化FAISS索引（不使用IndexIDMap2，直接用IndexFlatIP节省内存）"""
        if self.index is not None:
            return
        
        self.dim = dim
        print(f"[init] 创建FAISS索引: IndexFlatIP, 维度={dim}")
        # 直接使用IndexFlatIP（内积检索），向量已归一化
        # 不使用IndexIDMap2以节省内存（400万向量时ID映射表会占用大量内存）
        
        # 选项：使用IndexScalarQuantizer以FP16存储（节省50%空间）
        # self.index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        # self.index.train(np.zeros((1, dim), dtype=np.float32))  # FP16需要train
        
        # 默认：使用IndexFlatIP（FP32存储，更高精度）
        self.index = faiss.IndexFlatIP(dim)
        print(f"  索引类型: {type(self.index)}")
    
    def _load_checkpoint(self):
        """加载断点进度"""
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                self.processed_chunks = set(checkpoint.get('processed_chunk_ids', []))
                self.current_id = checkpoint.get('current_id', 0)
                print(f"[resume] 加载断点: 已处理 {len(self.processed_chunks):,} 个chunks")
                
                # 加载chunk_id映射
                if os.path.exists(self.chunk_id_map_path):
                    with open(self.chunk_id_map_path, 'r', encoding='utf-8') as f:
                        self.chunk_id_map = json.load(f)
                        # 转换键为int类型
                        self.chunk_id_map = {int(k): v for k, v in self.chunk_id_map.items()}
                    print(f"[resume] 加载chunk_id映射: {len(self.chunk_id_map):,} 条")
                
                # 加载已有的FAISS索引
                if os.path.exists(self.index_path):
                    print(f"[resume] 加载已有索引: {self.index_path}")
                    self.index = faiss.read_index(self.index_path)
                    self.dim = self.index.d
                    print(f"  索引类型: {type(self.index)}")
                    print(f"  索引维度: {self.dim}, 向量数: {self.index.ntotal}")
        except Exception as e:
            print(f"[resume] 加载断点失败: {e}，从头开始")
            self.processed_chunks = set()
            self.current_id = 0
    
    def _save_checkpoint(self):
        """保存断点进度"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        checkpoint = {
            'processed_chunk_ids': list(self.processed_chunks),
            'current_id': self.current_id,
            'total_chunks': len(self.processed_chunks),
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def _load_parquet_files(self) -> List[str]:
        """获取所有chunk Parquet文件"""
        parquet_dir = Path(self.parquet_dir)
        parquet_files = sorted(parquet_dir.glob("chunks_*.parquet"))
        print(f"[load] 找到 {len(parquet_files)} 个Parquet文件")
        return [str(p) for p in parquet_files]
    
    def _read_parquet_batch(self, parquet_path: str):
        """读取Parquet文件，返回批次数据
        
        Yields:
            (chunk_id, child_ids, rerank_text, doc_id) tuples
        """
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        for _, row in df.iterrows():
            chunk_id = row['chunk_id']
            
            # 跳过已处理的
            if self.resume and chunk_id in self.processed_chunks:
                continue
            
            # child_ids已经是list[int]（uint16转为int）
            child_ids = row['child_ids'].tolist() if hasattr(row['child_ids'], 'tolist') else list(row['child_ids'])
            rerank_text = row['rerank_text']
            doc_id = row['doc_id']
            
            yield chunk_id, child_ids, rerank_text, doc_id
    
    def build(self):
        """构建向量索引主流程"""
        print("=" * 80)
        print("[build] 开始构建FAISS索引")
        print(f"  Parquet目录: {self.parquet_dir}")
        print(f"  模型: {self.model_id}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Index路径: {self.index_path}")
        print(f"  Chunk ID映射: {self.chunk_id_map_path}")
        print("=" * 80)
        
        # 获取所有Parquet文件
        parquet_files = self._load_parquet_files()
        
        if not parquet_files:
            print("[error] 未找到Parquet文件！")
            return
        
        # metadata从chunks Parquet读取，不需要单独存储
        
        # 批次缓冲
        batch_chunk_ids = []
        batch_child_ids = []
        batch_rerank_texts = []
        batch_doc_ids = []
        
        # 统计
        total_chunks = 0
        total_vectors = 0
        start_time = time.time()
        last_save_time = start_time
        
        for parquet_path in parquet_files:
            print(f"\n[process] 处理: {parquet_path}")
            
            for chunk_id, child_ids, rerank_text, doc_id in self._read_parquet_batch(parquet_path):
                total_chunks += 1
                
                # 添加到批次
                batch_chunk_ids.append(chunk_id)
                batch_child_ids.append(child_ids)
                batch_rerank_texts.append(rerank_text)
                batch_doc_ids.append(doc_id)
                
                # 批次满了，进行嵌入
                if len(batch_child_ids) >= self.batch_size:
                    self._process_batch(
                        batch_chunk_ids,
                        batch_child_ids,
                    )
                    total_vectors += len(batch_child_ids)
                    
                    # 清空批次
                    batch_chunk_ids = []
                    batch_child_ids = []
                    batch_rerank_texts = []
                    batch_doc_ids = []
                    
                    # 进度显示（添加时间戳）
                    elapsed = time.time() - start_time
                    speed = total_vectors / elapsed if elapsed > 0 else 0
                    current_time = time.strftime("%H:%M:%S")
                    print(f"\r  [{current_time}] 已处理 {total_chunks:,} chunks | 已嵌入 {total_vectors:,} 向量 | 速度 {speed:.0f} vec/s", end='', flush=True)
                    
                    # 定期保存（每5分钟），释放内存
                    if time.time() - last_save_time > 300:  # 5分钟 = 300秒
                        current_time = time.strftime("%H:%M:%S")
                        print(f"\n[{current_time}] [save] 定期保存索引...")
                        self._save_index()
                        self._save_chunk_id_map()
                        self._save_checkpoint()
                        last_save_time = time.time()
                        # 强制垃圾回收，释放内存
                        import gc
                        gc.collect()
        
        # 处理剩余批次
        if batch_child_ids:
            self._process_batch(
                batch_chunk_ids,
                batch_child_ids,
            )
            total_vectors += len(batch_child_ids)
        
        # 最终保存
        print("\n[save] 保存最终索引...")
        self._save_index()
        self._save_chunk_id_map()
        self._save_checkpoint()
        
        # 统计
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("[stats] 构建完成")
        print(f"  总chunks: {total_chunks:,}")
        print(f"  总向量: {total_vectors:,}")
        print(f"  FAISS索引: {self.index.ntotal:,}")
        print(f"  索引维度: {self.dim}")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  速度: {total_vectors/elapsed:.0f} vec/s")
        print(f"  索引路径: {self.index_path}")
        print(f"  Chunk ID映射: {self.chunk_id_map_path}")
        print(f"  ✅ Metadata从chunks Parquet读取")
        print("=" * 80)
    
    def _process_batch(
        self,
        chunk_ids: List[str],
        child_ids_list: List[List[int]],
    ):
        """处理一个批次：嵌入 + 写入FAISS + 记录chunk_id映射"""
        if not child_ids_list:
            return
        
        # 1. 使用embed_from_ids直通路径生成向量
        embeddings = self._embed_batch_ids(child_ids_list)  # [B, D], numpy array
        
        # 2. 初始化索引（第一次）
        if self.index is None:
            self._init_index(embeddings.shape[1])
        
        # 3. 归一化（已在embed_batch_ids中完成，这里再次确认）
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-9, None)
        
        # 4. 转为float32（FAISS要求）
        embeddings = embeddings.astype(np.float32)
        
        # 5. 生成FAISS IDs（连续递增）
        faiss_ids = np.arange(self.current_id, self.current_id + len(chunk_ids), dtype=np.int64)
        
        # 6. 写入FAISS（直接使用add，不需要add_with_ids）
        self.index.add(embeddings)
        
        # 7. 记录chunk_id映射（faiss_id -> chunk_id）
        for i, chunk_id in enumerate(chunk_ids):
            self.chunk_id_map[int(faiss_ids[i])] = chunk_id
            
            # 标记已处理
            self.processed_chunks.add(chunk_id)
            self.current_id += 1
    
    def _embed_batch_ids(self, ids_list: List[List[int]]) -> np.ndarray:
        """使用embed_from_ids生成向量
        
        Returns:
            numpy array [B, D], float32, L2归一化
        """
        # 调用HFTextEmbedding的_embed_batch_ids
        with torch.inference_mode():
            vecs = self.embedding_model._embed_batch_ids(ids_list)  # [B, D], tensor, cpu
        
        # 转为numpy
        return vecs.numpy()
    
    def _save_index(self):
        """保存FAISS索引（使用LZ4压缩）"""
        if self.index is None:
            return
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # 计算索引大小
        index_size_mb = (self.index.ntotal * self.dim * 4) / (1024 * 1024)  # float32
        print(f"  索引大小估计: {index_size_mb:.1f} MB, 向量数: {self.index.ntotal:,}")
        
        # 使用LZ4压缩写入
        io_flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        faiss.write_index(self.index, self.index_path)
        
        # 注意：FAISS默认使用内部压缩，LZ4在文件级别
        # 如果需要额外的LZ4压缩，可以使用lz4.frame压缩整个文件
    
    def _save_chunk_id_map(self):
        """保存chunk_id映射（faiss_id -> chunk_id）"""
        if not self.chunk_id_map:
            return
        
        os.makedirs(os.path.dirname(self.chunk_id_map_path), exist_ok=True)
        
        with open(self.chunk_id_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_id_map, f, ensure_ascii=False, indent=2)
        
        print(f"[save] Chunk ID映射: {len(self.chunk_id_map):,} 条")


def main():
    """主函数"""
    # 加载配置
    cfg = Config()
    
    # 配置参数
    parquet_dir = os.path.join(
        cfg.get("preprocessing", "output_dir", default="data/processed/parquet"),
        "chunks"
    )
    
    model_id = cfg.get("embedding_bge_small", "model_id", default="BAAI/bge-small-en-v1.5")
    batch_size = cfg.get("pipeline", "batch_size", default=2048)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16"
    
    # BGE-small索引配置
    index_config = cfg.get("indices", "bge_small", default={})
    index_path = index_config.get("index_path", "data/faiss/bge_small_fp16_ip.faiss")
    chunk_id_map_path = index_config.get("chunk_id_map_path", "data/faiss/bge_small_chunk_id_map.json")
    
    checkpoint_path = cfg.get("pipeline", "resume", default={}).get(
        "checkpoint_path",
        "data/faiss/checkpoints/embedding_progress.json"
    )
    resume = cfg.get("pipeline", "resume", default={}).get("enabled", True)
    
    # 构建索引
    builder = EmbeddingBuilder(
        parquet_dir=parquet_dir,
        model_id=model_id,
        index_path=index_path,
        chunk_id_map_path=chunk_id_map_path,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    
    builder.build()


if __name__ == "__main__":
    main()
