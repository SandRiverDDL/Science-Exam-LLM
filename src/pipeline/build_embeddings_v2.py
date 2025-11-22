"""构建FAISS索引 V2

核心改进：
1. 使用IndexFlatIP（最高精度）
2. 从chunks.parquet读取字符索引，从documents_cleaned.parquet读取原文
3. 模型解耦，支持切换embedding模型（Qwen3/BGE/etc）
4. float16 + CUDA加速
5. FAISS索引用LZ4压缩
6. 批量写入 + 断点续跑
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import faiss

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from core.config import Config
from retrieval.embedding_qwen import Qwen3EmbeddingModel
from retrieval.embedding_base import BaseEmbeddingModel


class EmbeddingBuilderV2:
    """FAISS索引构建器 V2"""
    
    def __init__(
        self,
        chunks_path: str,
        documents_path: str,
        embedding_model: BaseEmbeddingModel,
        index_path: str,
        batch_size: int = 1024,
        checkpoint_path: Optional[str] = None,
        resume: bool = True,
    ):
        """
        Args:
            chunks_path: chunks.parquet路径
            documents_path: documents_cleaned.parquet路径
            embedding_model: embedding模型实例
            index_path: FAISS索引保存路径
            batch_size: 批量大小
            checkpoint_path: 断点文件路径
            resume: 是否启用断点续跑
        """
        self.chunks_path = chunks_path
        self.documents_path = documents_path
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path or "data/faiss/checkpoints/embedding_v2_progress.json"
        self.resume = resume
        
        # 加载documents（全部载入内存，因为只有几百MB）
        print(f"[init] 加载文档数据: {documents_path}")
        df_docs = pd.read_parquet(documents_path)
        # 创建doc_id到text的映射
        self.doc_texts = dict(zip(df_docs['doc_id'], df_docs['text']))
        print(f"  加载了 {len(self.doc_texts):,} 个文档")
        
        # 初始化FAISS索引
        self.index = None
        self.dim = embedding_model.get_dim()
        
        # 已处理的chunk_id集合
        self.processed_chunks = set()
        self.chunk_id_list = []  # 保持chunk_id顺序（faiss_id -> chunk_id）
        
        # 加载断点
        if resume and os.path.exists(self.checkpoint_path):
            self._load_checkpoint()
    
    def _init_index(self):
        """初始化FAISS索引（IndexFlatIP）"""
        if self.index is not None:
            return
        
        print(f"[init] 创建FAISS索引: IndexFlatIP, 维度={self.dim}")
        # 使用IndexFlatIP（内积检索），向量会被归一化
        self.index = faiss.IndexFlatIP(self.dim)
        print(f"  索引类型: {type(self.index)}")
    
    def _load_checkpoint(self):
        """加载断点进度"""
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                self.processed_chunks = set(checkpoint.get('processed_chunk_ids', []))
                self.chunk_id_list = checkpoint.get('chunk_id_list', [])
                print(f"[resume] 加载断点: 已处理 {len(self.processed_chunks):,} 个chunks")
                
                # 加载已有的FAISS索引
                if os.path.exists(self.index_path):
                    print(f"[resume] 加载已有索引: {self.index_path}")
                    self.index = faiss.read_index(self.index_path)
                    print(f"  索引维度: {self.index.d}, 向量数: {self.index.ntotal}")
        except Exception as e:
            print(f"[resume] 加载断点失败: {e}，从头开始")
            self.processed_chunks = set()
            self.chunk_id_list = []
    
    def _save_checkpoint(self):
        """保存断点进度"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        checkpoint = {
            'processed_chunk_ids': list(self.processed_chunks),
            'chunk_id_list': self.chunk_id_list,
            'total_chunks': len(self.processed_chunks),
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def _save_index(self):
        """保存FAISS索引（使用LZ4压缩）"""
        if self.index is None:
            return
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # 保存索引
        faiss.write_index(self.index, self.index_path)
        
        # 压缩索引（使用LZ4）
        import lz4.frame
        
        compressed_path = self.index_path + ".lz4"
        with open(self.index_path, 'rb') as f_in:
            with lz4.frame.open(compressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # 显示文件大小
        raw_size = os.path.getsize(self.index_path) / (1024**2)
        compressed_size = os.path.getsize(compressed_path) / (1024**2)
        ratio = (1 - compressed_size / raw_size) * 100 if raw_size > 0 else 0
        
        print(f"[save] 索引已保存: {self.index_path}")
        print(f"  原始大小: {raw_size:.2f} MB")
        print(f"  压缩后: {compressed_size:.2f} MB (LZ4)")
        print(f"  压缩率: {ratio:.1f}%")
    
    def _save_chunk_id_mapping(self):
        """保存chunk_id映射（faiss_id -> chunk_id）"""
        mapping_path = self.index_path.replace('.index', '_chunk_ids.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_id_list, f, ensure_ascii=False, indent=2)
        print(f"[save] Chunk ID映射已保存: {mapping_path}")
        print(f"  映射数量: {len(self.chunk_id_list):,}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示
        
        Args:
            seconds: 秒数
        
        Returns:
            格式化后的时间字符串 (HH:MM:SS)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def build(self):
        """构建索引主流程"""
        print("=" * 80)
        print("[build] 开始构建FAISS索引 V2")
        print(f"  Chunks路径: {self.chunks_path}")
        print(f"  Documents路径: {self.documents_path}")
        print(f"  模型: {self.embedding_model.get_model_name()}")
        print(f"  向量维度: {self.dim}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Index路径: {self.index_path}")
        print("=" * 80)
        
        # 初始化索引
        self._init_index()
        
        # 读取chunks
        print(f"\n[load] 读取chunks...")
        df_chunks = pd.read_parquet(self.chunks_path)
        total_chunks = len(df_chunks)
        print(f"  总chunks: {total_chunks:,}")
        
        # 过滤已处理的
        if self.resume and self.processed_chunks:
            df_chunks = df_chunks[~df_chunks['chunk_id'].isin(self.processed_chunks)]
            print(f"  剩余未处理: {len(df_chunks):,}")
        
        # 批量处理
        total_processed = len(self.processed_chunks)
        total_vectors = 0
        start_time = time.time()
        last_save_time = start_time
        last_progress_time = start_time
        
        batch_texts = []
        batch_chunk_ids = []
        
        print(f"\n[prepare] 提取和拼接文本（向量化）...")
        
        # Step 1: 提取批量数据（使用numpy避免逐条迭代）
        doc_ids = df_chunks['doc_id'].values
        titles = df_chunks['title'].fillna('').values
        child_starts = df_chunks['child_start'].values
        child_ends = df_chunks['child_end'].values
        chunk_ids = df_chunks['chunk_id'].values
        
        # Step 2: 检查正吧性
        missing_docs = set(doc_ids) - set(self.doc_texts.keys())
        if missing_docs:
            print(f"[warn] 未找到 {len(missing_docs)} 个文档，将跳过")
        
        # 仅保留有效文档
        valid_mask = np.array([doc_id in self.doc_texts for doc_id in doc_ids])
        valid_indices = np.where(valid_mask)[0]
        
        doc_ids = doc_ids[valid_indices]
        titles = titles[valid_indices]
        child_starts = child_starts[valid_indices]
        child_ends = child_ends[valid_indices]
        chunk_ids = chunk_ids[valid_indices]
        
        print(f"  有效chunks: {len(chunk_ids):,}")
        
        # Step 3: 使用列表推导式批量提取文本（避免逐条dict查询）
        child_texts = [self.doc_texts[doc_id][start:end] 
                      for doc_id, start, end in zip(doc_ids, child_starts, child_ends)]
        
        # Step 4: 拼接标题和正文
        full_texts = [f"{title}\n\n{text}" if title else text
                     for title, text in zip(titles, child_texts)]
        
        # Step 5: 批量embedding
        print(f"\n[embed] 开始批量嵌入...\n")
        total_processed = len(self.processed_chunks)
        total_vectors = 0
        start_time = time.time()
        last_save_time = start_time
        last_progress_time = start_time
        
        # 批量处理（使用range不需要.iterrows()）
        for batch_start in range(0, len(full_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(full_texts))
            batch_texts = full_texts[batch_start:batch_end]
            batch_chunk_ids = chunk_ids[batch_start:batch_end]
            
            # 批量encoding
            embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize=True,
                show_progress=False
            )
            
            # 写入FAISS
            self.index.add(embeddings.astype(np.float32))
            
            # 记录chunk_id映射
            self.chunk_id_list.extend(batch_chunk_ids.tolist())
            self.processed_chunks.update(batch_chunk_ids)
            
            total_vectors += len(embeddings)
            total_processed += len(batch_chunk_ids)
            
            # 进度显示（每秒更新一次）
            current_time = time.time()
            if current_time - last_progress_time >= 1.0:
                elapsed = current_time - start_time
                speed = total_vectors / elapsed if elapsed > 0 else 0
                
                # 计算预计剩余时间
                if speed > 0 and total_processed < total_chunks:
                    remaining_vectors = (total_chunks - total_processed)
                    remaining_time = remaining_vectors / speed
                    remaining_str = self._format_time(remaining_time)
                else:
                    remaining_str = "计算中"
                
                elapsed_str = self._format_time(elapsed)
                progress_pct = (total_processed / total_chunks * 100) if total_chunks > 0 else 0
                
                time_str = time.strftime("%H:%M:%S")
                print(f"\r  [{time_str}] 进度: {progress_pct:5.1f}% ({total_processed:,}/{total_chunks:,}) | "
                      f"速度: {speed:6.0f} chunks/s | "
                      f"已耗时: {elapsed_str} | "
                      f"预计剩余: {remaining_str}", end='', flush=True)
                
                last_progress_time = current_time
                
            
            # 定期保存（每5分钟）
            if current_time - last_save_time > 300:
                print(f"\n[{time.strftime('%H:%M:%S')}] [save] 定期保存索引...")
                self._save_index()
                self._save_chunk_id_mapping()
                self._save_checkpoint()
                last_save_time = current_time
                print(f"[{time.strftime('%H:%M:%S')}] [save] 保存完成")
                
                # 强制垃圾回收
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 最终保存
        print(f"\n\n[save] 最终保存...")
        self._save_index()
        self._save_chunk_id_mapping()
        self._save_checkpoint()
        
        # 统计信息
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"构建完成！")
        print(f"  总chunks: {total_chunks:,}")
        print(f"  处理chunks: {total_processed:,}")
        print(f"  索引向量数: {self.index.ntotal:,}")
        print(f"  总耗时: {elapsed/60:.1f} 分钟")
        print(f"  平均速度: {total_vectors/elapsed:.0f} vec/s")
        print(f"{'='*80}")


def main(model_name: str = "qwen3"):
    """主函数
    
    Args:
        model_name: 模型名称（qwen3, bge_small, gte等）
    """
    
    # 加载配置
    cfg = Config()
    
    # 配置路径
    chunks_path = "data/processed/chunks.parquet"
    documents_path = "data/processed/documents_cleaned.parquet"
    
    # 根据模型名称获取配置（从 embedding.{model_name} 读取）
    model_config = cfg.get("embedding", model_name, default={})
    
    if not model_config:
        raise ValueError(f"未找到模型配置: embedding.{model_name}")
    
    # 初始化embedding模型参数
    model_id = model_config.get("model_id", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    device = model_config.get("device") or "cuda"  # None 时默认为 cuda
    max_length = model_config.get("max_length", 512)
    dtype = model_config.get("dtype", "float16")
    index_path = model_config.get("index_path", f"data/faiss/{model_name}_fp16_ip.faiss")
    batch_size = cfg.get("pipeline", "batch_size", default=1024)
    
    print(f"[config] 使用模型: {model_name}")
    print(f"  model_id: {model_id}")
    print(f"  device: {device}")
    print(f"  max_length: {max_length}")
    print(f"  dtype: {dtype}")
    print(f"  index_path: {index_path}")
    print(f"  batch_size: {batch_size}")
    
    embedding_model = Qwen3EmbeddingModel(
        model_id=model_id,
        device=device,
        max_length=max_length,
        dtype=dtype
    )
    
    # 构建索引
    builder = EmbeddingBuilderV2(
        chunks_path=chunks_path,
        documents_path=documents_path,
        embedding_model=embedding_model,
        index_path=index_path,
        batch_size=batch_size,  # 根据显存调整
        resume=True
    )
    
    builder.build()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建FAISS索引 V2")
    parser.add_argument("--model", type=str, default="qwen3", 
                       help="模型名称 (qwen3, bge_small, gte等)")
    args = parser.parse_args()
    
    main(model_name=args.model)
