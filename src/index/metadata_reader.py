"""
从chunks Parquet文件读取metadata

设计：
1. FAISS索引只存储向量和faiss_id
2. faiss_id -> chunk_id映射存储在chunk_id_map.json
3. chunk_id -> metadata从chunks Parquet读取（按需加载）
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pyarrow.parquet as pq


class MetadataReader:
    """从chunks Parquet读取metadata"""
    
    def __init__(
        self,
        parquet_dir: str,
        chunk_id_map_path: str,
    ):
        """
        Args:
            parquet_dir: chunks Parquet目录
            chunk_id_map_path: chunk_id映射文件路径（faiss_id -> chunk_id）
        """
        self.parquet_dir = parquet_dir
        self.chunk_id_map_path = chunk_id_map_path
        
        # 加载chunk_id映射
        self.chunk_id_map = self._load_chunk_id_map()
        
        # 缓存：chunk_id -> metadata（可选，按需启用）
        self.metadata_cache = {}
        
        # Parquet文件路径列表
        self.parquet_files = self._load_parquet_files()
    
    def _load_chunk_id_map(self) -> Dict[int, str]:
        """加载faiss_id -> chunk_id映射"""
        if not os.path.exists(self.chunk_id_map_path):
            raise FileNotFoundError(f"Chunk ID映射文件不存在: {self.chunk_id_map_path}")
        
        with open(self.chunk_id_map_path, 'r', encoding='utf-8') as f:
            # JSON key会被转为str，需要转回int
            raw_map = json.load(f)
            return {int(k): v for k, v in raw_map.items()}
    
    def _load_parquet_files(self) -> List[str]:
        """获取所有chunks Parquet文件"""
        parquet_dir = Path(self.parquet_dir)
        if not parquet_dir.exists():
            raise FileNotFoundError(f"Parquet目录不存在: {self.parquet_dir}")
        
        parquet_files = sorted(parquet_dir.glob("chunks_*.parquet"))
        return [str(p) for p in parquet_files]
    
    def get_chunk_id(self, faiss_id: int) -> Optional[str]:
        """根据faiss_id获取chunk_id
        
        Args:
            faiss_id: FAISS内部ID
        
        Returns:
            chunk_id或None（不存在）
        """
        return self.chunk_id_map.get(faiss_id)
    
    def get_metadata(self, chunk_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """根据chunk_id从Parquet读取metadata
        
        Args:
            chunk_id: Chunk ID
            use_cache: 是否使用缓存
        
        Returns:
            metadata字典，包含：
            - chunk_id: str
            - doc_id: str
            - rerank_text: str（已拼接标题）
            - child_ids: list[int]
            - parent_start: int
            - parent_end: int
            - chunk_len: int
        """
        # 检查缓存
        if use_cache and chunk_id in self.metadata_cache:
            return self.metadata_cache[chunk_id]
        
        # 遍历Parquet文件查找
        for parquet_path in self.parquet_files:
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            
            # 查找chunk_id
            rows = df[df['chunk_id'] == chunk_id]
            
            if not rows.empty:
                row = rows.iloc[0]
                metadata = {
                    'chunk_id': row['chunk_id'],
                    'doc_id': row['doc_id'],
                    'rerank_text': row['rerank_text'],
                    'child_ids': row['child_ids'].tolist() if hasattr(row['child_ids'], 'tolist') else list(row['child_ids']),
                    'parent_start': int(row['parent_start']),
                    'parent_end': int(row['parent_end']),
                    'chunk_len': int(row['chunk_len']),
                }
                
                # 缓存
                if use_cache:
                    self.metadata_cache[chunk_id] = metadata
                
                return metadata
        
        # 未找到
        return None
    
    def get_metadata_batch(self, chunk_ids: List[str], use_cache: bool = True) -> List[Optional[Dict[str, Any]]]:
        """批量获取metadata（优化性能）
        
        Args:
            chunk_ids: Chunk ID列表
            use_cache: 是否使用缓存
        
        Returns:
            metadata列表（保持顺序，未找到的为None）
        """
        results = []
        
        # 先从缓存获取
        uncached_ids = []
        for chunk_id in chunk_ids:
            if use_cache and chunk_id in self.metadata_cache:
                results.append(self.metadata_cache[chunk_id])
            else:
                results.append(None)
                uncached_ids.append(chunk_id)
        
        if not uncached_ids:
            return results
        
        # 批量读取Parquet
        uncached_set = set(uncached_ids)
        found_metadata = {}
        
        for parquet_path in self.parquet_files:
            if not uncached_set:
                break
            
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            
            # 筛选chunk_ids
            rows = df[df['chunk_id'].isin(uncached_set)]
            
            for _, row in rows.iterrows():
                chunk_id = row['chunk_id']
                metadata = {
                    'chunk_id': chunk_id,
                    'doc_id': row['doc_id'],
                    'rerank_text': row['rerank_text'],
                    'child_ids': row['child_ids'].tolist() if hasattr(row['child_ids'], 'tolist') else list(row['child_ids']),
                    'parent_start': int(row['parent_start']),
                    'parent_end': int(row['parent_end']),
                    'chunk_len': int(row['chunk_len']),
                }
                
                found_metadata[chunk_id] = metadata
                uncached_set.remove(chunk_id)
                
                # 缓存
                if use_cache:
                    self.metadata_cache[chunk_id] = metadata
        
        # 填充结果
        idx = 0
        for i, chunk_id in enumerate(chunk_ids):
            if results[i] is None:
                results[i] = found_metadata.get(chunk_id)
        
        return results
    
    def get_metadata_by_faiss_id(self, faiss_id: int, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """根据faiss_id获取metadata（便捷方法）
        
        Args:
            faiss_id: FAISS内部ID
            use_cache: 是否使用缓存
        
        Returns:
            metadata字典或None
        """
        chunk_id = self.get_chunk_id(faiss_id)
        if chunk_id is None:
            return None
        
        return self.get_metadata(chunk_id, use_cache=use_cache)
    
    def get_metadata_by_faiss_ids(self, faiss_ids: List[int], use_cache: bool = True) -> List[Optional[Dict[str, Any]]]:
        """根据faiss_ids批量获取metadata（便捷方法）
        
        Args:
            faiss_ids: FAISS内部ID列表
            use_cache: 是否使用缓存
        
        Returns:
            metadata列表（保持顺序）
        """
        # 转换faiss_id -> chunk_id
        chunk_ids = [self.get_chunk_id(fid) for fid in faiss_ids]
        
        # 批量获取metadata
        return self.get_metadata_batch(chunk_ids, use_cache=use_cache)
    
    def clear_cache(self):
        """清空缓存"""
        self.metadata_cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self.metadata_cache)


def test_metadata_reader():
    """测试MetadataReader"""
    print("=" * 80)
    print("测试MetadataReader")
    print("=" * 80)
    
    # 配置路径
    parquet_dir = "data/processed/parquet/chunks"
    chunk_id_map_path = "data/faiss/bge_small_chunk_id_map.json"
    
    if not os.path.exists(parquet_dir):
        print(f"❌ Parquet目录不存在: {parquet_dir}")
        return
    
    if not os.path.exists(chunk_id_map_path):
        print(f"❌ Chunk ID映射文件不存在: {chunk_id_map_path}")
        print("   请先运行 build_embeddings.py 生成索引")
        return
    
    # 创建reader
    reader = MetadataReader(
        parquet_dir=parquet_dir,
        chunk_id_map_path=chunk_id_map_path,
    )
    
    print(f"\n✅ 初始化成功")
    print(f"  Parquet文件数: {len(reader.parquet_files)}")
    print(f"  Chunk ID映射数: {len(reader.chunk_id_map):,}")
    
    # 测试：根据faiss_id获取chunk_id
    print(f"\n测试1: 根据faiss_id获取chunk_id")
    faiss_id = 0
    chunk_id = reader.get_chunk_id(faiss_id)
    print(f"  faiss_id={faiss_id} -> chunk_id={chunk_id}")
    
    # 测试：根据chunk_id获取metadata
    if chunk_id:
        print(f"\n测试2: 根据chunk_id获取metadata")
        metadata = reader.get_metadata(chunk_id)
        if metadata:
            print(f"  chunk_id: {metadata['chunk_id']}")
            print(f"  doc_id: {metadata['doc_id']}")
            print(f"  chunk_len: {metadata['chunk_len']}")
            print(f"  rerank_text: {metadata['rerank_text'][:80]}...")
        else:
            print(f"  ❌ 未找到metadata")
    
    # 测试：批量获取
    print(f"\n测试3: 批量获取metadata（faiss_ids=[0,1,2]）")
    metadata_list = reader.get_metadata_by_faiss_ids([0, 1, 2])
    for i, meta in enumerate(metadata_list):
        if meta:
            print(f"  [{i}] {meta['chunk_id']} | len={meta['chunk_len']}")
        else:
            print(f"  [{i}] ❌ 未找到")
    
    print(f"\n缓存大小: {reader.get_cache_size()}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_metadata_reader()
