"""检索测试脚本：验证embedding索引的检索效果

流程：
1. 从chunks.parquet随机读取5个chunk
2. 从documents_cleaned.parquet获取这些chunk的原始文本
3. 使用Qwen3 embedding进行向量化
4. 搜索FAISS索引获取top5相似chunks
5. 对比结果，验证检索准确性
"""
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import faiss

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from retrieval.embedding_qwen import Qwen3EmbeddingModel
except ImportError:
    print("❌ 无法导入Qwen3EmbeddingModel，请确保模块正确安装")
    exit(1)


class RetrievalChecker:
    """检索检验器"""
    
    def __init__(self):
        self.chunks_path = project_root / 'data' / 'processed' / 'chunks.parquet'
        self.docs_path = project_root / 'data' / 'processed' / 'documents_cleaned.parquet'
        self.index_path = project_root / 'data' / 'faiss' / 'qwen3_fp16_ip.faiss'
        self.chunk_ids_path = project_root / 'data' / 'faiss' / 'qwen3_fp16_ip_chunk_ids.json'
        
        self.df_chunks = None
        self.df_docs = None
        self.index = None
        self.chunk_ids = None
        self.embedding_model = None
        
    def load_data(self):
        """加载所有需要的数据"""
        print("\n[1] 加载数据文件...")
        
        try:
            # 加载chunks
            print("  📥 加载 chunks.parquet...", end='', flush=True)
            self.df_chunks = pd.read_parquet(self.chunks_path)
            print(f" ✅ ({len(self.df_chunks):,}条)")
            
            # 加载文档
            print("  📥 加载 documents_cleaned.parquet...", end='', flush=True)
            self.df_docs = pd.read_parquet(self.docs_path)
            print(f" ✅ ({len(self.df_docs):,}条)")
            
            # 检查FAISS索引
            print("  📥 加载 FAISS索引...", end='', flush=True)
            
            # 如果.faiss不存在但.lz4存在，先解压
            if not self.index_path.exists() and Path(str(self.index_path) + '.lz4').exists():
                print("\n     正在解压LZ4文件...", end='', flush=True)
                import lz4.frame
                lz4_path = str(self.index_path) + '.lz4'
                with lz4.frame.open(lz4_path, 'rb') as f_in:
                    data = f_in.read()
                with open(self.index_path, 'wb') as f_out:
                    f_out.write(data)
                print(" 完成")
                print("  📥 加载 FAISS索引...", end='', flush=True)
            
            self.index = faiss.read_index(str(self.index_path))
            print(f" ✅ (向量数: {self.index.ntotal:,})")
            
            # 加载chunk_id映射
            print("  📥 加载 chunk_id映射...", end='', flush=True)
            with open(self.chunk_ids_path, 'r', encoding='utf-8') as f:
                self.chunk_ids = json.load(f)
            print(f" ✅ ({len(self.chunk_ids):,}条)")
            
        except Exception as e:
            print(f"\n  ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def init_embedding_model(self):
        """初始化embedding模型"""
        print("\n[2] 初始化Qwen3 Embedding模型...")
        try:
            self.embedding_model = Qwen3EmbeddingModel(
                model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                device="cuda",
                dtype="float16"
            )
            print("  ✅ 模型加载成功")
            return True
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            return False
    
    def get_chunk_text(self, chunk_id: str) -> str:
        """根据chunk_id获取chunk的完整文本
        
        Args:
            chunk_id: chunk标识，格式 "file:row:N:chunk:M"
        
        Returns:
            chunk的文本内容
        """
        try:
            # 找到该chunk在chunks.parquet中的行
            chunk_row = self.df_chunks[self.df_chunks['chunk_id'] == chunk_id].iloc[0]
            
            # 获取doc_id
            doc_id = chunk_row['doc_id']
            
            # 从documents_cleaned找到对应的文档
            doc_row = self.df_docs[self.df_docs['doc_id'] == doc_id]
            if len(doc_row) == 0:
                return f"[未找到doc_id: {doc_id}]"
            
            doc_text = doc_row.iloc[0]['text']
            title = chunk_row['title']
            
            # 提取chunk对应的文本
            child_start = chunk_row['child_start']
            child_end = chunk_row['child_end']
            
            chunk_text = doc_text[child_start:child_end]
            
            return f"[{title}]\n{chunk_text}"
        
        except Exception as e:
            return f"[错误: {e}]"
    
    def search_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """搜索相似chunks
        
        Args:
            query_text: 查询文本
            top_k: 返回top_k个结果
        
        Returns:
            [(chunk_id, distance, faiss_idx), ...] 的列表
        """
        try:
            # Embedding查询文本
            query_embedding = self.embedding_model.encode([query_text], batch_size=1)[0]
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # 搜索
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                chunk_id = self.chunk_ids[int(idx)]
                results.append((chunk_id, float(dist), int(idx)))
            
            return results
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def run_test(self, num_samples: int = 5):
        """运行检索测试
        
        Args:
            num_samples: 测试样本数
        """
        print("\n" + "=" * 100)
        print("检索测试")
        print("=" * 100)
        
        # 加载数据
        if not self.load_data():
            return
        
        # 初始化模型
        if not self.init_embedding_model():
            return
        
        print(f"\n[3] 随机采样 {num_samples} 个chunks 进行检索测试...\n")
        
        # 随机采样chunk
        sample_indices = random.sample(range(len(self.df_chunks)), num_samples)
        
        for test_idx, chunk_idx in enumerate(sample_indices, 1):
            chunk_row = self.df_chunks.iloc[chunk_idx]
            chunk_id = chunk_row['chunk_id']
            
            print(f"\n{'='*100}")
            print(f"测试 {test_idx}/{num_samples}: {chunk_id}")
            print(f"{'='*100}")
            
            # 获取该chunk的文本
            chunk_text = self.get_chunk_text(chunk_id)
            print(f"\n📄 查询Chunk内容:")
            print("-" * 100)
            print(chunk_text[:500] + ("..." if len(chunk_text) > 500 else ""))
            print("-" * 100)
            
            # 搜索相似chunks
            print(f"\n🔍 检索Top5相似chunks...\n")
            results = self.search_similar(chunk_text.split('\n', 1)[1] if '\n' in chunk_text else chunk_text, top_k=5)
            
            # 显示结果
            for rank, (result_chunk_id, distance, faiss_idx) in enumerate(results, 1):
                result_text = self.get_chunk_text(result_chunk_id)
                
                # 判断是否为自己
                is_self = "✅ [本身]" if result_chunk_id == chunk_id else ""
                
                # 判断是否同doc
                query_doc_id = chunk_row['doc_id']
                result_doc_id = self.df_chunks[self.df_chunks['chunk_id'] == result_chunk_id].iloc[0]['doc_id']
                is_same_doc = "✅ [同文档]" if result_doc_id == query_doc_id else "⚠️  [不同文档]"
                
                print(f"\n【Top {rank}】距离: {distance:.4f} {is_self} {is_same_doc}")
                print(f"ID: {result_chunk_id}")
                print(f"内容: {result_text[:300]}...")
                print("-" * 100)
        
        print(f"\n\n✅ 测试完成！\n")


if __name__ == "__main__":
    checker = RetrievalChecker()
    checker.run_test(num_samples=5)
