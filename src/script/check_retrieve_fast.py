"""快速检索测试脚本 - 使用HF镜像加速模型下载

如果还是卡住，可以尝试以下解决方案：
1. 手动预下载模型
2. 使用代理或镜像
3. 直接使用已缓存的模型
"""
import sys
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# 设置HuggingFace镜像（加速下载）
# 尝试多个镜像源
MIRROR_SOURCES = [
    "https://huggingface-mirror.com",  # 国内镜像1
    "https://hf-mirror.com",           # 国内镜像2
    "https://huggingface.co",          # 官方源
]

# 尝试设置镜像
for mirror in MIRROR_SOURCES[:2]:
    os.environ['HF_ENDPOINT'] = mirror
    print(f"[mirror] 尝试使用镜像: {mirror}")
    break

try:
    from retrieval.embedding_qwen import Qwen3EmbeddingModel
except ImportError:
    print("❌ 无法导入Qwen3EmbeddingModel")
    exit(1)


class FastRetrievalChecker:
    """快速检索检验器"""
    
    def __init__(self):
        self.chunks_path = project_root / 'data' / 'processed' / 'chunks.parquet'
        self.docs_path = project_root / 'data' / 'processed' / 'documents_cleaned.parquet'
        
        self.df_chunks = None
        self.df_docs = None
        self.embedding_model = None
        
    def load_data(self):
        """加载数据"""
        print("\n[1] 加载数据文件...")
        
        try:
            print("  📥 加载 chunks.parquet...", end='', flush=True)
            self.df_chunks = pd.read_parquet(self.chunks_path)
            print(f" ✅ ({len(self.df_chunks):,}条)")
            
            print("  📥 加载 documents_cleaned.parquet...", end='', flush=True)
            self.df_docs = pd.read_parquet(self.docs_path)
            print(f" ✅ ({len(self.df_docs):,}条)")
            
        except Exception as e:
            print(f"\n  ❌ 加载失败: {e}")
            return False
        
        return True
    
    def init_embedding_model(self):
        """初始化embedding模型"""
        print("\n[2] 初始化Qwen3 Embedding模型...")
        print("   ⏳ 这可能需要几分钟来下载模型（约3GB）...")
        
        try:
            self.embedding_model = Qwen3EmbeddingModel(
                model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                device="cuda",
                dtype="float16"
            )
            print("  ✅ 模型加载成功")
            return True
        except Exception as e:
            print(f"\n  ❌ 模型加载失败: {type(e).__name__}: {e}")
            print(f"\n  💡 解决方案：")
            print(f"     方案A - 更换镜像源（快速）：")
            print(f"       export HF_ENDPOINT=https://hf-mirror.com")
            print(f"       python src/script/check_retrieve_fast.py")
            print(f"")
            print(f"     方案B - 手动预下载模型（推荐）：")
            print(f"       huggingface-cli download Alibaba-NLP/gte-Qwen2-1.5B-instruct --resume-download")
            print(f"")
            print(f"     方案C - 检查本地缓存：")
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            print(f"       缓存目录: {cache_dir}")
            if cache_dir.exists():
                print(f"       已有模型: {list(cache_dir.glob('models--*'))}")
            return False
    
    def get_chunk_text(self, chunk_id: str) -> str:
        """获取chunk文本"""
        try:
            chunk_row = self.df_chunks[self.df_chunks['chunk_id'] == chunk_id].iloc[0]
            doc_id = chunk_row['doc_id']
            doc_row = self.df_docs[self.df_docs['doc_id'] == doc_id]
            
            if len(doc_row) == 0:
                return f"[未找到doc_id: {doc_id}]"
            
            doc_text = doc_row.iloc[0]['text']
            title = chunk_row['title']
            child_start = chunk_row['child_start']
            child_end = chunk_row['child_end']
            chunk_text = doc_text[child_start:child_end]
            
            return f"[{title}]\n{chunk_text}"
        
        except Exception as e:
            return f"[错误: {e}]"
    
    def get_doc_chunks(self, doc_id: str) -> List[str]:
        """获取该文档的所有chunk_id"""
        return self.df_chunks[self.df_chunks['doc_id'] == doc_id]['chunk_id'].tolist()
    
    def search_similar_in_doc(self, query_embedding: np.ndarray, chunk_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """在给定的chunk中搜索相似的"""
        try:
            # 对所有chunk进行embedding
            chunk_texts = [self.get_chunk_text(cid).split('\n', 1)[1] if '\n' in self.get_chunk_text(cid) else self.get_chunk_text(cid) 
                          for cid in chunk_ids]
            chunk_embeddings = self.embedding_model.encode(chunk_texts, batch_size=32)
            
            # 计算相似度
            similarities = cosine_similarity(query_embedding.reshape(1, -1), chunk_embeddings)[0]
            
            # 排序
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in sorted_indices:
                results.append((chunk_ids[idx], float(similarities[idx])))
            
            return results
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def run_test(self, num_samples: int = 5):
        """运行测试"""
        print("\n" + "=" * 100)
        print("快速检索测试 - 同文档内检索")
        print("=" * 100)
        
        if not self.load_data():
            return
        
        if not self.init_embedding_model():
            return
        
        print(f"\n[3] 随机采样 {num_samples} 个chunks 进行检索测试...\n")
        
        # 随机采样
        sample_indices = random.sample(range(len(self.df_chunks)), num_samples)
        
        for test_idx, chunk_idx in enumerate(sample_indices, 1):
            chunk_row = self.df_chunks.iloc[chunk_idx]
            chunk_id = chunk_row['chunk_id']
            doc_id = chunk_row['doc_id']
            
            print(f"\n{'='*100}")
            print(f"测试 {test_idx}/{num_samples}: {chunk_id}")
            print(f"{'='*100}")
            
            # 获取chunk文本
            chunk_text = self.get_chunk_text(chunk_id)
            text_only = chunk_text.split('\n', 1)[1] if '\n' in chunk_text else chunk_text
            
            print(f"\n📄 查询Chunk:")
            print("-" * 100)
            print(text_only[:300] + ("..." if len(text_only) > 300 else ""))
            print("-" * 100)
            
            # Embedding该chunk
            print(f"\n🔄 对查询chunk进行embedding...", end='', flush=True)
            query_embedding = self.embedding_model.encode([text_only], batch_size=1)[0]
            print(" ✅")
            
            # 获取同文档的其他chunks
            all_doc_chunks = self.get_doc_chunks(doc_id)
            print(f"\n📊 该文档共有 {len(all_doc_chunks)} 个chunks")
            
            # 搜索
            print(f"\n🔍 在同文档内搜索Top5相似chunks...\n")
            results = self.search_similar_in_doc(query_embedding, all_doc_chunks, top_k=5)
            
            # 显示结果
            for rank, (result_chunk_id, similarity) in enumerate(results, 1):
                is_self = "✅ [本身]" if result_chunk_id == chunk_id else ""
                result_text = self.get_chunk_text(result_chunk_id)
                text_only_result = result_text.split('\n', 1)[1] if '\n' in result_text else result_text
                
                print(f"\n【Top {rank}】相似度: {similarity:.4f} {is_self}")
                print(f"ID: {result_chunk_id}")
                print(f"内容: {text_only_result[:200]}...")
                print("-" * 100)
        
        print(f"\n\n✅ 测试完成！\n")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("HuggingFace 检索测试工具")
    print("=" * 100)
    print(f"\n🔗 当前使用镜像: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"💾 缓存目录: {Path.home() / '.cache' / 'huggingface'}")
    
    checker = FastRetrievalChecker()
    checker.run_test(num_samples=5)
