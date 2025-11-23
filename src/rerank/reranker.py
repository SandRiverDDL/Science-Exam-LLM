"""Cross-Encoder Reranker (交叉编码器重排序)

使用Jina Reranker进行最终的精准重排序
"""
from typing import List, Tuple
import pandas as pd


class CrossEncoderReranker:
    """交叉编码器重排序器"""
    
    def __init__(self, chunks_parquet: str, docs_parquet: str):
        """初始化Reranker
        
        Args:
            chunks_parquet: chunks.parquet文件路径
            docs_parquet: documents_cleaned.parquet文件路径
        """
        self.chunks_df = pd.read_parquet(chunks_parquet)
        self.docs_df = pd.read_parquet(docs_parquet)
        
        # 建立快速查询索引
        self.chunk_to_doc = dict(zip(self.chunks_df['chunk_id'], self.chunks_df['doc_id']))
        self.doc_texts = dict(zip(self.docs_df['doc_id'], self.docs_df['text']))
    
    def get_chunk_text(self, chunk_id: str) -> str:
        """获取chunk的完整文本"""
        try:
            chunk_row = self.chunks_df[self.chunks_df['chunk_id'] == chunk_id].iloc[0]
            doc_id = chunk_row['doc_id']
            doc_text = self.doc_texts.get(doc_id, "")
            
            # 提取子chunk文本
            child_start = chunk_row['child_start']
            child_end = chunk_row['child_end']
            chunk_text = doc_text[child_start:child_end]
            
            return chunk_text
        except Exception as e:
            return ""
    
    def get_parent_chunk_text(self, chunk_id: str) -> str:
        """获取chunk的父chunk（512 tokens上下文）"""
        try:
            chunk_row = self.chunks_df[self.chunks_df['chunk_id'] == chunk_id].iloc[0]
            doc_id = chunk_row['doc_id']
            doc_text = self.doc_texts.get(doc_id, "")
            title = chunk_row['title']
            
            # 提取父chunk文本
            parent_start = chunk_row['parent_start']
            parent_end = chunk_row['parent_end']
            parent_text = doc_text[parent_start:parent_end]
            
            # 拼接标题和父chunk
            if title:
                return f"{title}\n\n{parent_text}"
            else:
                return parent_text
        except Exception as e:
            return ""
    
    def rerank(
        self,
        query: str,
        chunk_ids: List[str],
        reranker_model,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """使用Cross-Encoder重排序
        
        Args:
            query: 查询文本
            chunk_ids: 待重排序的chunk_id列表
            reranker_model: 已加载的reranker模型
            top_k: 返回top_k个结果
        
        Returns:
            重排序后的结果 [(chunk_id, score), ...]
        """
        # 准备文本对
        query_chunk_pairs = []
        valid_chunk_ids = []
        
        for chunk_id in chunk_ids:
            # 获取父chunk作为上下文
            chunk_text = self.get_parent_chunk_text(chunk_id)
            
            if chunk_text:
                query_chunk_pairs.append([query, chunk_text])
                valid_chunk_ids.append(chunk_id)
        
        if not query_chunk_pairs:
            return []
        
        # 使用reranker计算分数
        scores = reranker_model.rank(query_chunk_pairs)
        
        # 排序并返回top_k
        ranked = sorted(
            zip(valid_chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return ranked
