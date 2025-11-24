"""检索融合与重排序 (Retrieval Fusion & Reranking)

实现RRF融合、MMR去重和cross-encoder reranking
"""
import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalFusion:
    """检索融合器 - 合并dense和BM25结果"""
    
    @staticmethod
    def paragraph_boosting(
        dense_results: List[Tuple[str, float]],
        bm25_results: List[Tuple],
        chunk_id_to_doc: Dict[str, str],
        alpha: float = 0.03
    ) -> Dict[str, float]:
        """段落指数增强
        
        同一文档的多个chunks出现时，使用指数增强来提升分数
        
        Args:
            dense_results: 密集检索结果 [(chunk_id, similarity), ...]
            bm25_results: BM25检索结果 [(chunk_id, score), ...] 或 [(chunk_idx, score), ...]
            chunk_id_to_doc: chunk_id到doc_id的映射
            alpha: 增强系数（0.02-0.04）
        
        Returns:
            {chunk_id: boosted_score, ...}
        """
        # 统计每个文档在结果中出现的次数
        doc_counts = defaultdict(int)
        chunk_scores = {}
        
        # 处理dense结果
        for chunk_id, score in dense_results:
            doc_id = chunk_id_to_doc.get(chunk_id, chunk_id)
            doc_counts[doc_id] += 1
            chunk_scores[chunk_id] = score
        
        # 应用指数增强
        boosted_scores = {}
        for chunk_id, base_score in chunk_scores.items():
            doc_id = chunk_id_to_doc.get(chunk_id, chunk_id)
            count = doc_counts[doc_id]
            
            # score = score * (1 + alpha * exp(n))
            boost_factor = 1 + alpha * math.exp(count)
            boosted_score = base_score * boost_factor
            
            boosted_scores[chunk_id] = boosted_score
        
        return boosted_scores
    
    @staticmethod
    def rrf_fusion(
        dense_results: List[Tuple[str, float]],
        bm25_results: List[Tuple],
        chunk_ids: List[str],
        rrf_k: int = 30
    ) -> List[Tuple[str, float]]:
        """RRF融合 (Reciprocal Rank Fusion)
        
        结合密集检索和BM25检索的排名
        
        Args:
            dense_results: 密集检索结果 [(chunk_id, similarity), ...]
            bm25_results: BM25检索结果 [(chunk_id, score), ...] 或 [(chunk_idx, score), ...]
            chunk_ids: 所有chunk_id列表 (用于从 chunk_idx 推导 chunk_id)
            rrf_k: RRF参数（越小越敏感排名，推荐30-50）
        
        Returns:
            融合后的结果 [(chunk_id, fused_score), ...]
        """
        rrf_scores = {}
        
        # 处理dense结果（rank越小权重越大）
        for rank, (chunk_id, _) in enumerate(dense_results, 1):
            rrf_score = 1.0 / (rrf_k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        
        # 处理BM25结果
        for rank, (chunk_id_or_idx, _) in enumerate(bm25_results, 1):
            # 新 BM25 rerank 返回 chunk_id (str)，旧格式返回 chunk_idx (int)
            # 简单判断：如果是 int，使用 chunk_ids 查表；如果是 str，直接使用
            if isinstance(chunk_id_or_idx, int):
                # 旧格式：需要从 chunk_ids 查表推导
                if chunk_id_or_idx < len(chunk_ids):
                    chunk_id = chunk_ids[chunk_id_or_idx]
                    rrf_score = 1.0 / (rrf_k + rank)
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
            else:
                # 新格式：直接使用 chunk_id
                chunk_id = chunk_id_or_idx
                rrf_score = 1.0 / (rrf_k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
        
        # 排序
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    @staticmethod
    def mmr_reranking(
        results: List[Tuple[str, float]],
        embeddings_dict: Dict[str, np.ndarray],
        lambda_param: float = 0.8,
        top_k: int = 400
    ) -> List[Tuple[str, float]]:
        """MMR重排序 (Maximal Marginal Relevance)
        
        平衡相关性和多样性，避免返回冗余chunks
        
        Args:
            results: 融合后的结果 [(chunk_id, score), ...]
            embeddings_dict: chunk_id到embedding的映射
            lambda_param: 相关性权重（0.8表示强调相关性）
            top_k: 最终返回的数量
        
        Returns:
            MMR重排序后的结果 [(chunk_id, mmr_score), ...]
        """
        if not results:
            return []
        
        mmr_results = []
        selected_set: Set[str] = set()
        remaining = list(results)
        
        # 贪心选择
        while remaining and len(mmr_results) < top_k:
            best_idx = 0
            best_score = float('-inf')
            
            for i, (chunk_id, relevance_score) in enumerate(remaining):
                # 如果是第一个，直接选择
                if not selected_set:
                    mmr_score = relevance_score
                else:
                    # 计算与已选chunks的平均相似度
                    if chunk_id in embeddings_dict:
                        chunk_emb = embeddings_dict[chunk_id].reshape(1, -1)
                        
                        max_similarity = 0
                        for selected_id in selected_set:
                            if selected_id in embeddings_dict:
                                selected_emb = embeddings_dict[selected_id].reshape(1, -1)
                                sim = cosine_similarity(chunk_emb, selected_emb)[0][0]
                                max_similarity = max(max_similarity, sim)
                        
                        # MMR = lambda * relevance - (1 - lambda) * max_similarity
                        mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
                    else:
                        mmr_score = relevance_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # 选择最佳chunk
            chunk_id, _ = remaining[best_idx]
            mmr_results.append((chunk_id, best_score))
            selected_set.add(chunk_id)
            
            # 移除已选择的
            remaining.pop(best_idx)
        
        return mmr_results
