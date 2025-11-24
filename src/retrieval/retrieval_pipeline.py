"""完整检索管道 (End-to-End Retrieval Pipeline)

从查询到最终结果的完整流程：
1. Dense Retrieval (top-600)
2. BM25 Retrieval (top-300)
3. Paragraph Boosting (同文档增强)
4. RRF Fusion (融合排序)
5. MMR Reranking (去重多样性)
6. Cross-Encoder Reranking (精准排序)
"""
import json
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from .dense_retrieval import DenseRetriever
    from .fusion import RetrievalFusion
    from .bm25 import BM25Retriever
    from ..rerank.reranker import CrossEncoderReranker
except ImportError:
    # 绝对导入方式（用于直接脚本运行）
    from retrieval.dense_retrieval import DenseRetriever
    from retrieval.fusion import RetrievalFusion
    from retrieval.bm25 import BM25Retriever
    from rerank.reranker import CrossEncoderReranker


class RetrievalPipeline:
    """完整检索管道"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
        chunk_ids_path: Optional[str] = None,
        chunks_parquet: Optional[str] = None,
        docs_parquet: Optional[str] = None,
        bm25_index_path: Optional[str] = None,
        reranker_model = None
    ):
        """初始化检索管道
        
        Args:
            config_path: config.yaml路径（如果提供，其他参数将从配置文件读取）
            index_path: FAISS索引路径
            chunk_ids_path: chunk_id映射路径
            chunks_parquet: chunks.parquet路径
            docs_parquet: documents_cleaned.parquet路径
            bm25_index_path: BM25索引路径（如果存在则加载，否则构建）
            reranker_model: 已加载的reranker模型
        """
        # 加载配置
        if config_path:
            self.config = self._load_config(config_path)
        else:
            # 默认配置路径
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / 'config.yaml'
            if config_file.exists():
                self.config = self._load_config(str(config_file))
            else:
                self.config = {}
        
        # 优先使用传入的参数，其次使用配置文件
        qwen3_config = self.config.get('embedding', {}).get('qwen3', {})
        
        # 初始化Dense检索器
        self.dense_retriever = DenseRetriever(
            index_path or qwen3_config.get('index_path', 'data/faiss/qwen3_fp16_ip.faiss'),
            chunk_ids_path or 'data/faiss/qwen3_fp16_ip_chunk_ids.json',
            model_id=qwen3_config.get('model_id', 'Qwen/Qwen3-Embedding-0.6B'),
            device=qwen3_config.get('device') or 'cuda',
            max_length=qwen3_config.get('max_length', 168),
            dtype=qwen3_config.get('dtype', 'float16')
        )
        
        # 从DenseRetriever获取tokenizer（避免重复加载）
        self.bm25_tokenizer = self.dense_retriever.tokenizer
        
        # 初始化Reranker
        self.reranker = CrossEncoderReranker(
            chunks_parquet or 'data/processed/chunks.parquet',
            docs_parquet or 'data/processed/documents_cleaned.parquet'
        )
        self.reranker_model = reranker_model
        
        # 加载chunks和docs
        self.chunks_df = pd.read_parquet(
            chunks_parquet or 'data/processed/chunks.parquet'
        )
        self.docs_df = pd.read_parquet(
            docs_parquet or 'data/processed/documents_cleaned.parquet'
        )
        
        # 建立映射关系
        self.chunk_ids_list = self.chunks_df['chunk_id'].tolist()
        self.chunk_id_to_doc = dict(zip(
            self.chunks_df['chunk_id'],
            self.chunks_df['doc_id']
        ))
        
        # 初始化BM25
        project_root = Path(__file__).parent.parent.parent
        bm25_config = self.config.get('retrieval', {}).get('bm25', {})
        bm25_index_path = bm25_config.get('index_path', str(project_root / 'data' / 'processed' / 'bm25_index'))
        
        if Path(bm25_index_path).exists():
            # 加载已有索引
            self.bm25 = BM25Retriever(index_path=bm25_index_path, config=bm25_config)
        else:
            # 构建新索引
            print(f"[BM25] 索引不存在，开始构建...")
            self.bm25 = BM25Retriever(
                chunks_parquet=chunks_parquet or 'data/processed/chunks.parquet',
                docs_parquet=docs_parquet or 'data/processed/documents_cleaned.parquet',
                tokenizer=self.bm25_tokenizer,
                config=bm25_config
            )
            # 保存索引
            self.bm25.save(bm25_index_path)
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def retrieve(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
        mmr_lambda: Optional[float] = None,
        mmr_top_k: Optional[int] = None,
        reranker_input_k: Optional[int] = None,
        reranker_top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """执行完整的检索管道
        
        Args:
            query: 查询文本
            dense_top_k: Dense检索的top-k (从配置文件读取，默认600)
            bm25_top_k: BM25检索的top-k (从配置文件读取，默认300)
            alpha: 段落增强系数 (从配置文件读取，默认0.03)
            rrf_k: RRF融合参数 (从配置文件读取，默认30)
            mmr_lambda: MMR相关性权重 (从配置文件读取，默认0.8)
            mmr_top_k: MMR返回数量 (从配置文件读取，默认500)
            reranker_top_k: 最终返回数量 (从配置文件读取，默认10)
            verbose: 是否打印详细日志
        
        Returns:
            [(chunk_id, score), ...] 的列表，最多reranker_top_k条
        """
        # 从配置文件读取默认参数
        retrieval_config = self.config.get('retrieval', {})
        
        dense_top_k = dense_top_k or retrieval_config.get('dense', {}).get('top_k', 600)
        bm25_top_k = bm25_top_k or retrieval_config.get('bm25', {}).get('top_k', 300)
        alpha = alpha or retrieval_config.get('paragraph_boosting', {}).get('alpha', 0.03)
        rrf_k = rrf_k or retrieval_config.get('fusion', {}).get('rrf_k', 30)
        mmr_lambda = mmr_lambda or retrieval_config.get('mmr', {}).get('lambda', 0.8)
        mmr_top_k = mmr_top_k or retrieval_config.get('mmr', {}).get('top_k', 500)
        reranker_input_k = reranker_input_k or self.config.get('reranker', {}).get('reranker_input_k', 100)
        reranker_top_k = reranker_top_k or self.config.get('reranker', {}).get('top_k', 5)
        
        if verbose:
            print("\n" + "="*80)
            print("检索管道执行")
            print("="*80)
        
        # Step 1: Dense Retrieval
        if verbose:
            print(f"\n[1] Dense Retrieval (top-{dense_top_k})...", end='', flush=True)
        
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_top_k)
        
        if verbose:
            print(f" ✅ 找到{len(dense_results)}条")
        
        # Step 2: BM25 Retrieval
        if verbose:
            print(f"\n[2] BM25 Retrieval (top-{bm25_top_k})...", end='', flush=True)
        
        try:
            # 使用BM25Retriever
            bm25_results = self.bm25.retrieve(
                query,
                tokenizer=self.bm25_tokenizer,
                top_k=bm25_top_k
            )
            
            if verbose:
                print(f" ✅ 找到{len(bm25_results)}条")
        except Exception as e:
            bm25_results = []
            if verbose:
                print(f" ❌ BM25不可用: {e}")
        
        # Step 3: Paragraph Boosting (可选)
        if verbose:
            print(f"\n[3] Paragraph Boosting (alpha={alpha})...", end='', flush=True)
        
        boosted_scores = RetrievalFusion.paragraph_boosting(
            dense_results, bm25_results, self.chunk_id_to_doc, alpha=alpha
        )
        
        # 转换为列表格式
        dense_results_boosted = [(cid, boosted_scores[cid]) for cid, _ in dense_results if cid in boosted_scores]
        
        if verbose:
            print(f" ✅")
        
        # Step 4: RRF Fusion
        if verbose:
            print(f"\n[4] RRF Fusion (k={rrf_k})...", end='', flush=True)
        
        fused_results = RetrievalFusion.rrf_fusion(
            dense_results_boosted, bm25_results, self.chunk_ids_list, rrf_k=rrf_k
        )
        
        if verbose:
            print(f" ✅ 融合后{len(fused_results)}条")
        
        # Step 5: MMR Reranking
        if verbose:
            print(f"\n[5] MMR Reranking (lambda={mmr_lambda}, top-{mmr_top_k})...", end='', flush=True)
        
        # 为MMR准备embeddings（仅用于多样性计算，可简化）
        mmr_results = RetrievalFusion.mmr_reranking(
            fused_results, {}, lambda_param=mmr_lambda, top_k=mmr_top_k
        )
        # 🚨 必须截断：只给 reranker 输入 20 条
        mmr_results = mmr_results[:reranker_input_k]
        if verbose:
            print(f" ✅ 去重后{len(mmr_results)}条")
        
        # Step 6: Cross-Encoder Reranking
        if self.reranker_model is not None:
            if verbose:
                print(f"\n[6] Cross-Encoder Reranking (top-{reranker_top_k})...", end='', flush=True)
            
            mmr_chunk_ids = [cid for cid, _ in mmr_results]
            final_results = self.reranker.rerank(
                query, mmr_chunk_ids, self.reranker_model, top_k=reranker_top_k
            )
            
            if verbose:
                print(f" ✅ 最终{len(final_results)}条")
        else:
            # 没有 reranker，直接返回 MMR 结果
            if verbose:
                print(f"\n[6] Cross-Encoder Reranking: ⏭️  跳过（未加载reranker）")
            final_results = [(cid, score) for cid, score in mmr_results[:reranker_top_k]]
        
        if verbose:
            print("="*80 + "\n")
        
        return final_results
