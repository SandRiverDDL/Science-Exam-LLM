"""完整检索管道 (End-to-End Retrieval Pipeline - 优化版)

从查询到最终结果的完整流程（高性能）：
1. Dense Retrieval via FAISS on-disk
2. BM25 Retrieval via 倒排表（term posting）
3. Paragraph Boosting + RRF Fusion + MMR
4. DuckDB 按需读取 parquet 文本
5. Cross-Encoder Reranking（仅对top-20）

特点：
- 一次性加载模型（embedding, reranker），整个过程不卸载
- FAISS on-disk，mmap 不加载全索引
- BM25 倒排表 pkl 格式，mmap 加载
- DuckDB 按需查询文本，不常驻内存
"""
import json
import yaml
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

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
    """完整检索管道（高性能优化版）
    
    一次性加载所有模型，整个 RAG 过程中不卸载。
    使用 FAISS on-disk, BM25 mmap, DuckDB 按需查询等技术。
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        reranker_model=None,
        verbose: bool = False
    ):
        """初始化检索管道（一次性加载模型）
        
        Args:
            config_path: config.yaml 路径
            reranker_model: 已加载的 reranker 模型（若无则跳过 reranker）
            verbose: 是否打印加载日志
        """
        # ===== 加载配置 =====
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
        
        if verbose:
            print("[初始化] 加载检索管道...")
        
        # ===== Step 1: 一次性加载 Embedding 模型 =====
        if verbose:
            print("  [1/3] 加载 Embedding 模型...", end='', flush=True)
        
        # embedding.qwen3: 用于构建 chunks 索引的模型（FAISS 索引）
        # retrieval.query_encoder: 用于查询编码的模型（独立的 query encoder）
        # 注意：两个模型逻辑上独立，未来可能使用不同的模型
        embedding_config = self.config.get('embedding', {}).get('qwen3', {})
        query_encoder_config = self.config.get('retrieval', {}).get('query_encoder', {})
        
        # query_encoder 参数优先级：query_encoder > embedding > 默认值
        # 但 index_path 总是来自 embedding.qwen3（因为索引是用 embedding 模型生成的）
        model_id = query_encoder_config.get('model_id', embedding_config.get('model_id', 'Qwen/Qwen3-Embedding-0.6B'))
        max_length = query_encoder_config.get('max_length', embedding_config.get('max_length', 168))
        device = query_encoder_config.get('device') or embedding_config.get('device') or 'cuda'
        dtype = query_encoder_config.get('dtype', embedding_config.get('dtype', 'float16'))
        # index_path 来自 embedding.qwen3（这是用 embedding 模型生成的索引）
        index_path = embedding_config.get('index_path', 'data/faiss/qwen3_fp16_ip.faiss')
        
        self.dense_retriever = DenseRetriever(
            index_path=index_path,
            chunk_ids_path='data/faiss/qwen3_fp16_ip_chunk_ids.json',
            model_id=model_id,
            device=device,
            max_length=max_length,
            dtype=dtype
        )
        
        # 缓存 tokenizer（供 BM25 复用）
        self.bm25_tokenizer = self.dense_retriever.tokenizer
        
        if verbose:
            print(" ✅")
        
        # ===== Step 2: 初始化 BM25 检索器 =====
        if verbose:
            print("  [2/3] 初始化 BM25...", end='', flush=True)
        
        project_root = Path(__file__).parent.parent.parent
        bm25_config = self.config.get('retrieval', {}).get('bm25', {})
        bm25_index_path = bm25_config.get('index_path', str(project_root / 'data' / 'processed' / 'bm25_index'))
        
        if Path(bm25_index_path).exists():
            # 加载已有索引
            self.bm25 = BM25Retriever(index_path=bm25_index_path, config=bm25_config)
        else:
            # 构建新索引
            if verbose:
                print("\n    索引不存在，构建中...", end='', flush=True)
            
            self.bm25 = BM25Retriever(
                chunks_parquet='data/processed/chunks.parquet',
                docs_parquet='data/processed/documents_cleaned.parquet',
                tokenizer=self.bm25_tokenizer,
                config=bm25_config
            )
            # 保存索引
            self.bm25.save(bm25_index_path)
        
        if verbose:
            print(" ✅")
        
        # ===== Step 3: 初始化 Reranker =====
        if verbose:
            print("  [3/3] 初始化 Reranker...", end='', flush=True)
        
        if verbose:
            print("\n    加载 chunks 和 docs parquet...", end='', flush=True)
        self.reranker = CrossEncoderReranker(
            chunks_parquet='data/processed/chunks.parquet',
            docs_parquet='data/processed/documents_cleaned.parquet'
        )
        if verbose:
            print(" ✅")
        self.reranker_model = reranker_model
        
        if verbose:
            print(" ✅")
        
        # ===== 缓存元数据 =====
        if verbose:
            print("  缓存元数据...", end='', flush=True)
        self.chunks_df = pd.read_parquet('data/processed/chunks.parquet')
        self.docs_df = pd.read_parquet('data/processed/documents_cleaned.parquet')
        if verbose:
            print(f" ✅ ({len(self.chunks_df)} chunks)")
        
        # 建立映射关系
        if verbose:
            print("  建立映射关系...", end='', flush=True)
        self.chunk_ids_list = self.chunks_df['chunk_id'].tolist()
        self.chunk_id_to_doc = dict(zip(
            self.chunks_df['chunk_id'],
            self.chunks_df['doc_id']
        ))
        if verbose:
            print(" ✅")
        
        if verbose:
            print("\n[初始化完成] ✅\n")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def retrieve(
        self,
        queries: Union[str, List[str]],
        verbose: bool = False
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """执行完整的检索管道（高性能版，支持 batch 查询）
        
        所有参数从 config.yaml 读取，确保一致性和可维护性。
        支持单个查询或 batch 查询，各步骤分别计时。
        
        Args:
            queries: 单个查询字符串或查询列表
            verbose: 是否打印详细日志和计时信息
        
        Returns:
            单个查询时返回 [(chunk_id, score), ...] 的列表
            Batch 查询时返回 [[(chunk_id, score), ...], ...] 的列表
        """
        # 标准化输入：统一转换为列表形式
        if isinstance(queries, str):
            queries = [queries]
            is_single_query = True
        else:
            is_single_query = False
        
        num_queries = len(queries)
        
        # 从配置文件读取所有参数
        retrieval_config = self.config.get('retrieval', {})
        reranker_config = retrieval_config.get('reranker', {})
        
        # Embedding batch_size
        query_encoder_config = retrieval_config.get('query_encoder', {})
        embedding_batch_size = query_encoder_config.get('batch_size', 32)
        
        # Dense batch_size
        dense_config = retrieval_config.get('dense', {})
        dense_batch_size = dense_config.get('batch_size', 64)
        
        dense_top_k = retrieval_config.get('dense', {}).get('top_k', 600)
        # bm25_top_k 不再需要，BM25 现在只对 Dense 的 top-K 进行重打分
        alpha = retrieval_config.get('paragraph_boosting', {}).get('alpha', 0.03)
        rrf_k = retrieval_config.get('fusion', {}).get('rrf_k', 30)
        mmr_lambda = retrieval_config.get('mmr', {}).get('lambda', 0.8)
        mmr_top_k = retrieval_config.get('mmr', {}).get('top_k', 500)

        reranker_top_k = reranker_config.get('top_k', 5)
        
        if verbose:
            print("\n" + "="*70)
            print(f"检索管道执行 (batch_size={num_queries})")
            print("="*70)
        
        # ===== Step 1: Dense Retrieval (FAISS on-disk) =====
        if verbose:
            print(f"\n[1] Dense Retrieval (top-{dense_top_k})...", end='', flush=True)
        
        time_dense_start = time.time()
        with torch.no_grad():
            dense_results_list = []
            # 二次分batch：先为pipeline batch，然后embedding batch
            for query_batch_start in range(0, num_queries, embedding_batch_size):
                query_batch_end = min(query_batch_start + embedding_batch_size, num_queries)
                query_batch = queries[query_batch_start:query_batch_end]
                
                # embedding一次性处理一个embedding batch
                embeddings = self.dense_retriever.embedding_model.encode(
                    query_batch,
                    batch_size=embedding_batch_size
                )
                
                # FAISS查询（支持批量，提供高效）
                embeddings_fp32 = embeddings.astype(np.float32)
                distances, indices = self.dense_retriever.index.search(embeddings_fp32, dense_top_k)
                
                # 转换为chunk_id和距离
                for dist, idx in zip(distances, indices):
                    results = []
                    for d, i in zip(dist, idx):
                        chunk_id = self.dense_retriever.chunk_ids[int(i)]
                        similarity = float(d)
                        results.append((chunk_id, similarity))
                    dense_results_list.append(results)
        
        time_dense = time.time() - time_dense_start
        
        if verbose:
            print(f" ✅ 找到 {sum(len(r) for r in dense_results_list)} 条 (耗时 {time_dense:.2f}s, {num_queries} queries)")
        
        # ===== Step 2: BM25 Reranking (仅对 Dense top-K 重打分) =====
        if verbose:
            print(f"[2] BM25 Reranking (rerank on top-{dense_top_k})...", end='', flush=True)
        
        time_bm25_start = time.time()
        bm25_results_list = []
        try:
            for dense_results in dense_results_list:
                # Dense 结果已经是 [(chunk_id, score), ...] 的格式
                # 需要为Each chunk_id 求取对应的 text
                candidates = []
                for chunk_id, score in dense_results:
                    # 从 reranker 中获取 chunk 文本
                    chunk_text = self.reranker.get_chunk_text(chunk_id)
                    if chunk_text:
                        candidates.append({
                            'chunk_id': chunk_id,
                            'text': chunk_text,
                            'score': score
                        })
                
                # 用 BM25 对 Dense 的 top-K 进行重打分
                if candidates:
                    bm25_reranked = self.bm25.rerank(
                        queries[len(bm25_results_list)],
                        candidates,
                        tokenizer=self.bm25_tokenizer
                    )
                    # rerank 结果已经是 [(chunk_id, score), ...] 格式
                    bm25_results = bm25_reranked
                else:
                    bm25_results = []
                
                bm25_results_list.append(bm25_results)
            
            time_bm25 = time.time() - time_bm25_start
            if verbose:
                print(f" ✅ 重打分 {sum(len(r) for r in bm25_results_list)} 条 (耗时 {time_bm25:.2f}s, {num_queries} queries)")
        except Exception as e:
            time_bm25 = time.time() - time_bm25_start
            bm25_results_list = [[] for _ in queries]
            if verbose:
                print(f" ❌ BM25 不可用: {e} (耗时 {time_bm25:.2f}s)")
        
        # ===== Step 3-5: 融合 (BM25 Rerank + RRF + Paragraph Boosting + MMR) =====
        if verbose:
            print(f"[3] RRF Fusion (rrf_k={rrf_k})...", end='', flush=True)
        
        time_fusion_start = time.time()
        time_rrf_start = time.time()
        
        # RRF Fusion
        rrf_results_list = []
        for dense_results, bm25_results in zip(dense_results_list, bm25_results_list):
            # RRF Fusion: 聚合 Dense 和 BM25 的结果
            fused_results = RetrievalFusion.rrf_fusion(
                dense_results, bm25_results, self.chunk_ids_list, rrf_k=rrf_k
            )
            rrf_results_list.append(fused_results)
        
        time_rrf = time.time() - time_rrf_start
        
        if verbose:
            print(f" ✅")
            print(f"    RRF 后 {sum(len(r) for r in rrf_results_list)} 条 (耗时 {time_rrf:.2f}s, {num_queries} queries)")
        
        # Paragraph Boosting: RRF 融合后、MMR 之前
        if verbose:
            print(f"[3.5] Paragraph Boosting (alpha={alpha})...", end='', flush=True)
        
        time_pb_start = time.time()
        pb_results_list = []
        
        for rrf_results in rrf_results_list:
            # 使用 RRF 结果进行 Paragraph Boosting
            boosted_scores = RetrievalFusion.paragraph_boosting(
                rrf_results, [], self.chunk_id_to_doc, alpha=alpha
            )
            # 应用 boosted scores
            pb_results = [(cid, boosted_scores.get(cid, score)) for cid, score in rrf_results]
            # 按 boosted score 排序
            pb_results = sorted(pb_results, key=lambda x: x[1], reverse=True)
            pb_results_list.append(pb_results)
        
        time_pb = time.time() - time_pb_start
        
        if verbose:
            print(f" ✅")
            print(f"    Boosting 后 {sum(len(r) for r in pb_results_list)} 条 (耗时 {time_pb:.2f}s, {num_queries} queries)")
        
        # MMR Reranking
        if verbose:
            print(f"[3.6] MMR Reranking (lambda={mmr_lambda})...", end='', flush=True)
        
        time_mmr_start = time.time()
        fused_results_list = []
        
        for pb_results in pb_results_list:
            # MMR Reranking
            mmr_results = RetrievalFusion.mmr_reranking(
                pb_results, {}, lambda_param=mmr_lambda, top_k=mmr_top_k
            )
            # 截断：仅输入 top-20 给 reranker（关键性能优化）
            fused_results_list.append(mmr_results)
        
        time_mmr = time.time() - time_mmr_start
        
        if verbose:
            print(f" ✅")
            print(f"    MMR 后 {sum(len(r) for r in fused_results_list)} 条 (耗时 {time_mmr:.2f}s, {num_queries} queries)")
        
        time_fusion = time.time() - time_fusion_start
        
        # ===== Step 6: Cross-Encoder Reranking（仅对 top-20） =====
        if self.reranker_model is not None:
            if verbose:
                print(f"[4] Cross-Encoder Reranking (top-{reranker_top_k})...", end='', flush=True)
            
            time_reranker_start = time.time()
            reranker_batch_size = reranker_config.get('batch_size', 8)
            
            with torch.no_grad():
                final_results_list = []
                for query, mmr_results in zip(queries, fused_results_list):
                    mmr_chunk_ids = [cid for cid, _ in mmr_results]
                    
                    # 将chunk_ids分成小 batch（reranker极小 batch）
                    chunk_results = []
                    for chunk_batch_start in range(0, len(mmr_chunk_ids), reranker_batch_size):
                        chunk_batch_end = min(chunk_batch_start + reranker_batch_size, len(mmr_chunk_ids))
                        chunk_batch = mmr_chunk_ids[chunk_batch_start:chunk_batch_end]
                        
                        # reranker批处理
                        batch_results = self.reranker.rerank(
                            query, chunk_batch, self.reranker_model, top_k=len(chunk_batch)
                        )
                        chunk_results.extend(batch_results)
                    
                    # 排序并截断至top-k
                    final_results = sorted(chunk_results, key=lambda x: x[1], reverse=True)[:reranker_top_k]
                    final_results_list.append(final_results)
            
            time_reranker = time.time() - time_reranker_start
            
            if verbose:
                print(f" ✅ 最终 {sum(len(r) for r in final_results_list)} 条 (耗时 {time_reranker:.2f}s, {num_queries} queries)")
        else:
            if verbose:
                print(f"[4] Cross-Encoder Reranking: ⏭️  跳过（未加载 reranker）")
            final_results_list = [[(cid, score) for cid, score in mmr_results[:reranker_top_k]] for mmr_results in fused_results_list]
        
        if verbose:
            print("="*70)
            print(f"\n[性能统计]")
            print(f"  Dense Retrieval:      {time_dense:.2f}s ({num_queries} queries)")
            print(f"  BM25 Reranking:       {time_bm25:.2f}s ({num_queries} queries)")
            print(f"  Fusion (RRF + PB + MMR):")
            print(f"    - RRF:              {time_rrf:.2f}s ({num_queries} queries)")
            print(f"    - Paragraph Boost:  {time_pb:.2f}s ({num_queries} queries)")
            print(f"    - MMR:              {time_mmr:.2f}s ({num_queries} queries)")
            print(f"    - Subtotal:         {time_fusion:.2f}s ({num_queries} queries)")
            if self.reranker_model is not None:
                print(f"  Reranker:             {time_reranker:.2f}s ({num_queries} queries)")
                total_time = time_dense + time_bm25 + time_fusion + time_reranker
            else:
                total_time = time_dense + time_bm25 + time_fusion
            print(f"  \u603b耗时:          {total_time:.2f}s")
            print(f"  平均耗时/query:  {total_time/num_queries:.2f}s")
            print("="*70 + "\n")
        
        # 返回结果（处理单/多查询）
        if is_single_query:
            return final_results_list[0]
        else:
            return final_results_list
