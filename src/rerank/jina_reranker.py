"""Jina Reranker v3 Implementation

Uses jinaai/jina-reranker-v3 model for document reranking.
"""
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModel


class JinaReranker:
    """Jina Reranker v3 Model"""
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        batch_size: int = 8,
        trust_remote_code: bool = True,
        config: Optional[dict] = None
    ):
        """Initialize Jina Reranker
        
        Args:
            model_id: Model ID from HuggingFace (从config读取)
            device: Device to use (auto-detect if None)
            dtype: Data type ("auto" for automatic)
            batch_size: Batch size for inference (从config读取)
            trust_remote_code: Whether to trust remote code
            config: Config dict (优先级高于单独参数)
        """
        # 从config读取参数
        if config is None:
            try:
                from core.config import Config
                cfg = Config()
                # cfg 对象本身不为 None，但控制地获取 reranker 键，或者执行失败
                if cfg is not None and hasattr(cfg, 'get'):
                    try:
                        # reranker 已经归到 retrieval 下了，所以从 retrieval.reranker 读取
                        config = cfg.get('retrieval', {}).get('reranker', {})
                    except:
                        config = {}
                else:
                    config = {}
            except Exception as e:
                print(f"[Jina] Warning: Failed to load config from Config class: {e}")
                config = {}
        
        # 优先使用config中的参数
        if config is None:
            config = {}
        
        model_id = model_id or config.get('model_id', 'jinaai/jina-reranker-v3') if isinstance(config, dict) else 'jinaai/jina-reranker-v3'
        batch_size = config.get('batch_size', batch_size) if isinstance(config, dict) else batch_size
        device = device or (config.get('device') if isinstance(config, dict) else None)
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_id = model_id
        self.batch_size = batch_size
        
        print(f"[Jina] Loading model: {model_id}")
        print(f"  Device: {device}, Dtype: {dtype}, Batch size: {batch_size}")
        
        # Load model (batch_size不是AutoModel参数，移除)
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code
        )
        self.model.eval()
        self.model.to(device)
        
        print(f"[Jina] Model loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict]:
        """Rerank documents based on query
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Return top-k results (None for all)
            batch_size: Batch size for inference (None=使用默认配置)
        
        Returns:
            List of dicts with keys:
                - index: Original index in documents
                - document: Document text
                - relevance_score: Relevance score (0-1)
        """
        if not documents:
            return []
        
        # 使用配置的batch_size
        if batch_size is None:
            batch_size = self.batch_size
        
        # Process in batches to avoid OOM
        all_results = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # Get scores for this batch
            batch_results = self.model.rerank(query, batch_docs)
            
            # Adjust indices to global position
            for result in batch_results:
                result["index"] = result["index"] + i
            
            all_results.extend(batch_results)
        
        # Sort all results by relevance score (descending)
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            all_results = all_results[:top_k]
        
        return all_results
    
    def rank(
        self,
        query_document_pairs: List[List[str]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[float]:
        """Rank query-document pairs
        
        Args:
            query_document_pairs: List of [query, document] pairs
            top_k: Return top-k (not used for this method)
            batch_size: Batch size for inference (None=使用默认配置)
        
        Returns:
            List of relevance scores (in original order)
        """
        if not query_document_pairs:
            return []
        
        # 使用配置的batch_size
        if batch_size is None:
            batch_size = self.batch_size
        
        # Assume all pairs have the same query
        query = query_document_pairs[0][0]
        documents = [pair[1] for pair in query_document_pairs]
        
        # Process in batches
        all_scores = [0.0] * len(documents)
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # Get scores for this batch
            batch_results = self.model.rerank(query, batch_docs)
            
            # Store scores in original order
            for result in batch_results:
                original_idx = i + result["index"]
                all_scores[original_idx] = result["relevance_score"]
        
        return all_scores