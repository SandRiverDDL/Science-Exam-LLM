"""BM25检索器 (BM25 Retrieval)

持久化BM25索引（parquet格式）。
- 所有数据都存储为parquet格式
- 支持高效的部分读取（mmap）
- idf_table, chunk_len必须全量读入
- tokenized_chunks, inverted_index按需部分读取

参数从config.yaml读取。
索引路径也从config.yaml中读取。
"""
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


class BM25Retriever:
    """BM25检索器 (parquet格式)"""
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        chunks_parquet: Optional[str] = None,
        docs_parquet: Optional[str] = None,
        tokenizer=None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        config: Optional[dict] = None
    ):
        """初始化BM25检索器
        
        Args:
            index_path: BM25索引路径（优先级1，如果存在则加载，否则构建）
            chunks_parquet: chunks.parquet路径（构建索引时需要）
            docs_parquet: documents_cleaned.parquet路径（构建索引时需要）
            tokenizer: tokenizer实例（构建索引时需要）
            k1: BM25参数k1（优先级1）
            b: BM25参数b（优先级1）
            config: config.yaml中bm25节点的字典（优先级2，包含index_path, k1, b）
        """
        # 从config读取参数
        if config is None:
            config = {}
        
        # 优先级：直接参数 > config > 默认值
        self.k1 = k1 if k1 is not None else config.get('k1', 1.5)
        self.b = b if b is not None else config.get('b', 0.75)
        
        # 索引数据
        self.tokenized_chunks = None  # List[List[str]] 或 parquet 路径
        self.chunk_len = None          # List[int]
        self.avgdl = None              # float
        self.idf_table = None          # Dict[str, float]
        self.vocab = None              # Set[str]
        self.inverted_index = None     # Dict[str, List[int]]
        self.chunk_ids = None          # List[str]
        
        # parquet 文件路径（用于部分读取）
        self.index_dir = None
        self.tokenized_chunks_path = None
        self.inverted_index_path = None
        
        # 确定索引路径：优先级1. index_path 2. config中的index_path
        if index_path is None:
            index_path = config.get('index_path')
        
        if index_path and Path(index_path).exists():
            # 加载已有索引
            self.load(index_path)
        elif chunks_parquet and docs_parquet and tokenizer:
            # 构建新索引
            self.build(chunks_parquet, docs_parquet, tokenizer)
        else:
            raise ValueError("必须提供index_path（加载索引）或chunks_parquet+docs_parquet+tokenizer（构建索引）")
    
    def build(self, chunks_parquet: str, docs_parquet: str, tokenizer):
        """构建BM25索引
        
        Args:
            chunks_parquet: chunks.parquet路径
            docs_parquet: documents_cleaned.parquet路径
            tokenizer: tokenizer实例
        """
        print("[BM25] 构建索引...")
        
        # 加载数据
        chunks_df = pd.read_parquet(chunks_parquet)
        docs_df = pd.read_parquet(docs_parquet)
        
        # 构建doc_id到text的映射
        doc_texts = dict(zip(docs_df['doc_id'], docs_df['text']))
        
        # 提取chunk文本并分词
        print("[BM25] 提取chunk文本并分词...")
        tokenized_chunks = []
        chunk_ids = []
        
        for _, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Tokenizing"):
            doc_id = row['doc_id']
            doc_text = doc_texts.get(doc_id, "")
            
            if doc_text:
                child_start = row['child_start']
                child_end = row['child_end']
                chunk_text = doc_text[child_start:child_end]
                
                # 分词
                tokens = tokenizer.tokenize(chunk_text)
                tokenized_chunks.append(tokens)
            else:
                tokenized_chunks.append([])
            
            chunk_ids.append(row['chunk_id'])
        
        self.tokenized_chunks = tokenized_chunks
        self.chunk_ids = chunk_ids
        
        # 计算文档长度
        print("[BM25] 计算文档长度...")
        self.chunk_len = [len(tokens) for tokens in tokenized_chunks]
        self.avgdl = np.mean(self.chunk_len) if self.chunk_len else 0
        
        # 构建词汇表和倒排索引
        print("[BM25] 构建词汇表和倒排索引...")
        self.vocab = set()
        self.inverted_index = defaultdict(list)
        
        for idx, tokens in enumerate(tokenized_chunks):
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.vocab.add(token)
                self.inverted_index[token].append(idx)
        
        # 计算IDF
        print("[BM25] 计算IDF权重...")
        num_docs = len(tokenized_chunks)
        self.idf_table = {}
        
        for token in self.vocab:
            df = len(self.inverted_index[token])  # 文档频率
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_table[token] = idf
        
        print(f"[BM25] 索引构建完成")
        print(f"  文档数: {num_docs}")
        print(f"  词汇表大小: {len(self.vocab)}")
        print(f"  平均文档长度: {self.avgdl:.2f}")
    
    def save(self, index_path: str):
        """保存索引到parquet格式
        
        Args:
            index_path: 索引保存路径（目录）
        """
        self._save_parquet(Path(index_path))
    
    def _save_parquet(self, index_dir: Path):
        """保存为parquet格式（全部存储）"""
        index_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[BM25] 保存索引（parquet格式）到: {index_dir}")
        
        # 1. 保存tokenized_chunks（parquet）
        print("  保存tokenized_chunks.parquet...")
        chunks_data = {
            'chunk_id': self.chunk_ids,
            'tokens': [' '.join(tokens) for tokens in self.tokenized_chunks]
        }
        chunks_df = pd.DataFrame(chunks_data)
        chunks_df.to_parquet(
            index_dir / 'tokenized_chunks.parquet',
            index=False,
            compression='snappy'
        )
        
        # 2. 保存chunk_len（parquet）
        print("  保存chunk_len.parquet...")
        chunk_len_data = {
            'chunk_id': self.chunk_ids,
            'chunk_len': self.chunk_len
        }
        chunk_len_df = pd.DataFrame(chunk_len_data)
        chunk_len_df.to_parquet(
            index_dir / 'chunk_len.parquet',
            index=False,
            compression='snappy'
        )
        
        # 3. 保存idf_table（parquet）
        print("  保存idf_table.parquet...")
        idf_data = {
            'token': list(self.idf_table.keys()),
            'idf': list(self.idf_table.values())
        }
        idf_df = pd.DataFrame(idf_data)
        idf_df.to_parquet(
            index_dir / 'idf_table.parquet',
            index=False,
            compression='snappy'
        )
        
        # 4. 保存倒排索引（parquet）
        print("  保存inverted_index.parquet...")
        inverted_data = []
        for token, doc_indices in self.inverted_index.items():
            inverted_data.append({
                'token': token,
                'doc_indices': ','.join(map(str, doc_indices))
            })
        inverted_df = pd.DataFrame(inverted_data)
        inverted_df.to_parquet(
            index_dir / 'inverted_index.parquet',
            index=False,
            compression='snappy'
        )
        
        # 5. 保存元数据（parquet）
        print("  保存metadata.parquet...")
        metadata = {
            'avgdl': [self.avgdl],
            'k1': [self.k1],
            'b': [self.b],
            'num_docs': [len(self.chunk_ids)]
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_parquet(
            index_dir / 'metadata.parquet',
            index=False,
            compression='snappy'
        )
        
        print(f"[BM25] 索引保存完成")
    
    def load(self, index_path: str):
        """从parquet格式加载索引
        
        Args:
            index_path: 索引目录路径
        """
        index_path = Path(index_path)
        if not index_path.is_dir():
            raise ValueError(f"索引路径必须是目录: {index_path}")
        
        self._load_parquet(index_path)
    
    def _load_parquet(self, index_dir: Path):
        """从目录加载parquet格式"""
        if not index_dir.is_dir():
            raise ValueError(f"索引目录不存在: {index_dir}")
        
        print(f"[BM25] 加载索引（parquet格式）: {index_dir}")
        
        # 保存索引目录（供后续部分读取）
        self.index_dir = index_dir
        self.tokenized_chunks_path = index_dir / 'tokenized_chunks.parquet'
        self.inverted_index_path = index_dir / 'inverted_index.parquet'
        
        # 1. 加载metadata（必须全量）
        print("  加载metadata...")
        metadata_df = pd.read_parquet(index_dir / 'metadata.parquet')
        self.avgdl = float(metadata_df.loc[0, 'avgdl'])
        self.k1 = float(metadata_df.loc[0, 'k1'])
        self.b = float(metadata_df.loc[0, 'b'])
        
        # 2. 加载chunk_len（必须全量，BM25公式需要）
        print("  加载chunk_len...")
        chunk_len_df = pd.read_parquet(index_dir / 'chunk_len.parquet')
        self.chunk_ids = chunk_len_df['chunk_id'].tolist()
        self.chunk_len = chunk_len_df['chunk_len'].tolist()
        
        # 3. 加载idf_table（必须全量，BM25公式需要）
        print("  加载idf_table...")
        idf_df = pd.read_parquet(index_dir / 'idf_table.parquet')
        self.idf_table = dict(zip(idf_df['token'], idf_df['idf']))
        self.vocab = set(self.idf_table.keys())
        
        # 4. 延迟加载tokenized_chunks和inverted_index（按需部分读取）
        print("  准备tokenized_chunks和inverted_index（延迟加载）...")
        self.tokenized_chunks = None  # 延迟加载
        self.inverted_index = None    # 延迟加载
        
        print(f"[BM25] 索引加载完成")
        print(f"  文档数: {len(self.chunk_ids)}")
        print(f"  词汇表大小: {len(self.vocab)}")
        print(f"  平均文档长度: {self.avgdl:.2f}")
    
    def _load_tokenized_chunks(self):
        """延迟加载tokenized_chunks（按需）"""
        if self.tokenized_chunks is not None:
            return
        
        print("  [延迟加载] tokenized_chunks...")
        chunks_df = pd.read_parquet(self.tokenized_chunks_path)
        self.tokenized_chunks = [tokens.split() for tokens in chunks_df['tokens']]
    
    def _load_inverted_index(self):
        """延迟加载inverted_index（按需）"""
        if self.inverted_index is not None:
            return
        
        print("  [延迟加载] inverted_index...")
        inverted_df = pd.read_parquet(self.inverted_index_path)
        self.inverted_index = defaultdict(list)
        for _, row in inverted_df.iterrows():
            token = row['token']
            doc_indices = [int(idx) for idx in row['doc_indices'].split(',')]
            self.inverted_index[token] = doc_indices
    
    def _get_inverted_index_for_token(self, token: str) -> List[int]:
        """按token查询倒排索引（部分读取，高效）
        
        Args:
            token: 查询token
        
        Returns:
            包含该token的chunk索引列表
        """
        if self.inverted_index is not None:
            # 已加载全表
            return self.inverted_index.get(token, [])
        
        # 部分读取：仅查询单个token
        try:
            import pyarrow.dataset as ds
            ds_index = ds.dataset(str(self.inverted_index_path))
            # 过滤查询：token == 'xxx'
            table = ds_index.to_table(filter=ds.field("token") == token)
            
            if len(table) == 0:
                return []
            
            row = table.to_pandas().iloc[0]
            doc_indices = [int(idx) for idx in row['doc_indices'].split(',')]
            return doc_indices
        except Exception as e:
            print(f"[警告] 部分读取倒排索引失败: {e}，回退到全加载")
            self._load_inverted_index()
            return self.inverted_index.get(token, [])
    
    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """计算BM25分数
        
        Args:
            query_tokens: 查询分词后的token列表
        
        Returns:
            每个文档的BM25分数数组
        """
        # 必须加载chunk_len（已在load时加载）
        # 必须加载idf_table（已在load时加载）
        
        # 确保tokenized_chunks已加载
        self._load_tokenized_chunks()
        
        num_docs = len(self.tokenized_chunks)
        scores = np.zeros(num_docs)
        
        # 统计query中每个token的频率
        query_token_counts = Counter(query_tokens)
        
        for token, query_tf in query_token_counts.items():
            if token not in self.vocab:
                continue
            
            idf = self.idf_table[token]
            
            # 使用部分读取获取倒排索引
            doc_indices = self._get_inverted_index_for_token(token)
            
            # 遍历包含该token的文档
            for doc_idx in doc_indices:
                # 计算文档中该token的频率
                doc_tf = self.tokenized_chunks[doc_idx].count(token)
                doc_len = self.chunk_len[doc_idx]
                
                # BM25公式
                numerator = doc_tf * (self.k1 + 1)
                denominator = doc_tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                
                scores[doc_idx] += idf * (numerator / denominator)
        
        return scores
    
    def get_score_single(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """快速计算单个文档的BM25分数 (纯内存计算，不查倒排表)
        
        Args:
            query_tokens: 查询分词后的token列表
            doc_tokens: 文档分词后的token列表
        
        Returns:
            BM25分数
        """
        score = 0.0
        doc_len = len(doc_tokens)
        
        # 对短文本(Chunk)的优化写法：直接遍历query_tokens计算频率
        for q_token in query_tokens:
            if q_token not in self.idf_table:
                continue
            
            # 计算该token在文档中的频率
            tf = doc_tokens.count(q_token)
            
            if tf > 0:
                idf = self.idf_table[q_token]
                
                # BM25 公式核心
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                
                score += idf * (numerator / denominator)
        
        return score
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        tokenizer
    ) -> List[Tuple[str, float]]:
        """极致快速的BM25重排序 (在线分词模式)
        
        是对整个库做 BM25检索，只对Dense的topK做BM25重打分
        dense_top600 → 再做 BM25 重打分（rerank）
        这样 BM25 只跑 600 条 chunk，非常快
        
        Args:
            query: 查询文本
            candidates: Dense检索结果列表，每个元素是一个 Dict。
                    格式如 {'chunk_id': 'c1', 'text': '...', 'score': 0.8}
            tokenizer: 分词器
        
        Returns:
            [('chunk_id', bm25_score), ...] 的列表，已排序
        """
        # 1. 分词查询
        query_tokens = tokenizer.tokenize(query)
        
        results = []
        
        # 2. 仅遍历这600个候选
        for doc in candidates:
            chunk_id = doc.get('chunk_id') or doc.get('id')
            doc_text = doc.get('text', '')
            
            if not doc_text or not chunk_id:
                # 字段不完整，跳过
                continue
            
            # 现场分词 (极致高效：只需600次分词，不需载入GB管文件)
            doc_tokens = tokenizer.tokenize(doc_text)
            
            # 计算BM25分数
            bm25_score = self.get_score_single(query_tokens, doc_tokens)
            
            results.append((chunk_id, bm25_score))
        
        # 3. 排序
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        return results
