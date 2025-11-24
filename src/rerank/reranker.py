"""Cross-Encoder Reranker (交叉编码器重排序)

使用Jina Reranker进行最终的精准重排序
"""
from typing import List, Tuple
import pandas as pd

class CrossEncoderReranker:
    """交叉编码器重排序器（高性能优化版）"""
    
    def __init__(self, chunks_parquet: str, docs_parquet: str):
        """初始化Reranker（保持原接口不变）"""
        self.chunks_df = pd.read_parquet(chunks_parquet)
        self.docs_df = pd.read_parquet(docs_parquet)

        # ---- O(1) 加速：构建字典索引 ----
        self.chunk_rows = {
            row["chunk_id"]: row
            for _, row in self.chunks_df.iterrows()
        }
        self.doc_texts = dict(zip(self.docs_df["doc_id"], self.docs_df["text"]))

        # ---- 父 chunk 缓存 ----
        self.parent_cache = {}

    # -------------------------------
    # 工具：截断文本避免超 token
    # -------------------------------
    @staticmethod
    def truncate(text: str, max_chars: int = 2200) -> str:
        """
        为 CrossEncoder 截断文本（模型多为 512 tokens，≈2000 chars）
        """
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    # -------------------------------
    # 子chunk文本（基本不用）
    # -------------------------------
    def get_chunk_text(self, chunk_id: str) -> str:
        row = self.chunk_rows.get(chunk_id)
        if row is None:
            return ""
        
        doc_text = self.doc_texts.get(row["doc_id"], "")
        return doc_text[row["child_start"]:row["child_end"]]

    # -------------------------------
    # 父 chunk 文本（主用于重排）
    # -------------------------------
    def get_parent_chunk_text(self, chunk_id: str) -> str:
        """获取父 chunk（含标题 + 上下文 512 tokens）"""
        
        # ---- 缓存命中 ----
        if chunk_id in self.parent_cache:
            return self.parent_cache[chunk_id]

        row = self.chunk_rows.get(chunk_id)
        if row is None:
            return ""

        doc_id = row["doc_id"]
        doc_text = self.doc_texts.get(doc_id, "")

        parent_text = doc_text[row["parent_start"]:row["parent_end"]]
        
        # 标题拼接
        if row.get("title"):
            combined = f"{row['title']}\n\n{parent_text}"
        else:
            combined = parent_text

        # ---- token 长度控制 ----
        combined = self.truncate(combined, max_chars=2200)

        self.parent_cache[chunk_id] = combined
        return combined

    # -------------------------------
    # Cross-Encoder Rerank
    # -------------------------------
    def rerank(
        self,
        query: str,
        chunk_ids: List[str],
        reranker_model,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        使用 Cross-Encoder 进行重排序
        返回格式 [(chunk_id, score), ...]
        """
        query_chunk_pairs = []
        valid_chunk_ids = []

        for cid in chunk_ids:
            chunk_text = self.get_parent_chunk_text(cid)
            if not chunk_text:
                continue

            # 仍保持你的双文本格式
            query_chunk_pairs.append([query, chunk_text])
            valid_chunk_ids.append(cid)

        if not query_chunk_pairs:
            return []

        # ---- 模型计算 ----
        scores = reranker_model.rank(query_chunk_pairs)

        # 安全性检查
        if len(scores) != len(valid_chunk_ids):
            raise RuntimeError(
                f"Reranker score length mismatch: "
                f"{len(scores)} vs {len(valid_chunk_ids)}"
            )

        # ---- 排序 + 截断 ----
        ranked = sorted(
            zip(valid_chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return ranked

