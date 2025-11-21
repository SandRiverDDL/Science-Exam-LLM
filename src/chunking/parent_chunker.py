"""父文档索引 (Parent Document) Chunking策略

SOTA 2025 版本：
- 子文档：128 tokens，用于精准检索（无overlap）
- 父文档：512 tokens，提供给 LLM 的完整上下文
- 回溯断句：在标点符号处切分，保持语义完整性
"""
from typing import List, Tuple, Dict, Any, Optional


class ParentDocumentChunker:
    """父文档索引分块器
    
    核心策略：
    1. 子chunk (128 tokens) 用于检索 - 语义集中
    2. 父chunk (512 tokens) 用于LLM - 上下文完整
    3. 在标点符号处断句 - 保持语义完整性
    """
    
    def __init__(
        self,
        tokenizer,
        child_size: int = 128,
        parent_size: int = 512,
        min_chunk_tokens: int = 32
    ):
        """初始化
        
        Args:
            tokenizer: transformers tokenizer
            child_size: 子文档大小（用于检索）
            parent_size: 父文档大小（用于LLM）
            min_chunk_tokens: 最小chunk大小
        """
        self.tokenizer = tokenizer
        self.child_size = child_size
        self.parent_size = parent_size
        self.min_chunk = min_chunk_tokens
        
        # 预计算标点符号的 token ID（用于回溯断句）
        self.break_tokens = set()
        for punctuation in ['.', '?', '!', '\n', '。', '！', '？', ';', '；']:
            try:
                ids = tokenizer.encode(punctuation, add_special_tokens=False)
                if ids:
                    self.break_tokens.add(ids[-1])
            except Exception:
                pass
    
    def chunk_document(
        self,
        doc_id: str,
        title_ids: List[int],
        doc_ids: List[int],
        title_text: str = ""
    ) -> List[Dict[str, Any]]:
        """将文档切分为子chunk和父chunk
        
        Args:
            doc_id: 文档ID
            title_ids: 标题的token IDs
            doc_ids: 文档正文的token IDs
            title_text: 标题文本（用于rerank_text）
        
        Returns:
            chunk列表，每个chunk包含：
            - chunk_id: chunk唯一ID
            - doc_id: 所属文档ID
            - child_ids: 子chunk的token IDs (128)
            - parent_start: 父chunk在doc_ids中的起始位置
            - parent_end: 父chunk在doc_ids中的结束位置
            - rerank_text: 拼接了标题的文本（用于reranker）
            - chunk_len: 子chunk长度
        """
        chunks = []
        
        if not doc_ids:
            return chunks
        
        # 计算title占用的空间
        title_len = len(title_ids)
        
        # 子chunk的实际可用大小（保留空间给标题和特殊token）
        # [CLS] + title_ids + [SEP] + child_ids + [SEP]
        # 假设特殊token占用2个位置
        child_available = self.child_size - title_len - 2
        
        # 确保至少有一些空间给正文
        if child_available < 20:
            child_available = 20
        
        # 开始切分子chunk（无overlap）
        start_idx = 0
        chunk_count = 0
        
        while start_idx < len(doc_ids):
            # 1. 确定子chunk的结束位置
            end_idx = min(start_idx + child_available, len(doc_ids))
            
            # 2. 回溯断句（Lookback Strategy）
            # 如果不是文档末尾，尝试在标点符号处切分
            if end_idx < len(doc_ids):
                lookback_range = min(20, end_idx - start_idx)
                found_break = False
                
                for i in range(lookback_range):
                    curr_pos = end_idx - 1 - i
                    if curr_pos > start_idx and doc_ids[curr_pos] in self.break_tokens:
                        end_idx = curr_pos + 1  # 包含标点符号
                        found_break = True
                        break
            
            # 3. 提取子chunk
            child_ids = doc_ids[start_idx:end_idx]
            
            # 4. 处理过短的最后一个chunk
            if end_idx == len(doc_ids) and len(child_ids) < self.min_chunk:
                if chunks:
                    # 合并到上一个chunk
                    # 这里简单处理：丢弃过短的尾巴
                    break
                # 如果是唯一的chunk，保留它
            
            # 5. 计算父chunk的范围（扩展到512 tokens）
            # 策略：以子chunk中心为基准，向两侧扩展
            child_center = (start_idx + end_idx) // 2
            parent_half = self.parent_size // 2
            
            parent_start = max(0, child_center - parent_half)
            parent_end = min(len(doc_ids), child_center + parent_half)
            
            # 确保父chunk至少包含子chunk
            parent_start = min(parent_start, start_idx)
            parent_end = max(parent_end, end_idx)
            
            # 调整父chunk大小到目标大小
            current_parent_size = parent_end - parent_start
            if current_parent_size < self.parent_size:
                # 向后扩展
                expand = self.parent_size - current_parent_size
                parent_end = min(len(doc_ids), parent_end + expand)
                
                # 如果还不够，向前扩展
                if parent_end - parent_start < self.parent_size:
                    expand_front = self.parent_size - (parent_end - parent_start)
                    parent_start = max(0, parent_start - expand_front)
            
            # 6. 生成rerank_text（标题 + 子chunk文本）
            # Reranker需要完整的文本用于Cross-Attention
            try:
                child_text = self.tokenizer.decode(child_ids, skip_special_tokens=True)
                if title_text:
                    rerank_text = f"{title_text}\n{child_text}"
                else:
                    rerank_text = child_text
            except Exception:
                rerank_text = ""
            
            # 7. 创建chunk记录
            chunk_id = f"{doc_id}:chunk:{chunk_count}"
            
            chunk_data = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'child_ids': child_ids,
                'parent_start': parent_start,
                'parent_end': parent_end,
                'rerank_text': rerank_text,
                'chunk_len': len(child_ids),
            }
            
            chunks.append(chunk_data)
            chunk_count += 1
            
            # 8. 移动到下一个子chunk（无overlap）
            start_idx = end_idx
        
        return chunks
    
    def prepare_embedding_inputs(
        self,
        title_ids: List[int],
        child_ids: List[int],
        add_special_tokens: bool = True
    ) -> List[int]:
        """准备用于embedding的输入序列
        
        Args:
            title_ids: 标题token IDs
            child_ids: 子chunk token IDs
            add_special_tokens: 是否添加[CLS]和[SEP]
        
        Returns:
            完整的输入token IDs
        """
        if add_special_tokens:
            # [CLS] + title + [SEP] + child + [SEP]
            input_ids = [self.tokenizer.cls_token_id]
            
            if title_ids:
                input_ids.extend(title_ids)
                input_ids.append(self.tokenizer.sep_token_id)
            
            input_ids.extend(child_ids)
            input_ids.append(self.tokenizer.sep_token_id)
        else:
            # 直接拼接
            input_ids = title_ids + child_ids if title_ids else child_ids
        
        # 截断到最大长度（兜底）
        if len(input_ids) > self.child_size:
            input_ids = input_ids[:self.child_size]
        
        return input_ids
