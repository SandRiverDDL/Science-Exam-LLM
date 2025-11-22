"""基于句子的动态Chunking策略（使用tiktoken）

核心改进：
1. 使用tiktoken作为tokenizer（模型解耦）
2. 基于句子+滑动窗口实现动态chunking
3. 使用字符offset管理文本位置
4. 父chunk 512 tokens / 子chunk 128 tokens
"""
import re
import tiktoken
from typing import List, Dict, Any, Tuple


class SentenceChunker:
    """基于句子的动态分块器
    
    特点：
    1. 句子级切分 + token计数
    2. 使用字符offset，不依赖token offset
    3. 动态组合句子直到达到token限制
    """
    
    def __init__(
        self,
        model_name: str = "cl100k_base",  # GPT-3.5/4使用的编码
        child_size: int = 128,
        parent_size: int = 512,
        min_chunk_chars: int = 50
    ):
        """初始化
        
        Args:
            model_name: tiktoken模型名称（cl100k_base, p50k_base等）
            child_size: 子chunk大小（tokens）
            parent_size: 父chunk大小（tokens）
            min_chunk_chars: 最小chunk字符数
        """
        self.encoding = tiktoken.get_encoding(model_name)
        self.child_size = child_size
        self.parent_size = parent_size
        self.min_chunk_chars = min_chunk_chars
    
    def split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """将文本切分为句子，并记录字符位置
        
        Args:
            text: 输入文本
        
        Returns:
            [(sentence, start_char, end_char), ...]
        """
        sentences = []
        
        # 使用正则表达式切分句子
        # 支持：. ? ! 换行 以及中文标点
        pattern = r'([.!?\n。！？;；]+[\s]*)'
        parts = re.split(pattern, text)
        
        current_pos = 0
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if not part:
                continue
            
            # 如果是标点符号（分隔符）
            if re.match(pattern, part):
                if current_sentence:
                    current_sentence += part
                    # 记录句子
                    start = current_pos
                    end = start + len(current_sentence)
                    sentences.append((current_sentence.strip(), start, end))
                    current_pos = end
                    current_sentence = ""
            else:
                # 普通文本
                current_sentence += part
        
        # 处理最后一个句子
        if current_sentence.strip():
            start = current_pos
            end = start + len(current_sentence)
            sentences.append((current_sentence.strip(), start, end))
        
        # 如果没有找到句子，将整个文本作为一个句子
        if not sentences:
            sentences = [(text.strip(), 0, len(text))]
        
        return sentences
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量
        
        Args:
            text: 输入文本
        
        Returns:
            token数量
        """
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception:
            # 降级：粗略估算（1个token约4个字符）
            return len(text) // 4
    
    def chunk_document(
        self,
        doc_id: str,
        text: str,
        title: str = ""
    ) -> List[Dict[str, Any]]:
        """将文档切分为子chunk和父chunk
        
        Args:
            doc_id: 文档ID
            text: 文档正文
            title: 文档标题
        
        Returns:
            chunk列表，每个chunk包含：
            - chunk_id: chunk唯一ID
            - doc_id: 所属文档ID
            - title: 文档标题
            - child_start: 子chunk在原文中的字符起始位置
            - child_end: 子chunk在原文中的字符结束位置
            - parent_start: 父chunk在原文中的字符起始位置
            - parent_end: 父chunk在原文中的字符结束位置
            - chunk_len: 子chunk的token长度估计
        """
        chunks = []
        
        if not text.strip():
            return chunks
        
        # 计算标题占用的token数
        title_tokens = self.count_tokens(title) if title else 0
        
        # 子chunk可用空间（保留空间给标题）
        child_available = self.child_size - title_tokens - 2  # 预留2个token给特殊符号
        if child_available < 20:
            child_available = 20
        
        # Step 1: 切分句子
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return chunks
        
        # Step 2: 动态组合句子形成子chunks
        child_chunks = []
        current_chunk_text = ""
        current_start = sentences[0][1]
        current_end = sentences[0][1]
        
        for sent_text, sent_start, sent_end in sentences:
            # 尝试添加这个句子
            test_text = current_chunk_text + " " + sent_text if current_chunk_text else sent_text
            test_tokens = self.count_tokens(test_text)
            
            if test_tokens > child_available and current_chunk_text:
                # 超过限制，保存当前chunk
                child_chunks.append({
                    'text': current_chunk_text.strip(),
                    'start': current_start,
                    'end': current_end
                })
                # 开始新chunk
                current_chunk_text = sent_text
                current_start = sent_start
                current_end = sent_end
            else:
                # 添加到当前chunk
                if current_chunk_text:
                    current_chunk_text += " " + sent_text
                else:
                    current_chunk_text = sent_text
                    current_start = sent_start
                current_end = sent_end
        
        # 保存最后一个chunk
        if current_chunk_text.strip():
            child_chunks.append({
                'text': current_chunk_text.strip(),
                'start': current_start,
                'end': current_end
            })
        
        # Step 3: 为每个子chunk生成父chunk
        chunk_count = 0
        for i, child in enumerate(child_chunks):
            child_text = child['text']
            child_start = child['start']
            child_end = child['end']
            
            # 跳过过短的chunk
            if len(child_text) < self.min_chunk_chars:
                continue
            
            # 计算父chunk范围（基于字符位置）
            # 策略：以子chunk为中心，向两侧扩展
            child_center = (child_start + child_end) // 2
            
            # 估算512 tokens对应的字符数（粗略：1 token ≈ 4 chars）
            parent_chars = self.parent_size * 4
            parent_half = parent_chars // 2
            
            parent_start = max(0, child_center - parent_half)
            parent_end = min(len(text), child_center + parent_half)
            
            # 确保父chunk包含子chunk
            parent_start = min(parent_start, child_start)
            parent_end = max(parent_end, child_end)
            
            # 在句子边界调整
            # 向前调整到句子开始
            for sent_text, sent_start, sent_end in sentences:
                if sent_start <= parent_start < sent_end:
                    parent_start = sent_start
                    break
            
            # 向后调整到句子结束
            for sent_text, sent_start, sent_end in sentences:
                if sent_start < parent_end <= sent_end:
                    parent_end = sent_end
                    break
            
            # 创建chunk记录
            chunk_id = f"{doc_id}:chunk:{chunk_count}"
            
            chunk_data = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'title': title,  # 存储标题
                'child_start': child_start,
                'child_end': child_end,
                'parent_start': parent_start,
                'parent_end': parent_end,
                'chunk_len': self.count_tokens(child_text),  # token长度估计
            }
            
            chunks.append(chunk_data)
            chunk_count += 1
        
        return chunks
