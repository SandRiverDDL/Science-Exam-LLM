"""独立的父Chunk生成器 - SOTA 2025版本

核心原则：
1. 完全独立于子chunk划分
2. 按自然段落和语义单位切分
3. 使用Blingfire进行高效句子切分
4. 每个父chunk覆盖3-6个子chunk
5. 保存字符索引范围，支持子chunk映射

使用流程：
1. 对原始文档按语义单位生成父chunks (256 tokens)
2. 并行生成子chunks (128 tokens) - 不相互影响
3. 通过字符索引范围建立父子映射关系
"""
import re
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

try:
    import blingfire
    HAS_BLINGFIRE = True
except ImportError:
    HAS_BLINGFIRE = False


class ParentChunkGenerator:
    """独立的父Chunk生成器
    
    特点：
    - 不依赖子chunk划分
    - 基于句子和自然段落
    - 使用字符索引标记范围
    - 可配置目标大小和覆盖度
    """
    
    def __init__(
        self,
        tokenizer,
        parent_size: int = 256,  # 用户要求的新大小
        min_parent_tokens: int = 50,
        target_coverage: int = 4  # 每个父chunk覆盖约4个子chunks
    ):
        """初始化
        
        Args:
            tokenizer: transformers tokenizer
            parent_size: 父chunk目标大小（tokens）
            min_parent_tokens: 最小父chunk大小（tokens）
            target_coverage: 每个父chunk覆盖的子chunks数量（约）
        """
        self.tokenizer = tokenizer
        self.parent_size = parent_size
        self.min_parent_tokens = min_parent_tokens
        self.target_coverage = target_coverage
        
        # 预编译断句规则
        self.sentence_break_pattern = re.compile(
            r'([.!?\n。！？;；\r]+\s*)',
            re.MULTILINE
        )
    
    def split_into_sentences(
        self, 
        text: str
    ) -> List[Tuple[str, int, int]]:
        """将文本切分为句子，保留字符位置
        
        使用Blingfire进行高效句子切分（如果可用），
        否则回退到正则表达式方案
        
        Args:
            text: 输入文本
        
        Returns:
            [(sentence_text, start_char, end_char), ...]
        """
        if HAS_BLINGFIRE:
            # 使用Blingfire进行句子切分
            return self._split_sentences_blingfire(text)
        else:
            # 回退正则表达式方案
            return self._split_sentences_regex(text)
    
    def _split_sentences_blingfire(
        self,
        text: str
    ) -> List[Tuple[str, int, int]]:
        """使用Blingfire进行高效的句子切分
        
        优点:
        - 极快（比 spaCy 快 10～50 倍）
        - 纯 C++ 核心，Python 调用开销极小
        - 已经内置 sentence tokenizer
        - 不会误切 "Dr." "e.g." 等缩写
        
        Args:
            text: 输入文本
        
        Returns:
            [(sentence_text, start_char, end_char), ...]
        """
        sentences = []
        
        try:
            # Blingfire输出每个句子一行
            sentence_str = blingfire.text_to_sentences(text)
            sent_list = [s.strip() for s in sentence_str.split("\n") if s.strip()]
            
            # 重新计算字符位置
            current_pos = 0
            for sent_text in sent_list:
                # 在原文中找到这个句子
                found_pos = text.find(sent_text, current_pos)
                if found_pos >= 0:
                    start = found_pos
                    end = found_pos + len(sent_text)
                    sentences.append((sent_text, start, end))
                    current_pos = end
                else:
                    # 备选：不找到时使用第一个字符位置
                    sentences.append((sent_text, current_pos, current_pos + len(sent_text)))
                    current_pos += len(sent_text)
        except Exception as e:
            print(f"[警告] Blingfire处理失败: {e}，回退正则方案")
            sentences = self._split_sentences_regex(text)
        
        return sentences
    
    def _split_sentences_regex(
        self,
        text: str
    ) -> List[Tuple[str, int, int]]:
        """使用正则表达式进行句子切分（备选）
        
        Args:
            text: 输入文本
        
        Returns:
            [(sentence_text, start_char, end_char), ...]
        """
        sentences = []
        
        # 使用正则分割，保留分割符
        parts = self.sentence_break_pattern.split(text)
        
        current_pos = 0
        i = 0
        
        while i < len(parts):
            part = parts[i]
            
            if not part:
                i += 1
                continue
            
            # 检查是否是分割符
            if self.sentence_break_pattern.match(part):
                # 这是分割符，跳过
                current_pos += len(part)
                i += 1
                continue
            
            # 这是文本部分
            text_part = part
            
            # 查看下一个是否是分割符
            if i + 1 < len(parts) and self.sentence_break_pattern.match(parts[i + 1]):
                text_part += parts[i + 1]
                i += 2
            else:
                i += 1
            
            # 记录句子
            start = current_pos
            end = start + len(text_part)
            
            sentence_text = text_part.strip()
            if sentence_text:
                sentences.append((sentence_text, start, end))
            
            current_pos = end
        
        # 如果没有找到任何句子，将整个文本作为一个句子
        if not sentences and text.strip():
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
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            # 降级：粗略估算
            return len(text) // 4
    
    def generate_parent_chunks(
        self,
        doc_id: str,
        text: str,
        title: str = ""
    ) -> List[Dict[str, Any]]:
        """独立生成父chunks - 不依赖子chunks
        
        算法：
        1. 将文本按句子切分
        2. 动态聚合句子直到达到parent_size
        3. 在标点符号处切割
        4. 保存字符索引范围
        
        Args:
            doc_id: 文档ID
            text: 文档正文
            title: 文档标题
        
        Returns:
            父chunk列表，每个包含：
            - parent_id: 父chunk唯一ID
            - doc_id: 所属文档ID
            - title: 文档标题
            - start_char: 在原文中的起始字符位置
            - end_char: 在原文中的结束字符位置
            - text: 父chunk文本（可选）
            - token_count: token计数
        """
        parent_chunks = []
        
        if not text or not text.strip():
            return parent_chunks
        
        # 计算标题占用的空间
        title_tokens = self.count_tokens(title) if title else 0
        available_size = self.parent_size - title_tokens - 2  # 预留特殊token
        
        if available_size < self.min_parent_tokens:
            available_size = self.min_parent_tokens
        
        # Step 1: 按句子切分
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return parent_chunks
        
        # Step 2: 聚合句子形成父chunks
        current_parent_text = ""
        current_start_char = sentences[0][1]  # 使用字符位置
        current_end_char = sentences[0][1]
        parent_count = 0
        
        for sent_text, sent_start, sent_end in sentences:
            # 尝试添加这个句子
            test_text = (current_parent_text + " " + sent_text).strip() if current_parent_text else sent_text
            test_tokens = self.count_tokens(test_text)
            
            # 判断是否超过大小限制
            if test_tokens > available_size and current_parent_text:
                # 保存当前父chunk
                parent_chunks.append(self._create_parent_chunk(
                    doc_id=doc_id,
                    chunk_index=parent_count,
                    text=current_parent_text.strip(),
                    start_char=current_start_char,
                    end_char=current_end_char,
                    title=title
                ))
                
                # 开始新父chunk
                current_parent_text = sent_text
                current_start_char = sent_start
                current_end_char = sent_end
                parent_count += 1
            else:
                # 添加到当前父chunk
                if current_parent_text:
                    current_parent_text += " " + sent_text
                else:
                    current_parent_text = sent_text
                    current_start_char = sent_start
                current_end_char = sent_end
        
        # 保存最后一个父chunk
        if current_parent_text.strip():
            # 检查大小是否满足最小要求
            if self.count_tokens(current_parent_text) >= self.min_parent_tokens:
                parent_chunks.append(self._create_parent_chunk(
                    doc_id=doc_id,
                    chunk_index=parent_count,
                    text=current_parent_text.strip(),
                    start_char=current_start_char,
                    end_char=current_end_char,
                    title=title
                ))
            elif parent_chunks:
                # 如果过小，合并到上一个parent_chunk
                last_chunk = parent_chunks[-1]
                combined_text = last_chunk['text'] + " " + current_parent_text
                last_chunk['text'] = combined_text
                last_chunk['end_char'] = current_end_char
                last_chunk['token_count'] = self.count_tokens(combined_text)
        
        return parent_chunks
    
    def _create_parent_chunk(
        self,
        doc_id: str,
        chunk_index: int,
        text: str,
        start_char: int,
        end_char: int,
        title: str = ""
    ) -> Dict[str, Any]:
        """创建单个父chunk记录
        
        Args:
            doc_id: 文档ID
            chunk_index: chunk序号
            text: chunk文本
            start_char: 起始字符位置
            end_char: 结束字符位置
            title: 文档标题
        
        Returns:
            父chunk记录
        """
        return {
            'parent_id': f"{doc_id}:parent:{chunk_index}",
            'doc_id': doc_id,
            'title': title,
            'start_char': start_char,
            'end_char': end_char,
            'text': text,  # 保留文本用于验证
            'token_count': self.count_tokens(text)
        }
    
    def map_child_to_parent(
        self,
        parent_chunks: List[Dict],
        child_chunks_list: List[Tuple[int, int]]  # [(start_char, end_char), ...]
    ) -> List[int]:
        """将子chunks映射到父chunks
        
        算法：
        1. 首选：完全包含 (p_start <= c_start && c_end <= p_end)
        2. 次选：最大overlap
        3. 末选：返回 -1（未找到）
        
        Args:
            parent_chunks: 父chunks列表（含start_char, end_char）
            child_chunks_list: 子chunks的字符位置列表
        
        Returns:
            [parent_idx_for_child_0, parent_idx_for_child_1, ...]
            其中 -1 表示该子chunk未被任何父chunk包含
        """
        child_to_parent_mapping = []
        
        for cstart, cend in child_chunks_list:
            # 策略1：首先查找完全包含的父chunk
            found = False
            for pidx, parent in enumerate(parent_chunks):
                pstart = parent['start_char']
                pend = parent['end_char']
                
                # 检查是否完全包含
                if pstart <= cstart and cend <= pend:
                    child_to_parent_mapping.append(pidx)
                    found = True
                    break
            
            if found:
                continue
            
            # 策略2：如果没有找到完全包含，查找最大overlap
            best_parent_idx = -1
            best_overlap = 0
            
            for pidx, parent in enumerate(parent_chunks):
                pstart = parent['start_char']
                pend = parent['end_char']
                
                # 计算overlap
                overlap = max(0, min(cend, pend) - max(cstart, pstart))
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent_idx = pidx
            
            child_to_parent_mapping.append(best_parent_idx)
        
        return child_to_parent_mapping
    
    def estimate_child_chunk_size(self, child_size: int = 128) -> int:
        """估算子chunk的字符长度
        
        Args:
            child_size: 子chunk的token大小
        
        Returns:
            对应的字符数（粗略）
        """
        # 粗略估算：1 token ≈ 4 字符
        return child_size * 4


def demonstrate_usage():
    """演示用法
    
    这个函数展示如何使用ParentChunkGenerator
    """
    # 示例：需要实际的tokenizer
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    sample_text = """
    Mitochondria are membrane-bound organelles found in eukaryotic cells. 
    They are often referred to as the "powerhouses" of the cell because they generate 
    most of the cell's supply of adenosine triphosphate (ATP). 
    
    The structure of mitochondria includes an outer membrane and an inner membrane. 
    The inner membrane is highly folded into structures called cristae. 
    This increases the surface area available for chemical reactions.
    
    While mitochondria are primarily known for energy production, they also play roles 
    in other cellular processes. These include signaling, differentiation, and cell death.
    """
    
    # generator = ParentChunkGenerator(tokenizer, parent_size=256)
    # parent_chunks = generator.generate_parent_chunks(
    #     doc_id="doc_001",
    #     text=sample_text.strip(),
    #     title="Mitochondria Structure and Function"
    # )
    
    # for chunk in parent_chunks:
    #     print(f"Parent ID: {chunk['parent_id']}")
    #     print(f"  Chars: {chunk['start_char']}-{chunk['end_char']}")
    #     print(f"  Tokens: {chunk['token_count']}")
    #     print(f"  Text: {chunk['text'][:100]}...\n")


if __name__ == "__main__":
    print(__doc__)
