
SOTA 改进方案 A：基于 Token 的“回溯断句” (Lookback Strategy)
不要在第 256 个 Token 处直接一刀切，而是往回看。
逻辑： 取出 256 个 Token 后，检查最后 10-20 个 Token。
动作： 寻找 “句子结束符”（如 . ? ! \n 的 Token ID）。
结果：
如果找到了（例如在第 245 个 Token），就在那里切断。
如果没找到（这句太长了），再进行硬切分。
优势： 保证了绝大多数 Chunk 都是以完整句子结束的，Embedding 质量显著提升。

#### SOTA 改进方案 B：父文档索引 (Parent Document / Small-to-Big)

这是目前 RAG 中最强的策略之一。

*   **逻辑：**
    *   **切分：** 使用较小的 `max_seq` (例如 128) 进行切分，甚至不做 overlap。
    *   **索引：** 对这 128 tokens 进行 Embedding。
    *   **存储：** 在 Metadata 中，存储该 Chunk 所属的**“父窗口”**（例如前后各扩 200 tokens，或者是整段文本）。
    *   **检索：** 搜到小 Chunk，给 LLM 看大窗口。
*   **优势：** 检索精准（小块语义集中），回答全面（大块上下文完整）。

---

### 2. 预 Tokenization 与 Chunking 的融合

**是的，必须融合。** 这就是“一次 Tokenize，多次切片”的高效模式。

#### 为什么？
Tokenize 是文本处理中最慢的一步（涉及查找树、合并 BPE 等）。如果你在 Chunking 循环里对每个切出来的文本块再做一次 Encode，你的效率会降低 10 倍以上。

#### 最佳实践流程

1.  **Doc Level:** 读取文本 -> 清洗 -> **一次性 Tokenize 整个文档** -> 得到 `full_doc_input_ids` (List[int])。
 **Chunk Level:** 读取 Parquet -> 在 `input_ids` 列表上做**列表切片 (List Slicing)** -> 得到 `chunk_input_ids`。
4.  **Final Assembly:** `[CLS] + Title_IDs + Chunk_Slice + [SEP]`。

---

### 3. 改进后的完整代码实现 (SOTA 版)

这段代码实现了**预 Tokenize** + **语义回溯切分** + **高效拼接**。可以直接用于你的工程中。

```python
import numpy as np
from typing import List, Tuple

class SOTAChunker:
    def __init__(self, tokenizer, max_seq_length=256, overlap=32, min_chunk_tokens=64):
        self.tokenizer = tokenizer
        self.max_seq = max_seq_length
        self.overlap = overlap
        self.min_chunk = min_chunk_tokens
        
        # 预先获取常见标点符号的 Token ID，用于回溯断句
        # 注意：不同 Tokenizer (BERT vs Llama) 的 ID 不同，这里需要根据你的模型调整
        # 比如 BERT 中 '.' 可能是 1012，'\n' 可能是 102
        self.break_tokens = set()
        for t in ['.', '?', '!', '\n', '。', '！', '？']:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if ids:
                self.break_tokens.add(ids[-1])  # 取最后一个 token id
                
    def process_document(self, doc_text: str, title: str = "", entities: str = "") -> List[List[int]]:
        """
        输入原始文本，输出处理好的、可以直接送入模型的 input_ids 列表
        """
        # 1. 预处理 & Tokenize (一次性完成)
        # 注意：add_special_tokens=False，我们最后手动加
        doc_ids = self.tokenizer.encode(doc_text, add_special_tokens=False)
        title_ids = self.tokenizer.encode(title, add_special_tokens=False) if title else []
        entity_ids = self.tokenizer.encode(entities, add_special_tokens=False) if entities else []
        
        # 2. 计算正文的最大可用窗口
        # max_seq - [CLS] - [SEP] - len(title) - len(entities) - [Sep_between_meta_body]
        reserved_len = 2 + len(title_ids) + len(entity_ids) + (1 if (title_ids or entity_ids) else 0)
        body_window_size = self.max_seq - reserved_len
        
        if body_window_size < 50: # 如果标题实体太长，强行截断它们，给正文留点地盘
            # 策略：保留 50 tokens 给正文，压缩 metadata
            # 这里简化处理，实际可做 truncate logic
            body_window_size = 50
            reserved_len = self.max_seq - 50

        final_chunks = []
        start_idx = 0
        
        while start_idx < len(doc_ids):
            # A. 确定初步的结束位置
            end_idx = min(start_idx + body_window_size, len(doc_ids))
            
            # B. 语义回溯 (Lookback) - 关键改进点
            # 如果不是文档末尾，尝试在标点符号处断开
            if end_idx < len(doc_ids):
                # 往回看最多 20 个 token
                lookback_range = 20 
                found_break = False
                for i in range(lookback_range):
                    curr_pos = end_idx - 1 - i
                    if curr_pos > start_idx and doc_ids[curr_pos] in self.break_tokens:
                        end_idx = curr_pos + 1 # 包含标点
                        found_break = True
                        break
                
                # 如果没找到标点，就硬切分，保持原样
            
            # C. 提取当前 Chunk 的 Body
            chunk_body_ids = doc_ids[start_idx:end_idx]
            
            # D. 短 Chunk 处理 (Merge Last Logic)
            # 如果这是最后一段，且太短
            if end_idx == len(doc_ids) and len(chunk_body_ids) < self.min_chunk:
                if final_chunks:
                    # 拿出上一个 chunk
                    prev_chunk = final_chunks.pop()
                    # 这是一个简单的合并策略：合并到上一个。
                    # 注意：合并后可能会超长，所以需要截断上一个的头部，或者接受略微超长(如果模型支持)
                    # 竞赛稳妥做法：丢弃这个极短的尾巴，或者拼接到上一个但不超过 max_seq
                    
                    # 简单的丢弃策略 (如果信息量真的太少):
                    pass 
                else:
                    # 如果只有一个 chunk 且很短，还是得保留
                    final_ids = self._build_final_input(title_ids, entity_ids, chunk_body_ids)
                    final_chunks.append(final_ids)
            else:
                # 正常构建
                final_ids = self._build_final_input(title_ids, entity_ids, chunk_body_ids)
                final_chunks.append(final_ids)
            
            # E. 计算下一个滑窗的起点 (包含 Overlap)
            if end_idx == len(doc_ids):
                break
            
            # 下一个起点 = 当前结束点 - Overlap
            start_idx = end_idx - self.overlap
            
            # 防止死循环 (如果 overlap >= window size)
            if start_idx >= end_idx:
                start_idx = end_idx
                
        return final_chunks

    def _build_final_input(self, title_ids, entity_ids, body_ids):
        """组装：[CLS] Title + Entities [SEP] Body [SEP]"""
        # 这里分隔符策略可以自定义，比如用 \n\n 的 token id
        # 简单起见使用 input format:
        # [CLS] title... entities... [SEP] body... [SEP]
        
        prefix = title_ids + entity_ids
        if prefix:
             # 假如 tokenizer.sep_token_id 是分隔符
            prefix += [self.tokenizer.sep_token_id] 
            
        combined = [self.tokenizer.cls_token_id] + prefix + body_ids + [self.tokenizer.sep_token_id]
        
        # 再次兜底截断 (防止 floating point 误差或逻辑漏洞)
        return combined[:self.max_seq]

# 使用示例
# chunker = SOTAChunker(tokenizer)
# all_chunks_for_doc = chunker.process_document(text, title, entities)
```

### 关键改进总结

1.  **Lookback Logic (回溯断句):**
    *   代码中的 `Section B`。它避免了把句子切断。这对于 Embedding 模型的语义理解至关重要。
2.  **预 Tokenization:**
    *   `doc_ids = tokenizer.encode(...)` 在循环外只做了一次。所有的切分都是在 `List[int]` 上操作，速度极快。
3.  **Token ID 拼接:**
    *   `_build_final_input` 方法完全没有进行 decode，直接拼接整数列表，效率最高。
4.  **动态窗口计算:**
    *   `body_window_size` 是根据 `title` 和 `entity` 的长度动态计算的。如果标题长，正文窗口就自动变小，保证总长度不超标。

### 还有一个小细节：Metadata 的分隔符

在 `title` 和 `body` 之间，最好放一个分隔符。
*   在 BERT 架构中，通常用 `[SEP]`。
*   或者，更自然的方式是插入 `\n` 或 `\n\n` 的 Token ID。
*   我的代码中为了简单使用了 `sep_token_id`，你可以根据具体的 Embedding 模型习惯（例如 BGE 推荐的方式）来调整。