下面我根据你最新的需求 **重新整理、重写并强化你的指令**，使其变成一个 **清晰、可执行、无歧义、可直接交给工程师或 LLM 执行**的版本。

我会给你两版：

1. **普通中文版本（最清晰）**
2. **适合让 AI 直接执行的“LLM 指令版本”（更严格格式化）**

---

# ✅ **（A）重新整理后的更清晰、专业的指令（推荐给你自己使用）**

下面是改进后的整体任务描述：

---

## **🎯 任务目标**

需要编写一个脚本，对现有的 `chunks.parquet` 中的父 chunk（parent chunks）进行重新划分，使每个父 chunk 的长度不超过 **256 token**。

在重新生成父 chunk 时：

* 父 chunk 的字符索引（`start_char` / `end_char`）必须保持与 `documents_cleaned.parquet` 中的原始 `text` 完全一致。
* **不能对 text 做任何修改**（例如去除空格、替换字符、正则清洗等），否则父子 chunk 的字符定位会错乱。
* 在重新切分父 chunk 时，要优先保证 **语义单元完整性**，分割点尽量靠近：

  * 句号（。．.）
  * 换行符
  * 中文句末标点（！？，；）
  * 英文句末标点 (.!?)
  * 其他自然句边界

---

## **📦 输入文件**

1. `data/processed/documents_cleaned.parquet`

   * 包含字段：`doc_id`, `text`
2. `data/processed/chunks.parquet`

   * 包含字段：`doc_id`, `chunk_id`, `is_parent`, `start_char`, `end_char`, `token_count` 等
   * 其中的 **parent chunk 的定位需要被重新调整**


## **🔗 父子 chunk 的映射（你给的算法）**

你不需要简单的中心点（child_center），而是要：

### **1) 优先选择“完全包含”父 chunk**

```python
if p.start_char <= c.start_char and c.end_char <= p.end_char
```

### **2) 若无完全包含，则选择“最大字符 overlap”的父 chunk**

你提供的算法已经很好，我格式化成更清晰版本：

```python
def map_child_to_parent(parent_chunks, child_positions):
    mapping = []
    for cstart, cend in child_positions:
        # 1. 完全包含优先
        found = False
        for pidx, p in enumerate(parent_chunks):
            if p['start_char'] <= cstart and cend <= p['end_char']:
                mapping.append(pidx)
                found = True
                break
        if found:
            continue

        # 2. 最大 overlap 次之
        best, best_ov = None, 0
        for pidx, p in enumerate(parent_chunks):
            overlap = max(0, min(cend, p['end_char']) - max(cstart, p['start_char']))
            if overlap > best_ov:
                best_ov = overlap
                best = pidx

        mapping.append(best if best is not None else -1)

    return mapping
