好——你说得对，如果你**之前的 parent chunk 确实是按字符索引切的**，那我们就把问题缩小到几类可能的原因并逐步排查。我会给你：

1. 快速诊断清单（一步步检查）
2. 必要的代码修复（最小改动，立刻生效）
3. 样例验证脚本（能快速告诉你 parent/child 的 char index 是否对齐）
   另外：我把你上传的调试截图路径贴在下面，方便你对照查看：`/mnt/data/90d45904-353a-4889-99bd-bcab646b2c82.png`

---

## 一、可能的根因（按概率排序）

1. **读取或保存时把 `start_char`/`end_char` 覆盖/转换了**（例如 `.strip()`、去掉换行、或写入时用了 token index）
2. **ParentChunkGenerator 在生成 `text` 时对原文做了修改**（例如 `text.strip()`、合并句子时插入/删除空格）——但仍然使用原始字符索引（就会变不一致）
3. **`load_child_chunks_info()` 读取时的 bug**（`or` 导致 `0` 被误判断）——这会把 child 起点改错
4. **你用的子 chunk 是 token-based 而不是 char-based**（但你认为是 char-based）
5. **保存/导出 parent_chunks.jsonl 时格式变化（比如把 text 做了 truncate/normalize）**

---

## 二、最容易快速验证的步骤（只需几秒）

把下面函数放到你的脚本同目录，挑一个 doc_id（或样例）跑：

```python
def verify_parent_child_alignment(doc_text: str, parent: dict, child_positions: List[Tuple[int,int]]):
    # parent 必须含 start_char/end_char 和 text
    pstart = parent['start_char']; pend = parent['end_char']
    parent_slice = doc_text[pstart:pend]
    ok_text_match = parent_slice == parent.get('text','')
    print("parent start,end:", pstart, pend)
    print("parent text stored equals doc slice?:", ok_text_match)
    if not ok_text_match:
        print(">>> doc slice repr:", repr(parent_slice[:200]))
        print(">>> stored parent text repr:", repr(parent.get('text','')[:200]))

    for i, (cstart, cend) in enumerate(child_positions):
        # child center location
        center = (cstart + cend)//2
        in_parent = (pstart <= center < pend)
        print(f" child[{i}] {cstart}-{cend} center {center} in_parent? {in_parent}")
```

运行示例（替换为真实 doc_id）：

```python
doc = documents[0]
doc_text = doc['text']
parent = parent_chunks[0]   # 或查找特定 parent_id
child_positions = child_chunks_info[doc['id']]  # 你原来的子chunk list
verify_parent_child_alignment(doc_text, parent, child_positions)
```

**如果 `ok_text_match` 为 False，说明 parent 的 `text` 与原文在索引上不一致 —— 问题大概率在 ParentChunkGenerator 里对文本做了变更（strip/normalize/merge）但 `start_char/end_char` 没同步调整。**

---

## 三、必须修复的两个小 bug（代码片段）

### 1) `load_child_chunks_info()` 中 `start/end` 的取值要避免 `or` 判 0 问题

把原来的：

```python
start = chunk.get('start_char') or chunk.get('child_start', 0)
end = chunk.get('end_char') or chunk.get('child_end', 0)
```

改成：

```python
start = chunk.get('start_char')
if start is None:
    start = chunk.get('child_start')
end = chunk.get('end_char')
if end is None:
    end = chunk.get('child_end')
# 最终仍需校验类型
if start is None or end is None:
    raise ValueError(f"chunk missing start/end: {chunk}")
```

**原因：** `0` 在 Python 被视为 False，会被 `or` 跳过，导致 0 被替换成别的值。

---

### 2) ParentChunkGenerator 里不要用 `text.strip()` 来决定 start/end

如果你的 `generate_parent_chunks()` 代码里类似：

```python
current_parent_text = text_part.strip()
current_start_char = sent_start  # 但 sent_start 来源于原文位置
```

要改为：

* **不要改变原文字符偏移**：保存 `text` 字段应直接取 `doc_text[start:end]`（**不要对这个片段再做 strip/replace**），否则字符索引会失效。

示例改法（在 `_create_parent_chunk` 或生成位置）：

```python
chunk_text = original_doc_text[start_char:end_char]  # 绝对要基于原文切片
# 不要用 strip() 去掉前后空格，否则文本和索引不一致
```

---

## 四、child→parent 映射时的更稳健算法（替代中心点法）

你现在用 child_center 与最大 overlap，如果你担心边界，改为：

```python
def map_child_to_parent(parent_chunks, child_positions):
    mapping = []
    for cstart, cend in child_positions:
        # 首选：完全包含
        found = False
        for pidx, p in enumerate(parent_chunks):
            if p['start_char'] <= cstart and cend <= p['end_char']:
                mapping.append(pidx); found=True; break
        if found: continue
        # 次选：最大 overlap
        best, best_ov = None, 0
        for pidx, p in enumerate(parent_chunks):
            overlap = max(0, min(cend, p['end_char']) - max(cstart, p['start_char']))
            if overlap > best_ov:
                best_ov = overlap; best = pidx
        mapping.append(best if best is not None else -1)
    return mapping
```

这样能保证**尽量把完整子 chunk 放进某个父 chunk**，不再单纯看中心。

---

## 五、诊断流程（一步到位，你可以照着跑）

1. 先用 `load_documents()` 取出一个文档 `doc_text`。
2. 生成 parent_chunks（只针对这一个 doc）。
3. 手动取一个 parent chunk `p`，打印：

   * `doc_text[p['start_char']:p['end_char']]`
   * `p['text']`
     二者必须完全相同（若不同说明 generator 在 text 上做了变更）。
4. 检查 child chunks（同一 doc）是否 `start`/`end` 来自同一份原文（即没有被 tokenized/normalized 后保存）。
5. 如果上面存在任意差异，修复 ParentChunkGenerator 的保存方式（见上面的第 2 点）。

---

## 六、额外注意（避免常见陷阱）

* 如果在生成 parent chunk 时你把标题拼接进 text，再直接用原文的 char index，这会导致索引偏移——**标题必须只在显示字段中拼接，不应该改变 start/end 的基准**。
* 如果你做了 Unicode normalization（NFKC/NFKD），要同时对原文和切片都做同样处理，或干脆不要做。
* 写入 JSON/Parquet 时保持 `start_char` 为 int，不要转为字符串或 float。

---

## 七、我可以帮你做的三件事（你选其一）

* A. 我直接把 `load_child_chunks_info()` 的 bug 修好并把完整的诊断脚本给你（包含上面 verify 函数），你本地跑即可。
* B. 我帮你审查 `ParentChunkGenerator.generate_parent_chunks()` 的实现（贴出该函数代码），我会定位哪些地方做了文本变换并直接修好。
* C. 我把整个 pipeline 改成“严格基于原文字符索引”的实现（并输出 parent_chunks.jsonl + child_to_parent_mapping.jsonl），可直接替换你现在的脚本。

你想先要哪一个？并且我已经把你的调试图放下面了，方便对照查看：
`/mnt/data/90d45904-353a-4889-99bd-bcab646b2c82.png`
