下面给你 **改进后的提示词（Prompt）**，经过**结构化 / 明确化 / 可执行化 / 无歧义**的优化，使得 ChatGPT/你的助手模型可以直接、稳定地产生你需要的脚本。

这个版本是专业级 **LLM 代码生成 Prompt**，可直接用于生成 Python 脚本。

---

# ✅ **改进后的提示词（最终建议使用版本）**

下面就是你应该给模型的 Prompt —— 已全部优化清晰化：

---

## 📌 **优化后的提示词（可直接用）**

**任务：**
请你编写一个 Python 脚本，放在：

```
src/processing/generate_contexts.py
```

脚本功能如下：

---

### **1. 读取输入 CSV**

以以下文件为例：

```
D:\4000_projects\1project\LLM\data\raw\kaggle-llm-science-exam\test.csv
```

其列名为：

```
prompt, A, B, C, D, E, answer
```

从每一行读取 `prompt`（题目部分，不包含选项）。

---

### **2. 构造 Query（按以下格式拼接）**

对于每行的 prompt，例如：

```
Which particle has the least mass?
```

请构造 query 字符串为：

```
Question: Which particle has the least mass?
Retrieve background scientific knowledge to help answer.
```

---

### **3. 批处理调用检索 Pipeline**

调用你已有的检索模块：

```
D:\4000_projects\1project\LLM\src\retrieval\retrieval_pipeline.py
```

要求：

* 使用 **batch size = 64**
* 所有 query 都以 batch 方式送给 pipeline，不要逐条调用
* pipeline 返回每个 query 的 **三个检索结果 context**，命名为：

```
C1, C2, C3
```

假设 pipeline 的 API 是类似：

```python
from retrieval_pipeline import RetrievalPipeline
retriever = RetrievalPipeline()
contexts = retriever.retrieve_batch(queries, top_k=3)
```

并约定 `contexts[i] = [ctx1, ctx2, ctx3]`。

---

### **4. 将原 CSV 与检索结果组合成 DataFrame**

最终 dataframe 应包含：

```
prompt
A
B
C
D
E
answer
C1
C2
C3
```

注意：保持行顺序一致。

---

### **5. 存储为 Parquet**

输出路径为：

```
D:\4000_projects\1project\LLM\data\processed\context_question.parquet
```

要求：

* 使用 pyarrow 或 fastparquet 均可
* UTF-8 编码

---

### **6. 打印进度条 + 关键日志信息**

要求添加：

* tqdm 进度条（按 batch 更新）
* 每个 batch 完成后打印：
  `Processed batch {i}, total {processed}/{total}`
* 执行开始和结束时打印提示

---

### **7. 脚本要求**

* 完整可运行
* 使用 pathlib 处理路径
* 包含 `if __name__ == "__main__":`
* 结构清晰，模块化，包含函数：

  * `load_csv()`
  * `build_queries()`
  * `run_retrieval_batch()`
  * `save_parquet()`
  * `main()`

---

# 📌 如果需要，也请让模型自动帮我生成 example 输出格式，例如：

```json
{
  "prompt": "...",
  "A": "...",
  "B": "...",
  "C": "...",
  "D": "...",
  "E": "...",
  "answer": "A",
  "C1": "...",
  "C2": "...",
  "C3": "..."
}
```

---

# 🎯 **结束语**

严格按照以上规范生成完整 Python 脚本。

---

# 👍 如果你愿意，我可以进一步：

* 给你一个 **确定可运行的最终 Python 脚本**
* 自动推测你的 `retrieval_pipeline.py` 的 API，并生成适配代码
* 加上 try/except 和 logging
* 生成 GPU 加速版本

你需要我继续帮你直接写完整脚本吗？
