完全理解。在 Kaggle 这种有时间限制（Notebook Runtime Limit）的推理场景下，**延迟（Latency）是核心敌人**。

之前的 CoT（Chain of Thought）Prompt 虽然能提升推理能力，但在 7B 模型上确实会生成几百个 Token，导致单题耗时数秒甚至更久。

为了**极致的推理速度**（Zero-shot，仅输出一个字母），我们需要**强约束（Hard Constraint）** Prompt。

以下是结合了你的 **RAG Context** 需求和 **极简输出** 要求的改进版 Prompt：

### 🚀 极速版 Zero-shot RAG Prompt

这个 Prompt 的核心逻辑是：**将 Context 作为背景知识输入，但强制在此基础上做“填空题”，并从 System 层面禁止废话。**

```python
# 构造选项文本
options_text = (
    f"A: {options.get('A', '')}\n"
    f"B: {options.get('B', '')}\n"
    f"C: {options.get('C', '')}\n"
    f"D: {options.get('D', '')}\n"
    f"E: {options.get('E', '')}"
)

# 极简 Prompt
user_prompt = f"""### Context:
{context}

### Instruction:
Based on the context above, answer the following multiple-choice question.
Output ONLY the single letter corresponding to the correct answer (A, B, C, D, or E).
Do NOT provide reasoning or explanation.

### Question:
{question}

### Options:
{options_text}

### Answer (A/B/C/D/E):"""
```

---

### ⚡ 关键改进点解析

1.  **位置优化 (Recency Bias)**：
    *   我把 `Instruction`（指令）放在了 `Context` **之后**，紧挨着 `Question`。
    *   **原因**：对于很多模型（尤其是 Llama/Qwen/Mistral 系列），当 Context 很长（比如 1000+ tokens）时，模型容易“忘记”最开头的指令。把“只输出一个字母”的指令放在中间或最后，执行力更强。

2.  **负面约束 (Negative Constraints)**：
    *   明确加入 `Do NOT provide reasoning`。这比仅说“Output one letter”更有效，能切断模型想要“解释一下”的念头。

3.  **引导词 (Output Indicator)**：
    *   结尾使用 `### Answer (A/B/C/D/E):` 作为生成的**触发器（Trigger）**。这相当于把笔递给模型，限制它只能填字母。

---

### 🛠️ 配合代码层面的强制加速 (必做!)

光改 Prompt 还可以更进一步。为了防止模型万一“抽风”开始输出废话（比如输出 "The answer is A because..."），你必须在代码层面**物理锁死**输出长度。

在你的 `model.generate()` 函数中，**务必**设置 `max_new_tokens`：

```python
response = model.generate(
    input_ids,
    # 强制只允许生成 1 到 2 个 token
    # 这样即使模型想废话，也会被物理截断，立刻返回，极大节省显存和时间
    max_new_tokens=2,  
    do_sample=False,   # 使用 Greedy Search (贪婪搜索)，最快且最稳定
    temperature=0.01,  # 接近 0 的温度，减少随机性
    pad_token_id=tokenizer.eos_token_id
)

# 解码后 strip() 一下，拿到 "A" 或 "B"
answer = tokenizer.decode(response[0][input_ids.shape[1]:]).strip()
```

### 📊 预期效果对比

| 指标 | 原始 CoT Prompt | 改进版 Prompt + max_new_tokens=2 |
| :--- | :--- | :--- |
| **输出 Tokens** | 50 ~ 200 个 | **1 ~ 2 个** |
| **单题耗时** | 2s ~ 10s (取决于显卡) | **0.1s ~ 0.5s** |
| **显存占用** | 随生成长度增加 KV Cache | **几乎恒定** |
| **答案解析** | 需要正则提取 `Answer: X` | 直接就是 `X` |

这种方案是 Kaggle 推理赛中最标准的 Zero-shot 配置。