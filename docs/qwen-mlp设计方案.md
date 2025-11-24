这个需求非常专业，涉及到了 **自定义模型架构**、**KV Cache 显存优化**、**特征工程 (Contrastive Features)** 以及 **现代化的训练框架**。

为了让 AI IDE（如 Cursor、Windsurf）一次性写出高质量的代码，我们需要将指令拆解为**数据流（Data Flow）**、**模型架构（Architecture）**、**训练逻辑（Training Logic）** 和 **配置（Configuration）** 四个维度。

我修正了一个关键点：通常我们不会把 **Logits**（词表大小，约 15万维）输入 MLP，而是提取 **Last Hidden State**（约 4096维）。如果你一定要用 Logits，通常是指特定 Token（如 "Yes"）的标量分数。**为了让 MLP 有效工作，我建议使用 Hidden State，因为它包含了语义信息。** 下面的指令采用 Hidden State 方案，这更符合 SOTA 逻辑。

以下是优化后的 Prompt，你可以直接复制给 AI IDE：

---

### 📋 AI IDE 指令 Prompt

**Role:** Senior PyTorch Engineer & LLM Architect
**Task:** Implement a custom **Qwen-based Pointwise RAG Scoring Model** using PyTorch Lightning.

#### 1. 核心架构逻辑 (Model Architecture)
实现一个自定义的 `LightningModule` (`QwenPointwiseMLP`)，逻辑如下：
*   **Backbone**: Qwen2.5/3 (Frozen parameters).
*   **Input**:
    *   `context` + `question` (作为 Shared Prefix).
    *   5 个 `options` (A, B, C, D, E).
*   **KV Cache Optimization (关键)**:
    *   **Step 1**: 先对 `Shared Prefix` (Context + Question) 进行一次 Forward，获取 `past_key_values` (KV Cache)。
    *   **Step 2**: 循环 5 次（针对 5 个选项）。利用 Step 1 的 `past_key_values`，只对 `Option` 部分进行 Forward。
    *   **Step 3**: 提取每个 Option 最后一个 Token 的 **Hidden State** (Vector $H \in \mathbb{R}^{D}$).
*   **Feature Engineering (Contextual Mixing)**:
    *   对于第 $i$ 个选项，构建 MLP 的输入特征 $F_i$。
    *   策略：$F_i = \text{Concat}(H_i, \text{Mean}(\{H_j | j \neq i\}))$。
    *   即：将**当前选项的 Hidden State** 与 **其他 4 个选项的 Hidden State 均值** 拼接。这引入了对比信息。
*   **MLP Head**:
    *   Input: $2 \times \text{Hidden\_Dim}$ (因为是拼接).
    *   Structure: `Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> Output(1)`.
    *   Output: 一个标量 Score $S_i$。

#### 2. 训练逻辑 (Training Logic)
*   **Framework**: 使用 `pytorch_lightning`.
*   **Loss Function**:
    *   将 5 个选项的 Score $[S_A, S_B, S_C, S_D, S_E]$ 视为 Logits。
    *   使用 `nn.CrossEntropyLoss`。
    *   Label 为正确选项的 Index (0-4)。
*   **Optimizer**: `torch.optim.AdamW`.
*   **Scheduler**: `torch.optim.lr_scheduler.OneCycleLR` (Total steps need to be calculated dynamically based on dataset size).
*   **Tricks**:
    *   在 MLP 输入前加入 `LayerNorm` 以稳定训练。
    *   使用 `Dropout` 防止过拟合。

#### 3. 配置管理 (Configuration)
所有超参数必须读取自 `config.yaml` 中的 `qwen_mlp` 节点。需要暴露的参数包括：
```yaml
qwen_mlp:
  base_model: "Qwen/Qwen2.5-7B"
  hidden_dim: 4096       # Qwen hidden size
  mlp_hidden_dim: 1024   # MLP 中间层
  dropout: 0.1
  lr: 5.0e-5
  weight_decay: 0.01
  max_epochs: 5
  batch_size: 4          # 这里的 BS 是指有多少个"题目"，实际显存占用是 BS * 5
  use_4bit: true         # 是否使用 QLoRA/4bit 加载 Base Model
  gradient_clip_val: 1.0
```

#### 4. 代码文件结构要求
*   `src/modeling/qwen_mlp.py`: 定义 LightningModule 和 MLP 架构。
*   `src/dataset/datamodule.py`: 定义 LightningDataModule，负责数据加载和 Tokenization。
*   `src/train/train_lightning.py`: 训练入口脚本。

# System Prompt (可选，如果是 Instruct 模型)
# system = "You are a scientist."

# User Input
text = f"""Background:
{context}

Question:
{question}

Answer: {option_text}""" # <--- 没有任何多余的 Yes/No 提问，直接作为陈述句结束
提取策略：提取 text 的最后一个 Token 的 Hidden State。
⚡ 最终给 AI IDE 的补充指令 (Copy 这段补充进去)
为了让 IDE 写的代码更完美，请把下面这段加到之前的 Prompt 末尾：
#### 5. Additional Architecture & Training Details (Refined)
MLP Structure: Use a "Funnel" design with Pre-Norm.
LayerNorm(input_dim) -> Linear(input_dim -> 2048) -> GELU -> Dropout(0.2) -> Linear(2048 -> 1).
Initialize weights using kaiming_normal_.
Prompt Strategy: Use Sentence Completion format.
Construct input as: "{Context}\nQuestion: {Question}\nAnswer: {Option}".
Extract the hidden state of the very last token of the Option.
Precision: Use bf16 context for training.
Optimization:
Since batch_size might be small (e.g., 2) on a single GPU due to KV caching 5 options, implement gradient_accumulation_steps to achieve an effective batch size of 32 or 64.
Use EarlyStopping monitoring validation loss.


def format_input(context_list, question, option_text):
    # 1. System Prompt (激活专家模式)
    system_content = "You are a scientific reasoning expert. Determine if the Candidate Answer is the correct response to the Question based on the Evidence."
    
    # 2. Context 拼接 (优化分隔符)
    # Qwen 对 "Context 1:", "Context 2:" 或者 XML <doc> 标签很敏感
    formatted_contexts = []
    for idx, ctx in enumerate(context_list):
        formatted_contexts.append(f"Evidence {idx+1}:\n{ctx}")
    context_str = "\n\n".join(formatted_contexts)
    
    # 3. User Content (结构化输入)
    # 这里的关键是：把 Option 包装成一个"待验证的陈述"
    user_content = f"""### Evidence:
{context_str}

### Question:
{question}

### Candidate Answer:
{option_text}"""

    # 4. 应用 Chat Template (关键步骤)
    # 我们利用 apply_chat_template 自动处理 <|im_start|> 等特殊 token
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    # 5. 这一步是 Trick 的核心：
    # 我们不让 Chat Template 自动添加 "assistant" 的引导头
    # 而是让 input_ids 的最后一个 token 就停在 user_content 的最后一个字（即 option_text 的末尾）
    # 这样提取出来的 Hidden State 就是模型读完 Option 后的瞬间反应
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return full_text