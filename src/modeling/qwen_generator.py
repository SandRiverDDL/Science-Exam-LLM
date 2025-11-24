"""Qwen Answer Generation Model

支持两种模式：
1. Zero-shot generation：直接使用 LLM 生成答案
2. Hidden state extraction + MLP：提取隐藏状态，连接 MLP 进行 LoRA 微调
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import numpy as np


class QwenGenerator:
    """Qwen 答案生成器"""
    
    def __init__(
        self,
        model_id: str = "ISTA-DASLab/Qwen3-8B-Instruct-FPQuant-QAT-MXFP4-TEMP",
        device_map: str = "auto",
        max_new_tokens: int = 1,
        gen_temperature: float = 0.0,
        do_sample: bool = False,
        trust_remote_code: bool = True,
        extract_hidden_states: bool = False
    ):
        """初始化 Qwen 生成器
        
        Args:
            model_id: 模型 ID
            device_map: 设备映射
            max_new_tokens: 最大生成 token 数
            gen_temperature: 生成温度（0.0 = 贪婪解码）
            do_sample: 是否采样
            trust_remote_code: 是否信任远程代码
            extract_hidden_states: 是否提取隐藏状态（用于 MLP）
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.gen_temperature = gen_temperature
        self.do_sample = do_sample
        self.extract_hidden_states = extract_hidden_states
        
        print(f"[QwenGenerator] 加载模型: {model_id}...")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16
        )
        
        self.model.eval()
        print(f"[QwenGenerator] 模型加载完成")
        print(f"[QwenGenerator] 模型上的设备：{next(self.model.parameters()).device}")
        
        # 获取 A/B/C/D/E 的 token IDs（用于处理）
        self.abcde_ids = self._get_abcde_token_ids()
    
    def _get_abcde_token_ids(self) -> List[List[int]]:
        """获取 [A], [B], [C], [D], [E] 对应的 Token ID 列表
        
        Returns:
            [[tid_A], [tid_B], [tid_C], [tid_D], [tid_E]] 格式的列表
        """
        ids = []
        
        # 遍历 A 到 E
        for char in ['A', 'B', 'C', 'D', 'E']:
            # 尝试编码单个字母
            token_id = self.tokenizer.encode(char, add_special_tokens=False)
            
            if len(token_id) == 1:
                # 单个 token，直接使用
                ids.append(token_id)
            else:
                # 尝试编码带空格的版本
                token_id_space = self.tokenizer.encode(' ' + char, add_special_tokens=False)
                if len(token_id_space) == 1:
                    ids.append(token_id_space)
                else:
                    # 如果都失败，尝试不带空格但作为独立word
                    print(f"⚠️ 警告: 字符 {char} 编码为多个 Token，尝试备选方案")
                    # 使用第一个 token 作为备选
                    if token_id:
                        ids.append([token_id[0]])
        
        print(f"[QwenGenerator] force_words_ids 已初始化: {ids}")
        return ids
    
    def build_prompt(
        self,
        question: str,
        options: Dict[str, str],
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        构建 prompt（SOTA 版本）
        """
        # 最强 SOTA baseline prompt
        user_prompt = f"""
            You MUST answer strictly based ONLY on the provided context below.
            If the context does not contain enough information, you MUST guess.
            DO NOT output <think>. DO NOT output chain-of-thought. 
            NEVER reveal reasoning. NEVER output anything except a single letter.
            Your output MUST be exactly ONE token: A, B, C, D, or E.

            ### Context:
            {context}

            ### Question:
            {question}

            ### Options:
            A: {options.get('A', '')}
            B: {options.get('B', '')}
            C: {options.get('C', '')}
            D: {options.get('D', '')}
            E: {options.get('E', '')}

            ### Output Format:
            Return ONLY the letter of the correct option (A, B, C, D, or E).
            No explanations.

            Answer:
            """.strip()

        system_prompt_fixed = (
            "You are an expert scientific question solver. "
            "Follow instructions EXACTLY. Never output <think>."
        )
        
        messages = [
            {"role": "system", "content": system_prompt_fixed},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        batch_size: int = 1
    ) -> List[str]:
        """
        批量生成答案（zero-shot）
        """
        all_responses = []

        # allowed tokens = ABCDE
        allowed_tokens = ["A", "B", "C", "D", "E"]
        allowed_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in allowed_tokens]

        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.model.device)

            # logits mask: 非 ABCDE token 一律 -inf
            vocab_size = self.model.config.vocab_size
            mask = torch.full((vocab_size,), float('-inf'), device=self.model.device)
            for tid in allowed_ids:
                mask[tid] = 0.0  # 允许的 token

            def logits_processor(input_ids, logits):
                return logits + mask

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,     # 只生成一个 token
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                logits_processor=[logits_processor]
            )

            for i, output in enumerate(outputs):
                input_len = inputs['input_ids'][i].shape[0]
                generated = output[input_len:]

                # decode
                response = self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True
                )

                # parse
                answer = self.parse_answer(response)
                all_responses.append(answer)

        return all_responses
    
    # @torch.no_grad()
    # def extract_hidden_states(
    #     self,
    #     prompts: List[str],
    #     batch_size: int = 1
    # ) -> Tuple[np.ndarray, List[str]]:
    #     """提取隐藏状态（用于 MLP 微调）
        
    #     Args:
    #         prompts: prompt 列表
    #         batch_size: 批处理大小
        
    #     Returns:
    #         (hidden_states, responses) 元组
    #         - hidden_states: [num_samples, hidden_dim]
    #         - responses: 生成的答案列表
    #     """
    #     all_hidden_states = []
    #     all_responses = []
        
    #     for batch_start in range(0, len(prompts), batch_size):
    #         batch_end = min(batch_start + batch_size, len(prompts))
    #         batch_prompts = prompts[batch_start:batch_end]
            
    #         # Tokenize
    #         inputs = self.tokenizer(
    #             batch_prompts,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=4096
    #         ).to(self.model.device)
            
    #         # Forward pass – 简化方式：直接在forward中提取隐藏状态
    #         # 设置output_hidden_states=True 即可
    #         outputs = self.model(
    #             **inputs,
    #             output_hidden_states=True
    #         )
            
    #         # 提取最后一层的隐藏状态
    #         # outputs.hidden_states[-1] 是最后一层， 形状 [batch, seq_len, hidden_dim]
    #         # 取最后一个 token 的表示作为每个样本的特征
    #         batch_hidden = outputs.hidden_states[-1][:, -1, :].cpu().numpy()  # [batch, hidden_dim]
    #         all_hidden_states.append(batch_hidden)
            
    #         # 生成回复：使用generate方法产生新的token
    #         with torch.inference_mode():
    #             generation_outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=2,  # 最多 2 个 token
    #                 temperature=0.0,   # 贪心解码
    #                 do_sample=False,   # 禁用采样
    #                 pad_token_id=self.tokenizer.pad_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id
    #             )
            
    #         # 解码生成部分
    #         for i, seq in enumerate(generation_outputs):
    #             input_length = inputs['input_ids'][i].shape[0]
    #             generated_tokens = seq[input_length:]
    #             response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
    #             # 使用 parse_answer 严格提取字母
    #             answer = self.parse_answer(response)
    #             all_responses.append(answer)
        
    #     # Concatenate all batches
    #     hidden_states = np.vstack(all_hidden_states)
        
    #     return hidden_states, all_responses
    
    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        """从生成的回复中严格提取单个字母
        
        策略：
        1. 移除一切 <...> 之类的特殊标记
        2. 提取第一个 A-E 字母
        3. 永不返回 None（兴底默认 A）
        
        Args:
            response: 模型生成的回复
        
        Returns:
            解析出的答案（'A', 'B', 'C', 'D', 'E'）
        """
        import re
        
        response = response.strip()
        
        # 策略 1：移除 <...> 等三尖括号内容 (例如 <think>...)</think>)
        response = re.sub(r'<[^>]+>', '', response)
        response = response.strip()
        
        # 策略 2：移除所有不是 A-E 的字符（但保留空格供检查）
        # 首先提取所有 A-E
        letters = re.findall(r'[A-E]', response)
        
        if letters:
            # 返回第一个找到的字母
            return letters[0]
        
        # 策略 3：如果一个字母都没找到，默认 A（永不返回 None）
        return 'A'
