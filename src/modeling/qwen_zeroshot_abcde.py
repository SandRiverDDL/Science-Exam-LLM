import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import re

class QwenZeroShot:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B", # 假设你用的是 Qwen2.5-Base (Qwen3 尚未发布或为内部代号)
        device_map: str = "auto",
    ):
        print(f"[Init] Loading Base model: {model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            padding_side='left'
        )
        # Base 模型通常没有 pad_token，手动指定 eos 为 pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
        print("[Init] Ready.")

    def build_prompt(self, question: str, options: Dict[str, str], context: str) -> str:
        """
        针对 Base 模型的 Prompt 策略：
        Base 模型不是对话机器人，它只是“接着写”。
        最好的 Prompt 是伪装成一份已经写了一半的试卷。
        """
        # 这是一个 Few-shot 风格的 header，告诉 Base 模型它正在做题
        prompt = f"""Background Information:
            {context}

            Question:
            {question}

            Options:
            A. {options.get("A", "")}
            B. {options.get("B", "")}
            C. {options.get("C", "")}
            D. {options.get("D", "")}
            E. {options.get("E", "")}

            The correct answer is:""" 
        # Base 模型看到 "The correct answer is:" 会倾向于直接补全字母
        return prompt

    @torch.no_grad()
    def generate(self, prompts: List[str], batch_size: int = 1) -> List[str]:
        results = []
        
        print(f"\n[Debug] 正在生成... (Batch Size: {batch_size})")

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.model.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,       # ✅ 放宽到 5，看看它想说什么
                do_sample=False,        # Greedy search，最稳
                temperature=None,       # Greedy 模式不需要 temp
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                # ❌ 移除了 logits_processor，允许输出 <think> 或其他任何词
            )

            # Decode
            for j, out in enumerate(outputs):
                # 只解码新生成的部分
                input_len = inputs["input_ids"][j].shape[0]
                generated_ids = out[input_len:]
                
                # 打印原始 Token ID，看看有没有 <think> (通常 ID 很大)
                # print(f"[Raw IDs] {generated_ids.tolist()}")
                
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=False) # ✅ 保留特殊字符
                
                # print(f"--------------------------------------------------")
                # print(f"[Model Output Raw]: {repr(text)}") # 使用 repr 打印，能看到 \n 和空格
                
                results.append(self._parse_letter(text))

        return results

    def _parse_letter(self, text: str) -> str:
        """宽松解析：尝试找到第一个出现的 A-E"""
        # 清理 <think> 等标签
        clean_text = text.replace("<think>", "").replace("</think>", "").strip()
        
        # 找第一个字母
        match = re.search(r"[A-E]", clean_text)
        if match:
            return match.group(0)
        
        return "B" # 兜底