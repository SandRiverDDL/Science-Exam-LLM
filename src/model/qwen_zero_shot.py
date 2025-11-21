import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class QwenZeroShotClassifier:
    def __init__(self, model_id: str, device_map: str = "auto", trust_remote_code: bool = True):
        hf_token = os.environ.get("huggingface_token", None)
        # 按你的 download_model.py 做 4bit QAT 加载
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            use_auth_token=hf_token,
            ignore_mismatched_sizes=True
        )
        self.device = self.model.device

        # 预计算 '0'、'1' token id
        self.id_0 = self.tokenizer.encode("0", add_special_tokens=False)[0]
        self.id_1 = self.tokenizer.encode("1", add_special_tokens=False)[0]

    def test_text_output(self):
        prompt = "你好！请输出一句简短的中文问候。"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        print(self.tokenizer.decode(out[0], skip_special_tokens=True))

    def _build_prefix(self, question: str):
        # 系统提示，限制仅输出 0/1
        sys_prompt = "你是一个严格的评分器，只能输出 0 或 1。"
        tpl = f"{sys_prompt}\n问题：{question}\n回答："
        prefix_ids = self.tokenizer(tpl, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

        with torch.no_grad():
            out = self.model(input_ids=prefix_ids, use_cache=True)
        past_kv = out.past_key_values
        return past_kv

    def classify(self, question: str, answer: str, use_kv_cache_prefix: bool = True) -> int:
        if use_kv_cache_prefix:
            past_kv = self._build_prefix(question)
            ans_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=ans_ids, past_key_values=past_kv, use_cache=True)
            next_logits = out.logits[:, -1, :]  # 下一个 token 的分布
        else:
            # 不复用 KV，直接拼接生成下一个 token
            sys_prompt = "你是一个严格的评分器，只能输出 0 或 1。"
            tpl = f"{sys_prompt}\n问题：{question}\n回答：{answer}\n结论："
            full_ids = self.tokenizer(tpl, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=full_ids, use_cache=True)
            next_logits = out.logits[:, -1, :]

        probs = torch.softmax(next_logits, dim=-1)
        p0 = probs[0, self.id_0].item()
        p1 = probs[0, self.id_1].item()
        return 1 if p1 >= p0 else 0

    def _mcq_allowed_token_ids(self):
        letters = ["A", "B", "C", "D", "E"]
        allowed = {}
        for L in letters:
            ids_plain = self.tokenizer.encode(L, add_special_tokens=False)
            ids_space = self.tokenizer.encode(" " + L, add_special_tokens=False)
            s = set()
            if ids_plain:
                s.add(ids_plain[0])
            if ids_space:
                s.add(ids_space[0])
            allowed[L] = list(s)
        return allowed

    def predict_mcq(self, question: str, A: str, B: str, C: str, D: str, E: str) -> str:
        tmpl = (
            "You are a system that outputs ONLY one of A, B, C, D, E.\n"
            "No explanation. No reasoning. Output must be exactly one letter.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Options:\n"
            f"A: {A}\n"
            f"B: {B}\n"
            f"C: {C}\n"
            f"D: {D}\n"
            f"E: {E}\n\n"
            "Answer (A/B/C/D/E):"
        )
        inputs = self.tokenizer(tmpl, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=inputs.input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        allowed = self._mcq_allowed_token_ids()
        best_L, best_p = "A", -1.0
        for L, ids in allowed.items():
            p = 0.0
            for tid in ids:
                p += probs[0, tid].item()
            if p > best_p:
                best_p = p
                best_L = L
        return best_L