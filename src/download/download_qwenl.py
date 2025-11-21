import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

model_id = "ISTA-DASLab/Qwen3-8B-Instruct-FPQuant-QAT-MXFP4-TEMP"
os.environ['huggingface_token'] = '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'
# 1. 定义 4-bit 量化配置
# 即使模型已量化，通常也需要此配置来加载 BitsAndBytes 格式或兼容设置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", # 或 "fp4"，具体取决于模型作者使用的 MXFP4 实现
    bnb_4bit_compute_dtype=torch.bfloat16 # 使用 bf16 进行计算，加速且保持精度
)

# 2. 自动下载并加载模型
# AutoModel会自动识别并下载所有 safetensors 分片文件 (model-00001-of-00004.safetensors等)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # 自动分配到可用的GPU上
    trust_remote_code=True # Qwen3 可能需要
)

# 3. 自动下载并加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

# 现在 model 就可以在你的 GPU 上使用了
print(f"Model loaded successfully on device: {model.device}")