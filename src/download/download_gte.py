# Load model directly
from transformers import AutoModel
import os

os.environ['huggingface_token'] = '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'
model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True, dtype="auto")