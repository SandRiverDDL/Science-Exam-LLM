# Load model directly
from transformers import AutoModel
import os


model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True, dtype="auto")