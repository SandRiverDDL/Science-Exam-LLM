# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v3", trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-reranker-v3", trust_remote_code=True)