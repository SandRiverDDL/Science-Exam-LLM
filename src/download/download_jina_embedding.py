# Load model directly


from transformers import AutoModel
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v4", trust_remote_code=True, dtype="auto")