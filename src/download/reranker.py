# !pip install llama-cpp-python
import os
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="jinaai/jina-reranker-v3-GGUF",
	filename="jina-reranker-v3-BF16.gguf",
)
output = llm(
	"Once upon a time,",
	max_tokens=512,
	echo=True
)
print(output)