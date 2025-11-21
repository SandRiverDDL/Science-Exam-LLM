from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("BAAI/bge-m3")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [4, 4]

# [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)

