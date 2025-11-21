from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")