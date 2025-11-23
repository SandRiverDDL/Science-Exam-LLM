import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "infgrad/Jasper-Token-Compression-600M",
    model_kwargs={"trust_remote_code": True},
    trust_remote_code=True
)

text = ["This is a test sentence."] * 64

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model.encode(text, compression_ratio=0.3)

print(prof.key_averages().table(sort_by="cuda_time_total"))