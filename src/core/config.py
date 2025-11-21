import os
import yaml

class Config:
    def __init__(self, path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config.yaml")):
        with open(os.path.abspath(path), "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, *keys, default=None):
        node = self.cfg
        for k in keys:
            if node is None or k not in node:
                return default
            node = node[k]
        return node

    @property
    def data(self):
        return self.cfg["data"]

    @property
    def embedding(self):
        return self.cfg["embedding"]

    @property
    def index(self):
        return self.cfg["index"]

    @property
    def retrieval(self):
        return self.cfg["retrieval"]

    @property
    def reranker(self):
        return self.cfg["reranker"]

    @property
    def qwen(self):
        return self.cfg["qwen"]

    @property
    def zero_shot(self):
        return self.cfg["zero_shot"]