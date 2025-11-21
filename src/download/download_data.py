import kagglehub

# Download latest version
path = kagglehub.dataset_download("bwandowando/wikipedia-index-and-plaintext-20230801")

print("Path to dataset files:", path)