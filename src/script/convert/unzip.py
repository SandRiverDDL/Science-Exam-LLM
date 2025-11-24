import os
import lz4.frame

def decompress_all_lz4(dir_path):
    for name in os.listdir(dir_path):
        if name.endswith(".lz4"):
            in_path = os.path.join(dir_path, name)
            out_path = os.path.join(dir_path, name[:-4])  # 去掉 .lz4

            with lz4.frame.open(in_path, 'rb') as f_in:
                data = f_in.read()

            with open(out_path, 'wb') as f_out:
                f_out.write(data)

            print(f"解压: {in_path} -> {out_path}")

if __name__ == "__main__":
    decompress_all_lz4("data/faiss")