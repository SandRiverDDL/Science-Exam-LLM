import os
import csv
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..') # 假设 core 和 pipeline 是兄弟目录
project_root = os.path.join(src_dir, '..')
sys.path.append(os.path.abspath(src_dir)) 
sys.path.append(os.path.abspath(project_root)) 
from core.config import Config
from modeling.qwen_zero_shot import QwenZeroShotClassifier

def run_train_eval():
    cfg = Config()
    train_csv = cfg.get("kaggle", "train_csv")
    out_csv = cfg.get("kaggle", "output_submission")

    clf = QwenZeroShotClassifier(
        model_id=cfg.qwen["model_id"],
        device_map=cfg.qwen.get("device_map", "auto"),
        trust_remote_code=cfg.qwen.get("trust_remote_code", True),
    )

    rows = []
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    correct = 0
    total = 0

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prediction"])
        for row in rows:
            q = row["prompt"]
            A = row["A"]
            B = row["B"]
            C = row["C"]
            D = row["D"]
            E = row["E"]
            pred = clf.predict_mcq(q, A, B, C, D, E)
            writer.writerow([row["id"], pred])
            if "answer" in row and row["answer"]:
                total += 1
                correct += 1 if pred.strip() == row["answer"].strip() else 0

    acc = (correct / total) if total > 0 else 0.0
    print(f"Zero-shot accuracy on train.csv: {acc:.4f}")

if __name__ == "__main__":
    run_train_eval()