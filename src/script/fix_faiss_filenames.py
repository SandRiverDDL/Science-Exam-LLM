"""修复FAISS文件命名

重命名错误命名的文件
"""
import os
import shutil
from pathlib import Path

# 配置路径
project_root = Path(__file__).parent.parent.parent
base_dir = project_root / 'data' / 'faiss'

print("=" * 80)
print("FAISS文件命名修复工具")
print("=" * 80)

# 检查源文件
wrong_name = base_dir / 'qwen3_fp16_ip.faiss'
correct_name = base_dir / 'qwen3_fp16_ip_chunk_ids.json'

print(f"\n检查文件...")
print(f"  错误的文件名: qwen3_fp16_ip.faiss")
print(f"  正确的文件名: qwen3_fp16_ip_chunk_ids.json")

if not wrong_name.exists():
    print(f"\n❌ 源文件不存在: {wrong_name}")
    exit(1)

if correct_name.exists():
    print(f"\n⚠️  目标文件已存在: {correct_name}")
    print(f"   请先删除旧文件或确认是否要覆盖")
    exit(1)

# 显示文件信息
wrong_size = os.path.getsize(wrong_name)
print(f"\n文件信息:")
print(f"  源文件大小: {wrong_size / (1024**2):.2f} MB")

# 执行重命名
print(f"\n执行重命名...")
try:
    shutil.move(str(wrong_name), str(correct_name))
    print(f"✅ 重命名成功!")
    print(f"   {wrong_name.name}")
    print(f"   ↓")
    print(f"   {correct_name.name}")
except Exception as e:
    print(f"❌ 重命名失败: {e}")
    exit(1)

# 验证
if correct_name.exists() and not wrong_name.exists():
    print(f"\n✅ 验证成功 - 文件已正确重命名")
else:
    print(f"\n❌ 验证失败 - 重命名可能未完成")
    exit(1)

print("\n" + "=" * 80)
print("修复完成！")
print("=" * 80)
print("\n现在您需要:")
print("1. 找到实际的 qwen3_fp16_ip.faiss 索引文件")
print("2. 确认其位置或重新生成它")
