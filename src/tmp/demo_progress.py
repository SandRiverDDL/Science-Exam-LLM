"""演示改进的进度显示和统计"""
import time

def demo_progress_display():
    """演示使用 \r 覆盖的进度显示"""
    print("=" * 80)
    print("演示：进度信息覆盖（使用 \\r）")
    print("=" * 80)
    print("\n预处理中...")
    
    kept = 0
    for i in range(1, 1001):
        # 模拟保留率
        if i % 10 == 0:
            kept += 1
        
        # 每100条显示一次，使用 \r 覆盖
        if i % 100 == 0:
            print(f"\r  已处理 {i:,} 条文档，保留 {kept:,} 条", end='', flush=True)
            time.sleep(0.1)  # 模拟处理时间
    
    print("\n\n处理完成！" + "=" * 80)


def demo_checkpoint_resume():
    """演示断点续跑的统计累加"""
    print("\n" + "=" * 80)
    print("演示：断点续跑的统计累加")
    print("=" * 80)
    
    # 模拟第一次运行
    print("\n[第一次运行]")
    print("[resume] 加载断点：已处理 0 个文档，之前保留 0 个")
    stats_run1 = {
        'total_docs': 1000,
        'kept_docs': 526,
        'total_chunks': 2100,
    }
    print(f"  本次保留: {stats_run1['kept_docs']:,}")
    print(f"  累计保留: {stats_run1['kept_docs']:,} (52.6%)")
    
    # 模拟第二次运行（断点续跑）
    print("\n[第二次运行（中断后恢复）]")
    initial_kept = stats_run1['kept_docs']
    print(f"[resume] 加载断点：已处理 700 个文档，之前保留 {initial_kept:,} 个")
    stats_run2 = {
        'total_docs': 1000 + 700,  # 加上新增
        'kept_docs': 200,  # 本次新增
        'total_chunks': 800,  # 本次新增
    }
    total_kept = stats_run2['kept_docs'] + initial_kept
    print(f"  本次保留: {stats_run2['kept_docs']:,}")
    print(f"  累计保留: {total_kept:,} ({total_kept/1000*100:.1f}%)")
    
    print("\n✅ 注意：累计保留数 = 本次保留 + 之前已保留")


def demo_token_warning():
    """解释 Token 超长警告"""
    print("\n" + "=" * 80)
    print("解释：Token 超长警告")
    print("=" * 80)
    
    print("""
警告信息：
  Token indices sequence length is longer than the specified maximum sequence length
  for this model (5947 > 512). Running this sequence through the model will result
  in indexing errors

原因：
  ✓ 这是在 tokenizer.encode() 時，對整個原始文檔進行 tokenization 時發出的
  ✓ 警告是正常的，代表文檔很長

為什麼不是問題：
  ✓ 在 chunking 階段，每個子 chunk 都會被截斷到 128 tokens
  ✓ 父 chunk 存儲的是索引範圍，不會直接輸入模型
  ✓ 實際 embedding 時，每個 chunk ≤ 128 tokens，遠小於 512 的限制

例子：
  原始文檔: 5947 tokens ⚠️
  ↓ chunking
  子chunk 1: 128 tokens ✓
  子chunk 2: 128 tokens ✓
  ...
  各個chunk都在安全範圍內
    """)


if __name__ == "__main__":
    demo_progress_display()
    demo_checkpoint_resume()
    demo_token_warning()
    
    print("\n" + "=" * 80)
    print("✅ 所有改进演示完成！")
    print("=" * 80)
