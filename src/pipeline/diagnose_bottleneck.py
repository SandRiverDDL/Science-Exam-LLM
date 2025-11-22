"""诊断embedding构建瓶颈

运行此脚本来确定是CPU还是GPU瓶颈
"""
import time
import psutil
import GPUtil
from threading import Thread

def monitor_resources(duration=60, interval=0.5):
    """监控CPU和GPU使用率
    
    Args:
        duration: 监控时长（秒）
        interval: 采样间隔（秒）
    """
    cpu_usage = []
    gpu_usage = []
    
    start_time = time.time()
    print(f"开始监控 {duration} 秒...\n")
    
    while time.time() - start_time < duration:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=interval)
        cpu_usage.append(cpu_percent)
        
        # GPU使用率
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                gpu_usage.append(gpu_percent)
            else:
                gpu_percent = 0
        except:
            gpu_percent = 0
            gpu_usage.append(0)
        
        # 实时显示
        print(f"\r  CPU: {cpu_percent:5.1f}% | GPU: {gpu_percent:5.1f}%", end='', flush=True)
    
    print("\n\n分析结果:")
    print("=" * 50)
    
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    avg_gpu = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0
    
    print(f"平均CPU使用率: {avg_cpu:.1f}%")
    print(f"平均GPU使用率: {avg_gpu:.1f}%")
    print()
    
    # 判断瓶颈
    if avg_cpu > 80 and avg_gpu < 50:
        print("🔴 结论: CPU瓶颈")
        print("   GPU在等待CPU准备数据")
        print("   建议: 优化字符串处理、使用Arrow、异步pipeline")
    elif avg_cpu < 50 and avg_gpu > 80:
        print("🟢 结论: GPU瓶颈")
        print("   CPU准备数据速度够快")
        print("   建议: 增大batch_size、使用更大模型")
    elif avg_cpu > 80 and avg_gpu > 80:
        print("🟡 结论: CPU+GPU双瓶颈")
        print("   两者都在满负载")
        print("   建议: 检查是否有IO瓶颈")
    else:
        print("⚪ 结论: 未充分利用资源")
        print("   可能存在IO等待或其他瓶颈")
    
    print("=" * 50)


if __name__ == "__main__":
    print("🔍 资源使用率监控工具")
    print("请在另一个终端运行embedding构建脚本，然后立即运行此脚本\n")
    
    try:
        monitor_resources(duration=60)
    except KeyboardInterrupt:
        print("\n\n监控已停止")
