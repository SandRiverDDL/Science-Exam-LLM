"""诊断FAISS文件类型

判断文件是FAISS索引还是chunk_id映射文件
"""
import os
import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


def read_file_header(file_path: str, num_bytes: int = 100) -> bytes:
    """读取文件头部
    
    Args:
        file_path: 文件路径
        num_bytes: 读取字节数
    
    Returns:
        文件头部字节
    """
    try:
        with open(file_path, 'rb') as f:
            return f.read(num_bytes)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return b''


def is_json_file(file_path: str) -> bool:
    """判断是否为JSON文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        是否为JSON文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试读取第一行判断是否以{ 或 [ 开头
            first_char = f.read(1)
            if first_char in ('{', '['):
                # 尝试解析为JSON
                f.seek(0)
                json.load(f)
                return True
    except Exception:
        pass
    return False


def is_faiss_index(file_path: str) -> bool:
    """判断是否为FAISS索引文件
    
    FAISS索引文件通常以特定的魔数开头
    - IndexFlatIP: 0x0a3e1337 (little-endian) 或类似的FAISS特定标识
    
    Args:
        file_path: 文件路径
    
    Returns:
        是否为FAISS索引文件
    """
    header = read_file_header(file_path, 16)
    
    if len(header) < 4:
        return False
    
    # FAISS索引的典型魔数
    # 检查是否包含FAISS特定的标识
    faiss_magic_numbers = [
        b'\x0a\x3e\x13\x37',  # IndexFlatIP等常见索引
        b'FAIS',              # 某些FAISS版本
    ]
    
    for magic in faiss_magic_numbers:
        if header.startswith(magic):
            return True
    
    # 另外，FAISS二进制文件通常包含大量非ASCII字符
    # 统计可打印字符比例
    printable_count = sum(1 for b in header if 32 <= b < 127 or b in (9, 10, 13))
    printable_ratio = printable_count / len(header) if header else 0
    
    # 如果可打印字符比例很低，很可能是二进制文件（FAISS索引）
    if printable_ratio < 0.3:
        return True
    
    return False


def analyze_faiss_file(file_path: str) -> dict:
    """分析FAISS文件
    
    尝试直接加载为FAISS索引
    
    Args:
        file_path: 文件路径
    
    Returns:
        分析结果字典
    """
    result = {
        'can_load_as_faiss': False,
        'index_type': None,
        'vector_count': None,
        'dimension': None,
        'error': None
    }
    
    try:
        import faiss
        
        # 尝试加载
        index = faiss.read_index(file_path)
        
        result['can_load_as_faiss'] = True
        result['index_type'] = type(index).__name__
        result['vector_count'] = index.ntotal if hasattr(index, 'ntotal') else None
        result['dimension'] = index.d if hasattr(index, 'd') else None
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    """主函数"""
    print("=" * 80)
    print("FAISS文件类型诊断工具")
    print("=" * 80)
    
    # 配置路径
    base_dir = project_root / 'data' / 'faiss'
    
    # 检查的文件列表
    files_to_check = [
        ('qwen3_fp16_ip.faiss', '可能是索引或映射'),
        ('qwen3_fp16_ip_chunk_ids.json', '应该是chunk_id映射'),
    ]
    
    results = {}
    
    for filename, description in files_to_check:
        file_path = base_dir / filename
        
        print(f"\n{'='*80}")
        print(f"检查文件: {filename}")
        print(f"描述: {description}")
        print(f"完整路径: {file_path}")
        print(f"{'='*80}")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在")
            results[filename] = {'exists': False}
            continue
        
        # 获取文件信息
        file_size = os.path.getsize(file_path)
        print(f"✓ 文件存在")
        print(f"✓ 文件大小: {file_size:,} 字节 ({file_size / (1024**2):.2f} MB)")
        
        # 检查文件头
        header = read_file_header(file_path, 100)
        if header:
            print(f"✓ 文件头（前64字节，十六进制）: {header[:64].hex()}")
            print(f"✓ 文件头（前50字符，ASCII）: {repr(header[:50])}")
        
        # 判断文件类型
        print(f"\n[诊断] 分析文件类型...")
        
        is_json = is_json_file(str(file_path))
        is_faiss = is_faiss_index(str(file_path))
        
        print(f"  JSON检测: {'✅ 是JSON文件' if is_json else '❌ 不是JSON文件'}")
        print(f"  FAISS检测: {'✅ 可能是FAISS索引' if is_faiss else '❌ 不像FAISS索引'}")
        
        # 尝试作为JSON加载
        if is_json or filename.endswith('.json'):
            print(f"\n[尝试] 作为JSON加载...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"  ✅ JSON加载成功")
                if isinstance(data, list):
                    print(f"  📊 列表类型，长度: {len(data)}")
                    if len(data) > 0:
                        print(f"  📊 第一个元素: {data[0]}")
                        print(f"  📊 最后一个元素: {data[-1]}")
                        # 检查是否看起来像chunk_id
                        if isinstance(data[0], str) and ':' in data[0]:
                            print(f"  ✅ 看起来像chunk_id列表")
                elif isinstance(data, dict):
                    print(f"  📊 字典类型，键数: {len(data)}")
                    keys = list(data.keys())[:5]
                    print(f"  📊 前5个键: {keys}")
                
                results[filename] = {
                    'exists': True,
                    'type': 'JSON',
                    'json_type': type(data).__name__,
                    'size_mb': file_size / (1024**2)
                }
            except Exception as e:
                print(f"  ❌ JSON加载失败: {e}")
                results[filename] = {
                    'exists': True,
                    'type': 'unknown',
                    'error': str(e),
                    'size_mb': file_size / (1024**2)
                }
        
        # 尝试作为FAISS加载
        if is_faiss or filename.endswith('.faiss'):
            print(f"\n[尝试] 作为FAISS索引加载...")
            faiss_result = analyze_faiss_file(str(file_path))
            
            if faiss_result['can_load_as_faiss']:
                print(f"  ✅ FAISS加载成功")
                print(f"  📊 索引类型: {faiss_result['index_type']}")
                print(f"  📊 向量总数: {faiss_result['vector_count']:,}")
                print(f"  📊 向量维度: {faiss_result['dimension']}")
                
                results[filename] = {
                    'exists': True,
                    'type': 'FAISS_INDEX',
                    'index_type': faiss_result['index_type'],
                    'vector_count': faiss_result['vector_count'],
                    'dimension': faiss_result['dimension'],
                    'size_mb': file_size / (1024**2)
                }
            else:
                print(f"  ❌ FAISS加载失败")
                print(f"  错误: {faiss_result['error']}")
                if 'type' not in results.get(filename, {}):
                    results[filename] = {
                        'exists': True,
                        'type': 'unknown',
                        'error': faiss_result['error'],
                        'size_mb': file_size / (1024**2)
                    }
    
    # 最终总结
    print(f"\n\n{'='*80}")
    print("诊断总结")
    print(f"{'='*80}\n")
    
    for filename, info in results.items():
        print(f"📄 {filename}")
        if not info.get('exists'):
            print(f"   状态: ❌ 文件不存在\n")
        else:
            file_type = info.get('type', 'unknown')
            size_mb = info.get('size_mb', 0)
            
            if file_type == 'JSON':
                json_type = info.get('json_type', 'unknown')
                print(f"   类型: ✅ JSON文件 ({json_type})")
                print(f"   大小: {size_mb:.2f} MB")
                print(f"   👉 这是 chunk_id 映射文件\n")
            
            elif file_type == 'FAISS_INDEX':
                index_type = info.get('index_type', 'unknown')
                vec_count = info.get('vector_count', 'unknown')
                dim = info.get('dimension', 'unknown')
                print(f"   类型: ✅ FAISS索引文件")
                print(f"   索引类型: {index_type}")
                print(f"   向量数量: {vec_count:,}")
                print(f"   向量维度: {dim}")
                print(f"   大小: {size_mb:.2f} MB")
                print(f"   👉 这是 FAISS 向量索引文件\n")
            
            else:
                error = info.get('error', 'unknown error')
                print(f"   类型: ❓ 无法判断")
                print(f"   错误: {error}")
                print(f"   大小: {size_mb:.2f} MB")
                print(f"   👉 需要进一步检查\n")
    
    # 最终建议
    print("="*80)
    print("建议")
    print("="*80)
    
    if 'qwen3_fp16_ip.faiss' in results:
        result = results['qwen3_fp16_ip.faiss']
        if result.get('type') == 'JSON':
            print("\n⚠️  文件命名错误!")
            print("   qwen3_fp16_ip.faiss 实际上是 JSON 映射文件")
            print("   应该重命名为: qwen3_fp16_ip_chunk_ids.json")
        elif result.get('type') == 'FAISS_INDEX':
            print("\n✅ 文件正确!")
            print("   qwen3_fp16_ip.faiss 确实是 FAISS 索引文件")
            print("   qwen3_fp16_ip_chunk_ids.json 应该是映射文件")
        else:
            print("\n❓ 无法确定文件类型，请检查文件是否损坏")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
