"""
并行语言识别处理 - 优化版
使用多进程大幅提高处理速度
"""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import json
import multiprocessing
from pathlib import Path
from functools import partial

# 设置随机种子
DetectorFactory.seed = 0

def detect_language(text):
    """检测文本语言"""
    try:
        sample_text = text[:500] if len(text) > 500 else text
        lang = detect(sample_text)
        return lang
    except (LangDetectException, Exception):
        return None

def process_batch(batch_data, target_languages):
    """处理一批数据"""
    results = []
    for record in batch_data:
        try:
            text = record.get('text', '')
            lang = detect_language(text)

            if lang in target_languages:
                record['language'] = lang
                results.append(record)
        except Exception:
            continue
    return results

def parallel_language_detection(input_file, output_file, target_languages=['en'], num_workers=None):
    """
    并行语言识别

    Args:
        input_file: 输入文件
        output_file: 输出文件
        target_languages: 目标语言
        num_workers: 工作进程数
    """
    import math

    if num_workers is None:
        # 使用CPU核心数的75%，保留一些核心给系统
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))

    print(f"开始并行语言识别: {input_file}")
    print(f"目标语言: {target_languages}")
    print(f"使用进程数: {num_workers}")

    # 读取所有数据
    print("正在读取数据...")
    all_records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                all_records.append(record)
            except json.JSONDecodeError:
                continue

    total_count = len(all_records)
    print(f"总共读取 {total_count:,} 条记录")

    # 分批处理
    batch_size = max(100, total_count // (num_workers * 10))  # 动态批次大小
    batches = [all_records[i:i + batch_size] for i in range(0, len(all_records), batch_size)]

    print(f"分成 {len(batches)} 个批次，每批次约 {batch_size} 条记录")

    # 创建进程池
    print(f"启动 {num_workers} 个工作进程...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用partial固定目标语言参数
        process_func = partial(process_batch, target_languages=target_languages)

        # 并行处理所有批次
        results = pool.map(process_func, batches)

    # 合并结果
    print("正在合并结果...")
    all_filtered = []
    for batch_result in results:
        all_filtered.extend(batch_result)

    # 写入输出文件
    print(f"正在写入输出文件: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_filtered:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 统计结果
    filtered_count = len(all_filtered)
    print(f"\n并行语言识别完成!")
    print(f"总记录数: {total_count:,}")
    print(f"目标语言记录: {filtered_count:,}")
    print(f"其他语言记录: {total_count - filtered_count:,}")
    print(f"目标语言比例: {filtered_count/total_count*100:.1f}%")
    print(f"处理速度提升: 约 {num_workers}x")

    # 文件大小
    output_path = Path(output_file)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"输出文件大小: {size_mb:.2f} MB")

    return {
        'total': total_count,
        'target': filtered_count,
        'other': total_count - filtered_count
    }

if __name__ == "__main__":
    # Windows下需要这个保护
    multiprocessing.freeze_support()

    import argparse
    parser = argparse.ArgumentParser(description='并行语言识别')
    parser.add_argument('--input', '-i', default='data/processed/c4_deduplicated.jsonl',
                       help='输入文件')
    parser.add_argument('--output', '-o', default='data/split_by_language/en/data_parallel.jsonl',
                       help='输出文件')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='工作进程数 (默认: CPU核心数*0.75)')
    parser.add_argument('--target-languages', nargs='+', default=['en'],
                       help='目标语言列表')

    args = parser.parse_args()

    # 运行并行处理
    parallel_language_detection(
        args.input,
        args.output,
        target_languages=args.target_languages,
        num_workers=args.workers
    )