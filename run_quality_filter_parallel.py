"""
并行质量过滤 - 优化版
使用多进程和KenLM进行质量评估
"""

import json
import multiprocessing
from pathlib import Path
from functools import partial
import re

# 尝试导入KenLM
try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False
    print("KenLM不可用，使用启发式质量评估")

def heuristic_quality_score(text):
    """
    启发式质量评分（当KenLM不可用时使用）

    Args:
        text: 输入文本

    Returns:
        质量分数 (0-1)
    """
    score = 1.0

    # 长度评分
    if len(text) < 50:
        score *= 0.5
    elif len(text) < 100:
        score *= 0.8

    # 词数评分
    words = text.split()
    if len(words) < 10:
        score *= 0.6
    elif len(words) < 20:
        score *= 0.9

    # 平均词长评分
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 3 or avg_word_length > 15:
            score *= 0.7

    # 符号比例评分
    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    symbol_ratio = symbols / len(text) if len(text) > 0 else 0
    if symbol_ratio > 0.3:
        score *= 0.5
    elif symbol_ratio > 0.2:
        score *= 0.8

    # 重复字符评分
    if len(text) > 10:
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1

        max_count = max(char_counts.values())
        if max_count / len(text) > 0.5:
            score *= 0.6

    return score

def kenlm_quality_score(text, model):
    """
    使用KenLM计算质量分数

    Args:
        text: 输入文本
        model: KenLM模型

    Returns:
        质量分数
    """
    try:
        # 归一化分数
        score = model.score(text)
        normalized_score = score / len(text.split()) if text.split() else score
        return normalized_score
    except Exception:
        return float('-inf')

def is_high_quality(record, quality_threshold=0.6, model=None):
    """
    检查记录是否高质量

    Args:
        record: 数据记录
        quality_threshold: 质量阈值
        model: KenLM模型（可选）

    Returns:
        是否高质量
    """
    text = record.get('text', '')

    if len(text) < 20:
        return False

    if model is not None:
        # 使用KenLM评分
        score = kenlm_quality_score(text, model)
        # KenLM分数越低越好（困惑度），通常-6.0是一个好的阈值
        return score > quality_threshold if quality_threshold < 0 else score < quality_threshold
    else:
        # 使用启发式评分
        score = heuristic_quality_score(text)
        return score >= quality_threshold

def process_batch(batch_data, quality_threshold=0.6, model=None):
    """处理一批数据"""
    results = []
    for record in batch_data:
        try:
            if is_high_quality(record, quality_threshold, model):
                results.append(record)
        except Exception:
            continue
    return results

def parallel_quality_filter(input_file, output_file, quality_threshold=0.6,
                          num_workers=None, model_path=None):
    """
    并行质量过滤

    Args:
        input_file: 输入文件
        output_file: 输出文件
        quality_threshold: 质量阈值
        num_workers: 工作进程数
        model_path: KenLM模型路径
    """
    import math

    if num_workers is None:
        # 使用CPU核心数的75%
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))

    print(f"开始并行质量过滤: {input_file}")
    print(f"质量阈值: {quality_threshold}")
    print(f"使用进程数: {num_workers}")

    # 加载KenLM模型（如果提供）
    model = None
    if model_path and KENLM_AVAILABLE and Path(model_path).exists():
        print(f"加载KenLM模型: {model_path}")
        model = kenlm.Model(model_path)
        print("KenLM模型加载成功")
    elif model_path:
        print(f"KenLM模型文件不存在: {model_path}")
        print("使用启发式质量评估")
    else:
        print("使用启发式质量评估")

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
    batch_size = max(100, total_count // (num_workers * 10))
    batches = [all_records[i:i + batch_size] for i in range(0, len(all_records), batch_size)]

    print(f"分成 {len(batches)} 个批次，每批次约 {batch_size} 条记录")

    # 创建进程池
    print(f"启动 {num_workers} 个工作进程...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用partial固定参数
        process_func = partial(process_batch, quality_threshold=quality_threshold, model=model)

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
    print(f"\n并行质量过滤完成!")
    print(f"总记录数: {total_count:,}")
    print(f"高质量记录: {filtered_count:,}")
    print(f"低质量记录: {total_count - filtered_count:,}")
    print(f"通过率: {filtered_count/total_count*100:.1f}%")
    print(f"处理速度提升: 约 {num_workers}x")

    # 文件大小
    output_path = Path(output_file)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"输出文件大小: {size_mb:.2f} MB")

    return {
        'total': total_count,
        'passed': filtered_count,
        'rejected': total_count - filtered_count
    }

if __name__ == "__main__":
    # Windows下需要这个保护
    multiprocessing.freeze_support()

    import argparse
    parser = argparse.ArgumentParser(description='并行质量过滤')
    parser.add_argument('--input', '-i', default='data/split_by_language/en/data.jsonl',
                       help='输入文件')
    parser.add_argument('--output', '-o', default='data/final/c4_final_data_parallel.jsonl',
                       help='输出文件')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='工作进程数 (默认: CPU核心数*0.75)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                       help='质量阈值 (默认: 0.6)')
    parser.add_argument('--model', '-m', default=None,
                       help='KenLM模型路径 (可选)')

    args = parser.parse_args()

    # 运行并行质量过滤
    parallel_quality_filter(
        args.input,
        args.output,
        quality_threshold=args.threshold,
        num_workers=args.workers,
        model_path=args.model
    )