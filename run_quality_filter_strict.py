"""
严格的质量过滤 - 重新设计
使用更严格的质量评估标准
"""

import json
import multiprocessing
from pathlib import Path
from functools import partial
import re

def strict_quality_score(text):
    """
    严格的质量评分（重新设计）

    Args:
        text: 输入文本

    Returns:
        质量分数 (0-1) 和详细评分信息
    """
    details = {}
    score = 1.0

    # 1. 长度评分 (更严格)
    if len(text) < 50:
        score *= 0.2
        details['length'] = 'very_short'
    elif len(text) < 100:
        score *= 0.6
        details['length'] = 'short'
    elif len(text) < 200:
        score *= 0.9
        details['length'] = 'medium'
    elif len(text) < 500:
        score *= 1.0
        details['length'] = 'good'
    elif len(text) < 1000:
        score *= 1.0
        details['length'] = 'very_good'
    else:
        score *= 0.8  # 过长可能有重复内容
        details['length'] = 'too_long'

    # 2. 词数评分 (更严格)
    words = text.split()
    word_count = len(words)

    if word_count < 10:
        score *= 0.3
        details['words'] = 'very_few'
    elif word_count < 20:
        score *= 0.7
        details['words'] = 'few'
    elif word_count < 50:
        score *= 1.0
        details['words'] = 'good'
    elif word_count < 100:
        score *= 1.0
        details['words'] = 'very_good'
    elif word_count < 200:
        score *= 0.9
        details['words'] = 'many'
    else:
        score *= 0.7  # 词数过多可能有重复
        details['words'] = 'too_many'

    # 3. 平均词长评分
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
        if avg_word_length < 2:
            score *= 0.4
            details['avg_word_len'] = 'very_short'
        elif avg_word_length < 3:
            score *= 0.7
            details['avg_word_len'] = 'short'
        elif avg_word_length < 5:
            score *= 1.0
            details['avg_word_len'] = 'good'
        elif avg_word_length < 8:
            score *= 1.0
            details['avg_word_len'] = 'very_good'
        else:
            score *= 0.6  # 平均词长过长
            details['avg_word_len'] = 'too_long'

    # 4. 符号比例评分 (更严格)
    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    symbol_ratio = symbols / len(text) if len(text) > 0 else 0

    if symbol_ratio > 0.4:
        score *= 0.3
        details['symbol_ratio'] = 'very_high'
    elif symbol_ratio > 0.3:
        score *= 0.6
        details['symbol_ratio'] = 'high'
    elif symbol_ratio > 0.2:
        score *= 0.9
        details['symbol_ratio'] = 'medium'
    else:
        details['symbol_ratio'] = 'good'

    # 5. 句子结构评分
    sentences = re.split(r'[.!?]+', text)
    non_empty_sentences = [s.strip() for s in sentences if s.strip()]

    if len(non_empty_sentences) < 2:
        score *= 0.5
        details['sentences'] = 'very_few'
    elif len(non_empty_sentences) < 3:
        score *= 0.8
        details['sentences'] = 'few'
    elif len(non_empty_sentences) < 10:
        score *= 1.0
        details['sentences'] = 'good'
    else:
        details['sentences'] = 'many'

    # 6. 重复内容检测 (新增)
    if len(text) > 50:
        # 检查重复的短语
        words_lower = [word.lower() for word in words]
        word_counts = {}
        for word in words_lower:
            word_counts[word] = word_counts.get(word, 0) + 1

        # 重复词比例
        repeated_words = sum(1 for count in word_counts.values() if count > 3)
        repeated_ratio = repeated_words / word_count if word_count > 0 else 0

        if repeated_ratio > 0.3:
            score *= 0.5
            details['repetition'] = 'high'
        elif repeated_ratio > 0.2:
            score *= 0.8
            details['repetition'] = 'medium'
        else:
            details['repetition'] = 'low'

    # 7. 特殊模式检测 (新增)
    # 检查导航栏、版权声明等低质量内容
    low_quality_patterns = [
        r'copyright\s+\d{4}',
        r'all\s+rights\s+reserved',
        r'home\s*[\|·]\s*about',
        r'privacy\s+policy',
        r'terms\s+of\s+service',
        r'cookie\s+policy',
        r'log\s+in\s*[\|·]\s*register',
        r'sign\s+up\s*[\|·]\s*login',
    ]

    text_lower = text.lower()
    pattern_matches = sum(1 for pattern in low_quality_patterns if re.search(pattern, text_lower))

    if pattern_matches >= 3:
        score *= 0.4
        details['low_quality_patterns'] = 'many'
    elif pattern_matches >= 2:
        score *= 0.7
        details['low_quality_patterns'] = 'some'
    elif pattern_matches >= 1:
        score *= 0.9
        details['low_quality_patterns'] = 'few'
    else:
        details['low_quality_patterns'] = 'none'

    # 8. 数字比例检测 (新增)
    digit_chars = sum(1 for char in text if char.isdigit())
    digit_ratio = digit_chars / len(text) if len(text) > 0 else 0

    if digit_ratio > 0.3:
        score *= 0.6
        details['digit_ratio'] = 'high'
    elif digit_ratio > 0.2:
        score *= 0.8
        details['digit_ratio'] = 'medium'
    else:
        details['digit_ratio'] = 'good'

    return score, details

def is_high_quality(record, quality_threshold=0.7):
    """
    检查记录是否高质量（使用严格评分）

    Args:
        record: 数据记录
        quality_threshold: 质量阈值

    Returns:
        是否高质量
    """
    text = record.get('text', '')

    if len(text) < 50:
        return False

    score, details = strict_quality_score(text)
    return score >= quality_threshold, score, details

def process_batch_strict(batch_data, quality_threshold=0.7):
    """处理一批数据（严格版本）"""
    results = []
    rejected_samples = []

    for record in batch_data:
        try:
            passed, score, details = is_high_quality(record, quality_threshold)
            if passed:
                record['quality_score'] = score
                record['quality_details'] = details
                results.append(record)
            else:
                # 保存一些拒绝样本用于分析
                if len(rejected_samples) < 10:
                    rejected_samples.append({
                        'text': record.get('text', '')[:100],
                        'score': score,
                        'details': details
                    })
        except Exception as e:
            continue

    return results, rejected_samples

def parallel_quality_filter_strict(input_file, output_file, quality_threshold=0.7,
                                 num_workers=None):
    """
    并行质量过滤（严格版本）

    Args:
        input_file: 输入文件
        output_file: 输出文件
        quality_threshold: 质量阈值
        num_workers: 工作进程数
    """
    import math

    if num_workers is None:
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))

    print(f"开始严格质量过滤: {input_file}")
    print(f"质量阈值: {quality_threshold}")
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
    batch_size = max(100, total_count // (num_workers * 10))
    batches = [all_records[i:i + batch_size] for i in range(0, len(all_records), batch_size)]

    print(f"分成 {len(batches)} 个批次，每批次约 {batch_size} 条记录")

    # 创建进程池
    print(f"启动 {num_workers} 个工作进程...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用partial固定阈值
        process_func = partial(process_batch_strict, quality_threshold=quality_threshold)

        # 并行处理所有批次
        results = pool.map(process_func, batches)

    # 合并结果
    print("正在合并结果...")
    all_filtered = []
    all_rejected = []

    for filtered_batch, rejected_batch in results:
        all_filtered.extend(filtered_batch)
        all_rejected.extend(rejected_batch)

    # 写入输出文件
    print(f"正在写入输出文件: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_filtered:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 统计结果
    filtered_count = len(all_filtered)
    rejected_count = total_count - filtered_count

    print(f"\n严格质量过滤完成!")
    print(f"总记录数: {total_count:,}")
    print(f"高质量记录: {filtered_count:,}")
    print(f"低质量记录: {rejected_count:,}")
    print(f"通过率: {filtered_count/total_count*100:.1f}%")
    print(f"拒绝率: {rejected_count/total_count*100:.1f}%")

    # 分析拒绝样本
    if all_rejected:
        print(f"\n拒绝样本分析 (前10个):")
        for i, sample in enumerate(all_rejected[:10], 1):
            print(f"\n样本 {i}:")
            print(f"  分数: {sample['score']:.3f}")
            print(f"  文本: {sample['text']}...")
            print(f"  评分详情: {sample['details']}")

    # 文件大小
    output_path = Path(output_file)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"输出文件大小: {size_mb:.2f} MB")

    return {
        'total': total_count,
        'passed': filtered_count,
        'rejected': rejected_count
    }

if __name__ == "__main__":
    multiprocessing.freeze_support()

    import argparse
    parser = argparse.ArgumentParser(description='严格质量过滤')
    parser.add_argument('--input', '-i', default='data/split_by_language/en/data.jsonl',
                       help='输入文件')
    parser.add_argument('--output', '-o', default='data/final/c4_final_strict.jsonl',
                       help='输出文件')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='工作进程数 (默认: CPU核心数*0.75)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                       help='质量阈值 (默认: 0.7)')

    args = parser.parse_args()

    # 运行严格质量过滤
    parallel_quality_filter_strict(
        args.input,
        args.output,
        quality_threshold=args.threshold,
        num_workers=args.workers
    )