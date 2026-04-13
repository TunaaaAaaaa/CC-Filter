"""
检查质量评分分布
"""

import json
from pathlib import Path
import re

def heuristic_quality_score(text):
    """启发式质量评分"""
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

# 分析质量评分分布
input_file = "data/split_by_language/en/data.jsonl"

scores = []
sample_low = []
sample_medium = []
sample_high = []

print("正在分析质量评分分布...")
with open(input_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        if line_num > 10000:  # 只分析前10000条
            break

        try:
            record = json.loads(line.strip())
            text = record.get('text', '')
            score = heuristic_quality_score(text)
            scores.append(score)

            if score < 0.6:
                if len(sample_low) < 3:
                    sample_low.append((text, score))
            elif score < 0.8:
                if len(sample_medium) < 3:
                    sample_medium.append((text, score))
            else:
                if len(sample_high) < 3:
                    sample_high.append((text, score))

        except Exception:
            continue

# 统计结果
import numpy as np
print(f"\n质量评分统计 (基于前10000条记录):")
print(f"平均分: {np.mean(scores):.3f}")
print(f"中位数: {np.median(scores):.3f}")
print(f"最低分: {np.min(scores):.3f}")
print(f"最高分: {np.max(scores):.3f}")

# 分数分布
low_count = sum(1 for s in scores if s < 0.6)
medium_count = sum(1 for s in scores if 0.6 <= s < 0.8)
high_count = sum(1 for s in scores if s >= 0.8)

print(f"\n分数分布:")
print(f"低质量 (<0.6): {low_count} ({low_count/len(scores)*100:.1f}%)")
print(f"中等质量 (0.6-0.8): {medium_count} ({medium_count/len(scores)*100:.1f}%)")
print(f"高质量 (≥0.8): {high_count} ({high_count/len(scores)*100:.1f}%)")

# 显示样本
print(f"\n低质量样本 (<0.6):")
for text, score in sample_low:
    print(f"  分数: {score:.3f}")
    print(f"  文本: {text[:100]}...")

print(f"\n中等质量样本 (0.6-0.8):")
for text, score in sample_medium:
    print(f"  分数: {score:.3f}")
    print(f"  文本: {text[:100]}...")

print(f"\n高质量样本 (≥0.8):")
for text, score in sample_high:
    print(f"  分数: {score:.3f}")
    print(f"  文本: {text[:100]}...")