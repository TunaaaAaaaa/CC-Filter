"""
运行语言识别步骤 - 使用langdetect替代fasttext
学习语言识别技术的基本原理
"""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import json
from pathlib import Path

# 设置随机种子以获得一致的结果
DetectorFactory.seed = 0

def detect_language(text):
    """
    检测文本语言

    Args:
        text: 输入文本

    Returns:
        语言代码或None（如果检测失败）
    """
    try:
        # 只检测前500个字符以提高速度
        sample_text = text[:500] if len(text) > 500 else text
        lang = detect(sample_text)
        return lang
    except LangDetectException:
        return None
    except Exception:
        return None

def is_target_language(text, target_languages=['en'], min_confidence=0.7):
    """
    检查文本是否为目标语言

    Args:
        text: 输入文本
        target_languages: 目标语言列表
        min_confidence: 最小置信度（这里简化处理）

    Returns:
        是否为目标语言
    """
    lang = detect_language(text)
    if lang is None:
        return False
    return lang in target_languages

def filter_by_language(input_file, output_file, target_languages=['en']):
    """
    按语言过滤文本

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        target_languages: 目标语言列表
    """
    print(f"开始语言识别: {input_file}")
    print(f"目标语言: {target_languages}")

    # 统计信息
    total_count = 0
    target_count = 0
    other_count = 0
    failed_count = 0

    # 语言统计
    language_stats = {}

    # 创建输出目录
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # 处理文件
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line.strip())
                text = record.get('text', '')
                total_count += 1

                # 检测语言
                lang = detect_language(text)

                if lang is None:
                    failed_count += 1
                else:
                    # 更新语言统计
                    language_stats[lang] = language_stats.get(lang, 0) + 1

                    # 检查是否为目标语言
                    if lang in target_languages:
                        target_count += 1
                        # 添加语言信息到记录
                        record['language'] = lang
                        outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                    else:
                        other_count += 1

                # 每处理10000行打印一次进度
                if line_num % 10000 == 0:
                    print(f"已处理 {line_num:,} 行，目标语言 {target_count:,} 条 ({target_count/line_num*100:.1f}%)")

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理第 {line_num} 行时出错: {e}")
                continue

    print(f"\n语言识别完成!")
    print(f"总记录数: {total_count:,}")
    print(f"目标语言记录: {target_count:,}")
    print(f"其他语言记录: {other_count:,}")
    print(f"检测失败: {failed_count:,}")
    print(f"目标语言比例: {target_count/total_count*100:.1f}%")

    # 打印语言统计
    print(f"\n语言分布:")
    for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count:,} ({count/total_count*100:.1f}%)")

    # 检查输出文件大小
    output_path = Path(output_file)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"输出文件大小: {size_mb:.2f} MB")

    return {
        'total': total_count,
        'target': target_count,
        'other': other_count,
        'failed': failed_count,
        'language_stats': language_stats
    }

if __name__ == "__main__":
    # 处理完整的去重数据
    input_file = "data/processed/c4_deduplicated.jsonl"
    output_file = "data/split_by_language/en/data.jsonl"

    print(f"开始处理完整的去重数据...")
    filter_by_language(input_file, output_file, target_languages=['en'])