"""
运行去重步骤 - 简化版（不使用Ray）
直接使用MinHash和LSH进行去重，适合学习理解算法原理
"""

from datasketch import MinHash, MinHashLSH
import json
from pathlib import Path
import re

def preprocess_text(text):
    """预处理文本：分词、清洗"""
    # 转换为小写
    text = text.lower()
    # 移除特殊字符和数字
    text = re.sub(r'[^a-z\s]', '', text)
    # 分词
    words = text.split()
    return words

def create_minhash(text, num_perm=128):
    """为文本创建MinHash签名"""
    words = preprocess_text(text)
    m = MinHash(num_perm=num_perm)
    for word in words:
        m.update(word.encode('utf-8'))
    return m

def deduplicate_file(input_file, output_file, threshold=0.8, num_perm=128):
    """
    去重函数

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        threshold: 相似度阈值 (0-1)
        num_perm: MinHash置换函数数量
    """
    print(f"开始去重处理: {input_file}")
    print(f"相似度阈值: {threshold}")
    print(f"MinHash置换函数数: {num_perm}")

    # 创建LSH索引
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # 统计信息
    total_count = 0
    unique_count = 0
    duplicate_count = 0

    # 处理文件
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line.strip())
                text = record.get('text', '')
                total_count += 1

                # 创建MinHash签名
                minhash = create_minhash(text, num_perm)

                # 检查是否为重复
                # 使用URL作为文档的唯一标识符
                doc_id = record.get('url', str(line_num))

                # 查找相似文档
                try:
                    # 尝试插入LSH索引，如果返回 False 则表示存在重复
                    is_duplicate = False
                    for similar_doc_id in lsh.query(minhash):
                        # 这里可以添加更精确的相似度计算
                        is_duplicate = True
                        break

                    if is_duplicate:
                        duplicate_count += 1
                    else:
                        # 插入到LSH索引
                        lsh.insert(doc_id, minhash)
                        unique_count += 1
                        outfile.write(line)

                except Exception as lsh_error:
                    # 如果查询失败，认为是新文档
                    lsh.insert(doc_id, minhash)
                    unique_count += 1
                    outfile.write(line)

                # 每处理10000行打印一次进度
                if line_num % 10000 == 0:
                    print(f"已处理 {line_num:,} 行，唯一 {unique_count:,} 条，重复 {duplicate_count:,} 条")

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理第 {line_num} 行时出错: {e}")
                continue

    print(f"\n去重完成!")
    print(f"总记录数: {total_count:,}")
    print(f"唯一记录: {unique_count:,}")
    print(f"重复记录: {duplicate_count:,}")
    print(f"去重率: {duplicate_count/total_count*100:.1f}%")

    # 检查输出文件大小
    output_path = Path(output_file)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"输出文件大小: {size_mb:.2f} MB")

    return {
        'total': total_count,
        'unique': unique_count,
        'duplicates': duplicate_count
    }

if __name__ == "__main__":
    # 处理文件
    input_file = "data/processed/c4_cleaned.jsonl"
    output_file = "data/processed/c4_deduplicated.jsonl"

    # 运行去重
    deduplicate_file(input_file, output_file, threshold=0.8, num_perm=128)