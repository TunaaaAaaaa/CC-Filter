"""
MinHash 去重模块 - 使用 Ray 实现分布式去重
基于局部敏感哈希 (LSH) 进行模糊去重
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
import ray
from datasketch import MinHash, MinHashLSH
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 配置 Ray（如果未初始化）
if not ray.is_initialized():
    try:
        ray.init(ignore_reinit_error=True)
        logger.info("Ray 初始化成功")
    except Exception as e:
        logger.warning(f"Ray 初始化失败: {e}")


class DistributedDeduplicator:
    """分布式去重器 - 使用 Ray 并行计算 MinHash"""

    def __init__(self,
                 num_perm: int = 128,
                 threshold: float = 0.8,
                 batch_size: int = 1000,
                 num_shingles: int = 5):
        """
        初始化分布式去重器

        Args:
            num_perm: MinHash 置换函数数量
            threshold: LSH 相似度阈值 (0-1)
            batch_size: 每批处理的数据量
            num_shingles: Shingle 长度
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_shingles = num_shingles

        self.stats = {
            'total_items': 0,
            'unique_items': 0,
            'duplicates': 0,
            'processing_time': 0
        }

    def deduplicate_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        对文件中的数据进行去重

        Args:
            input_path: 输入 JSONL 文件路径
            output_path: 输出 JSONL 文件路径

        Returns:
            去重统计信息
        """
        logger.info(f"开始去重处理: {input_path}")
        start_time = time.time()

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 读取数据
        data = self._load_data(input_path)
        self.stats['total_items'] = len(data)

        if len(data) == 0:
            logger.warning("输入文件为空")
            return self.stats

        # 使用 Ray 并行计算 MinHash
        logger.info(f"开始并行计算 {len(data)} 条数据的 MinHash...")
        minhashes = self._compute_minhashes_parallel(data)

        # 构建 LSH 索引
        logger.info("构建 LSH 索引...")
        lsh = self._build_lsh_index(minhashes)

        # 检测重复
        logger.info("检测重复数据...")
        unique_items, duplicate_count = self._find_duplicates(data, minhashes, lsh)

        self.stats['unique_items'] = len(unique_items)
        self.stats['duplicates'] = duplicate_count
        self.stats['processing_time'] = time.time() - start_time

        # 保存去重后的数据
        self._save_unique_data(unique_items, output_path)

        # 打印统计信息
        self._print_stats()

        return self.stats

    def _load_data(self, input_path: str) -> List[Dict]:
        """加载 JSONL 数据"""
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        return data

    def _compute_minhashes_parallel(self, data: List[Dict]) -> List[MinHash]:
        """
        使用 Ray 并行计算 MinHash

        Args:
            data: 待处理的数据列表

        Returns:
            MinHash 列表
        """
        # 将数据分批
        batches = [data[i:i + self.batch_size]
                  for i in range(0, len(data), self.batch_size)]

        # 创建并发的 Ray 任务
        futures = [compute_minhash_batch.remote(batch, i, self.num_perm, self.num_shingles)
                  for i, batch in enumerate(batches)]

        # 收集结果
        results = ray.get(futures)

        # 展平结果
        minhashes = []
        for batch_result in results:
            minhashes.extend(batch_result)

        return minhashes

    def _build_lsh_index(self, minhashes: List[MinHash]) -> MinHashLSH:
        """
        构建 LSH 索引

        Args:
            minhashes: MinHash 列表

        Returns:
            LSH 索引对象
        """
        # 创建 LSH 索引
        # num_perm: 与 MinHash 中的 num_perm 相同
        # threshold: 相似度阈值
        lsh = MinHashLSH(num_perm=self.num_perm,
                        params=(self.threshold,))

        # 添加所有 MinHash 到索引
        for i, mhash in enumerate(minhashes):
            lsh.insert(str(i), mhash)

        return lsh

    def _find_duplicates(self, data: List[Dict], minhashes: List[MinHash],
                        lsh: MinHashLSH) -> Tuple[List[Dict], int]:
        """
        查找重复数据

        Args:
            data: 原始数据
            minhashes: MinHash 列表
            lsh: LSH 索引

        Returns:
            (唯一数据列表, 重复数量)
        """
        unique_items = []
        seen_indices: Set[int] = set()
        duplicate_count = 0

        for i, mhash in enumerate(minhashes):
            if i in seen_indices:
                continue

            # 查询相似的 MinHash
            result = lsh.query(mhash)

            # 保留第一个出现的数据，标记其余为重复
            if result:
                first_index = int(result[0])
                if first_index == i:
                    unique_items.append(data[i])
                    # 将所有相似项标记为已见
                    for idx in result:
                        seen_indices.add(int(idx))
                else:
                    duplicate_count += 1

        return unique_items, duplicate_count

    def _save_unique_data(self, unique_items: List[Dict], output_path: str):
        """保存去重后的数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in unique_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"保存了 {len(unique_items)} 条唯一记录到 {output_path}")

    def _print_stats(self):
        """打印去重统计信息"""
        total = self.stats['total_items']
        unique = self.stats['unique_items']
        duplicates = self.stats['duplicates']
        time_cost = self.stats['processing_time']

        logger.info("=" * 50)
        logger.info("去重统计信息:")
        logger.info(f"  总记录数: {total}")
        logger.info(f"  唯一记录: {unique} ({unique/total*100:.1f}%)")
        logger.info(f"  重复记录: {duplicates} ({duplicates/total*100:.1f}%)")
        logger.info(f"  处理时间: {time_cost:.2f} 秒")
        logger.info(f"  处理速度: {total/time_cost:.1f} 条/秒")
        logger.info("=" * 50)


@ray.remote
def compute_minhash_batch(batch: List[Dict], batch_id: int,
                          num_perm: int, num_shingles: int) -> List[MinHash]:
    """
    Ray Worker: 并行计算一批数据的 MinHash

    Args:
        batch: 数据批次
        batch_id: 批次 ID（用于日志）
        num_perm: MinHash 置换函数数量
        num_shingles: Shingle 长度

    Returns:
        MinHash 列表
    """
    minhashes = []

    for item in batch:
        text = item.get('text', '')
        if not text:
            continue

        # 创建 MinHash
        mhash = MinHash(num_perm=num_perm)

        # 创建 Shingles
        words = text.lower().split()
        shingles = [' '.join(words[i:i+num_shingles])
                   for i in range(len(words) - num_shingles + 1)]

        # 更新 MinHash
        for shingle in shingles:
            mhash.update(shingle.encode('utf-8'))

        minhashes.append(mhash)

    return minhashes


def compute_single_minhash(text: str, num_perm: int = 128) -> MinHash:
    """
    计算单个文本的 MinHash（用于测试）

    Args:
        text: 输入文本
        num_perm: 置换函数数量

    Returns:
        MinHash 对象
    """
    mhash = MinHash(num_perm=num_perm)
    words = text.lower().split()

    for word in words:
        mhash.update(word.encode('utf-8'))

    return mhash


def estimate_jaccard_similarity(mhash1: MinHash, mhash2: MinHash) -> float:
    """
    估算两个 MinHash 之间的 Jaccard 相似度

    Args:
        mhash1: 第一个 MinHash
        mhash2: 第二个 MinHash

    Returns:
        Jaccard 相似度 (0-1)
    """
    return mhash1.jaccard(mhash2)


def main():
    """主函数示例"""
    # 示例用法
    deduplicator = DistributedDeduplicator(
        num_perm=128,
        threshold=0.8,
        batch_size=1000
    )

    # 测试 MinHash 相似度
    print("MinHash 相似度测试:")
    text1 = "This is a sample text about machine learning and natural language processing."
    text2 = "This is a sample text about machine learning and NLP techniques."
    text3 = "The weather is nice today and I want to go for a walk in the park."

    m1 = compute_single_minhash(text1)
    m2 = compute_single_minhash(text2)
    m3 = compute_single_minhash(text3)

    sim12 = estimate_jaccard_similarity(m1, m2)
    sim13 = estimate_jaccard_similarity(m1, m3)

    print(f"文本 1 vs 文本 2 相似度: {sim12:.3f}")
    print(f"文本 1 vs 文本 3 相似度: {sim13:.3f}")
    print()

    print("文件去重示例:")
    input_file = "data/processed/cleaned.jsonl"
    output_file = "data/processed/deduplicated.jsonl"
    print(f"deduplicator.deduplicate_file('{input_file}', '{output_file}')")


if __name__ == "__main__":
    main()