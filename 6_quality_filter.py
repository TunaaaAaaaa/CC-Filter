"""
质量过滤模块 - 使用 KenLM 计算困惑度 (Perplexity)
通过统计语言模型评估文本的自然度和质量
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# KenLM 导入
try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False
    logging.warning("KenLM 未安装，质量过滤功能将不可用")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityFilter:
    """质量过滤器 - 使用 KenLM 困惑度评估文本质量"""

    def __init__(self,
                 model_path: str = 'models/en.arpa.bin',
                 min_length: int = 20,
                 max_length: int = 10000,
                 perplexity_threshold: float = -6.0,
                 use_length_normalization: bool = True):
        """
        初始化质量过滤器

        Args:
            model_path: KenLM 模型路径
            min_length: 最小文本长度
            max_length: 最大文本长度
            perplexity_threshold: 困惑度阈值（归一化 log score）
            use_length_normalization: 是否使用长度归一化
        """
        self.model_path = model_path
        self.min_length = min_length
        self.max_length = max_length
        self.perplexity_threshold = perplexity_threshold
        self.use_length_normalization = use_length_normalization
        self.model = None

        self.stats = {
            'total': 0,
            'passed': 0,
            'failed': {
                'too_short': 0,
                'too_long': 0,
                'low_quality': 0,
                'error': 0
            },
            'score_distribution': []
        }

        if KENLM_AVAILABLE:
            self._load_model()
        else:
            logger.error("KenLM 不可用，无法加载模型")

    def _load_model(self):
        """加载 KenLM 模型"""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.warning(f"模型文件不存在: {self.model_path}")
                logger.info("将使用模拟的质量评估")
                self.model = None
                return

            self.model = kenlm.Model(str(self.model_path))
            logger.info(f"成功加载 KenLM 模型: {self.model_path}")

        except Exception as e:
            logger.error(f"加载 KenLM 模型失败: {e}")
            self.model = None

    def compute_perplexity_score(self, text: str) -> Tuple[float, float]:
        """
        计算文本的困惑度分数

        Args:
            text: 输入文本

        Returns:
            (归一化 log score, 原始 log score)
        """
        if not self.model:
            # 使用模拟的质量评估
            return self._simulate_perplexity(text)

        try:
            words = text.split()

            if not words:
                return -999.0, -999.0

            # KenLM 返回 log10 概率
            log_score = self.model.score(text)

            # 计算归一化分数
            if self.use_length_normalization and len(words) > 0:
                normalized_score = log_score / len(words)
            else:
                normalized_score = log_score

            return normalized_score, log_score

        except Exception as e:
            logger.warning(f"计算困惑度失败: {e}")
            return -999.0, -999.0

    def _simulate_perplexity(self, text: str) -> Tuple[float, float]:
        """
        模拟困惑度计算（当 KenLM 不可用时使用）
        基于简单的统计特征进行粗略质量评估

        Args:
            text: 输入文本

        Returns:
            (归一化分数, 原始分数)
        """
        if not text:
            return -999.0, -999.0

        words = text.split()
        if not words:
            return -999.0, -999.0

        # 计算一些简单的统计特征
        avg_word_length = np.mean([len(w) for w in words])
        unique_word_ratio = len(set(words)) / len(words) if words else 0

        # 简单的质量评分逻辑
        # 平均词长在合理范围内，词多样性高 => 高质量
        if 3 <= avg_word_length <= 6 and unique_word_ratio > 0.5:
            # 高质量
            normalized_score = -4.5 + np.random.uniform(-0.5, 0.5)
        elif 2 <= avg_word_length <= 8 and unique_word_ratio > 0.3:
            # 中等质量
            normalized_score = -5.5 + np.random.uniform(-0.5, 0.5)
        else:
            # 低质量
            normalized_score = -7.0 + np.random.uniform(-0.5, 0.5)

        return normalized_score, normalized_score * len(words)

    def is_high_quality(self, text: str) -> Tuple[bool, float]:
        """
        判断文本是否为高质量

        Args:
            text: 输入文本

        Returns:
            (是否高质量, 归一化困惑度分数)
        """
        # 检查文本长度
        if len(text.strip()) < self.min_length:
            return False, -999.0

        if len(text.strip()) > self.max_length:
            return False, -999.0

        # 计算困惑度
        normalized_score, _ = self.compute_perplexity_score(text)

        # 检查是否通过阈值
        is_quality = normalized_score > self.perplexity_threshold

        return is_quality, normalized_score

    def filter_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        过滤文件中的低质量文本

        Args:
            input_path: 输入 JSONL 文件路径
            output_path: 输出 JSONL 文件路径

        Returns:
            过滤统计信息
        """
        logger.info(f"开始质量过滤: {input_path}")

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        high_quality_data = []

        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats['total'] += 1

                try:
                    item = json.loads(line)
                    text = item.get('text', '')

                    # 检查文本长度
                    if len(text.strip()) < self.min_length:
                        self.stats['failed']['too_short'] += 1
                        continue

                    if len(text.strip()) > self.max_length:
                        self.stats['failed']['too_long'] += 1
                        continue

                    # 计算困惑度
                    normalized_score, log_score = self.compute_perplexity_score(text)

                    # 记录分数分布
                    self.stats['score_distribution'].append(normalized_score)

                    # 检查质量
                    is_quality = normalized_score > self.perplexity_threshold

                    if is_quality:
                        item['perplexity_score'] = round(normalized_score, 4)
                        item['raw_log_score'] = round(log_score, 4)
                        high_quality_data.append(item)
                        self.stats['passed'] += 1
                    else:
                        self.stats['failed']['low_quality'] += 1

                except Exception as e:
                    logger.warning(f"处理记录失败: {e}")
                    self.stats['failed']['error'] += 1
                    continue

        # 保存高质量数据
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in high_quality_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"保存了 {len(high_quality_data)} 条高质量记录到 {output_path}")

        # 打印统计信息
        self._print_stats()

        return self.stats

    def _print_stats(self):
        """打印质量过滤统计信息"""
        total = self.stats['total']
        passed = self.stats['passed']
        failed = self.stats['failed']

        logger.info("=" * 50)
        logger.info("质量过滤统计信息:")
        logger.info(f"  总记录数: {total}")
        logger.info(f"  高质量记录: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"  低质量记录: {total - passed} ({(total-passed)/total*100:.1f}%)")

        logger.info("\n拒绝原因分布:")
        for reason, count in failed.items():
            if count > 0:
                logger.info(f"  {reason}: {count} ({count/total*100:.1f}%)")

        # 分数统计
        if self.stats['score_distribution']:
            scores = self.stats['score_distribution']
            logger.info(f"\n困惑度分数统计:")
            logger.info(f"  平均分数: {np.mean(scores):.2f}")
            logger.info(f"  中位数分数: {np.median(scores):.2f}")
            logger.info(f"  标准差: {np.std(scores):.2f}")
            logger.info(f"  最大分数: {np.max(scores):.2f}")
            logger.info(f"  最小分数: {np.min(scores):.2f}")

        logger.info("=" * 50)


def generate_kenlm_model(training_data_path: str, output_path: str,
                         arpa_path: str = 'models/en.arpa') -> bool:
    """
    生成 KenLM 语言模型

    Args:
        training_data_path: 训练数据路径
        output_path: 输出二进制模型路径
        arpa_path: 中间 ARPA 文件路径

    Returns:
        True 如果生成成功
    """
    try:
        import subprocess

        logger.info("开始生成 KenLM 模型...")

        # 1. 首先生成 ARPA 格式模型
        arpa_dir = Path(arpa_path).parent
        arpa_dir.mkdir(parents=True, exist_ok=True)

        cmd_lmplz = [
            'lmplz',
            '--order', '3',           # 3-gram 模型
            '--temp_prefix', '/tmp',
            '--memory', '50%',
            '--text', training_data_path,
            '--arpa', arpa_path
        ]

        logger.info(f"执行命令: {' '.join(cmd_lmplz)}")
        result = subprocess.run(cmd_lmplz, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"lmplz 失败: {result.stderr}")
            return False

        logger.info("ARPA 模型生成成功")

        # 2. 将 ARPA 转换为二进制格式
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd_build_binary = [
            'build_binary',
            arpa_path,
            output_path
        ]

        logger.info(f"执行命令: {' '.join(cmd_build_binary)}")
        result = subprocess.run(cmd_build_binary, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"build_binary 失败: {result.stderr}")
            return False

        logger.info(f"KenLM 模型生成成功: {output_path}")
        return True

    except Exception as e:
        logger.error(f"生成模型失败: {e}")
        return False


def main():
    """主函数示例"""
    # 示例用法
    filter = QualityFilter(
        model_path='models/en.arpa.bin',
        min_length=20,
        max_length=10000,
        perplexity_threshold=-6.0
    )

    # 测试质量评估
    print("质量评估测试:")

    test_texts = [
        # 高质量文本
        "The James Webb Space Telescope has captured a new image of the Pillars of Creation, revealing never-before-seen details of the famous star-forming region.",
        "Machine learning algorithms have revolutionized the field of natural language processing, enabling computers to understand and generate human-like text.",
        # 中等质量文本
        "This article discusses various aspects of web development and provides some useful tips for beginners who want to learn coding.",
        # 低质量文本
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Keyword list: best, cheap, fast, quality, service, online, shopping, store, free, shipping, discount, offer, deal",
        "Broken sentence with many grammar error and bad structure that make it hard to read and understand meaning.",
        "function(x) { return x > 0 ? true : false; } var a = [1,2,3];"
    ]

    print(f"困惑度阈值: {filter.perplexity_threshold}")
    print()

    for i, text in enumerate(test_texts, 1):
        is_quality, score = filter.is_high_quality(text)
        status = "✓ 高质量" if is_quality else "✗ 低质量"
        print(f"样本 {i}: {status}")
        print(f"  分数: {score:.2f}")
        print(f"  文本: {text[:60]}...")
        print()

    print("文件过滤示例:")
    input_file = "data/processed/english_only.jsonl"
    output_file = "data/final/final_data.jsonl"
    print(f"filter.filter_file('{input_file}', '{output_file}')")


if __name__ == "__main__":
    main()