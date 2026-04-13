"""
数据清洗模块 - 实现启发式清洗规则
使用 Gopher 和 C4 论文中的经典规则过滤低质量文本
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗器 - 实现启发式规则"""

    def __init__(self,
                 min_word_length: int = 3,
                 max_word_length: int = 20,
                 min_sentence_length: int = 10,
                 symbol_ratio_threshold: float = 0.1,
                 min_alpha_ratio: float = 0.7):
        """
        初始化数据清洗器

        Args:
            min_word_length: 最小单词长度
            max_word_length: 最大单词长度
            min_sentence_length: 最小句子长度
            symbol_ratio_threshold: 符号密度阈值
            min_alpha_ratio: 最小字母比例
        """
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.min_sentence_length = min_sentence_length
        self.symbol_ratio_threshold = symbol_ratio_threshold
        self.min_alpha_ratio = min_alpha_ratio

        # 代码符号集合
        self.code_symbols = {'{', '}', '[', ']', '<', '>', '\\', '/', '*', '=',
                            ';', ':', '(', ')', '&', '|', '!', '^', '%'}

        # 无意义短语黑名单
        self.bad_phrases = [
            "lorem ipsum",
            "enable cookies",
            "403 forbidden",
            "404 not found",
            "access denied",
            "page not found",
            "click here",
            "read more",
            "subscribe now",
            "sign up",
            "login required",
            "javascript required",
            "please enable javascript",
            "this site uses cookies",
            "by continuing to use this site",
            "all rights reserved",
            "copyright ©",
            "terms of service",
            "privacy policy"
        ]

        self.stats = {
            'total': 0,
            'passed': 0,
            'failed': {
                'short_text': 0,
                'long_word': 0,
                'high_symbol_ratio': 0,
                'low_alpha_ratio': 0,
                'bad_phrase': 0,
                'repeated_chars': 0,
                'suspicious_pattern': 0
            }
        }

    def clean_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        清洗输入文件中的数据

        Args:
            input_path: 输入 JSONL 文件路径
            output_path: 输出 JSONL 文件路径

        Returns:
            清洗统计信息
        """
        logger.info(f"开始清洗文件: {input_path}")

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cleaned_data = []

        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats['total'] += 1

                try:
                    item = json.loads(line)
                    text = item.get('text', '')

                    # 应用清洗规则
                    if self._is_high_quality(text):
                        cleaned_data.append(item)
                        self.stats['passed'] += 1

                except json.JSONDecodeError:
                    logger.warning(f"JSON 解析失败: {line[:100]}...")
                    continue

        # 保存清洗后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"清洗完成: {len(cleaned_data)}/{self.stats['total']} 条记录保留")

        # 打印统计信息
        self._print_stats()

        return self.stats

    def _is_high_quality(self, text: str) -> bool:
        """
        判断文本是否为高质量

        Args:
            text: 待检查的文本

        Returns:
            True 如果文本通过所有质量检查
        """
        # 检查 1: 文本长度
        if len(text.strip()) < self.min_sentence_length:
            self.stats['failed']['short_text'] += 1
            return False

        # 检查 2: 平均词长（过滤代码片段）
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length > self.max_word_length:
                self.stats['failed']['long_word'] += 1
                return False

        # 检查 3: 符号密度（过滤代码片段）
        if len(text) > 0:
            symbol_count = sum(1 for char in text if char in self.code_symbols)
            symbol_ratio = symbol_count / len(text)
            if symbol_ratio > self.symbol_ratio_threshold:
                self.stats['failed']['high_symbol_ratio'] += 1
                return False

        # 检查 4: 字母比例（过滤乱码和代码）
        alpha_count = sum(1 for char in text if char.isalpha())
        if len(text) > 0:
            alpha_ratio = alpha_count / len(text)
            if alpha_ratio < self.min_alpha_ratio:
                self.stats['failed']['low_alpha_ratio'] += 1
                return False

        # 检查 5: 黑名单短语
        lower_text = text.lower()
        for phrase in self.bad_phrases:
            if phrase in lower_text:
                self.stats['failed']['bad_phrase'] += 1
                return False

        # 检查 6: 重复字符（过滤无意义内容）
        if self._has_repeated_chars(text):
            self.stats['failed']['repeated_chars'] += 1
            return False

        # 检查 7: 可疑模式（如 URL 列表、邮件地址等）
        if self._has_suspicious_patterns(text):
            self.stats['failed']['suspicious_pattern'] += 1
            return False

        return True

    def _has_repeated_chars(self, text: str, threshold: int = 10) -> bool:
        """
        检查是否有连续重复字符

        Args:
            text: 待检查的文本
            threshold: 连续重复的最大次数

        Returns:
            True 如果发现连续重复字符
        """
        if len(text) < threshold:
            return False

        for i in range(len(text) - threshold + 1):
            if len(set(text[i:i+threshold])) <= 2:  # 允许2种字符交替
                return True

        return False

    def _has_suspicious_patterns(self, text: str) -> bool:
        """
        检查可疑模式

        Args:
            text: 待检查的文本

        Returns:
            True 如果发现可疑模式
        """
        # 检查过多的 URL
        url_pattern = r'https?://\S+'
        urls = re.findall(url_pattern, text)
        if len(urls) > 5:  # 超过5个URL视为可疑
            return True

        # 检查过多的邮箱
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if len(emails) > 5:  # 超过5个邮箱视为可疑
            return True

        # 检查过多的数字序列
        number_pattern = r'\d{10,}'  # 10位以上的数字
        numbers = re.findall(number_pattern, text)
        if len(numbers) > 5:
            return True

        return False

    def _print_stats(self):
        """打印清洗统计信息"""
        total = self.stats['total']
        passed = self.stats['passed']
        failed = self.stats['failed']

        logger.info("=" * 50)
        logger.info("清洗统计信息:")
        logger.info(f"  总记录数: {total}")
        logger.info(f"  通过记录: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"  拒绝记录: {total - passed} ({(total-passed)/total*100:.1f}%)")

        logger.info("\n拒绝原因分布:")
        for reason, count in failed.items():
            if count > 0:
                logger.info(f"  {reason}: {count} ({count/total*100:.1f}%)")

        logger.info("=" * 50)


def clean_text_sample(text: str) -> bool:
    """
    单个文本清洗示例

    Args:
        text: 待清洗的文本

    Returns:
        True 如果文本是高质量的
    """
    cleaner = DataCleaner()
    return cleaner._is_high_quality(text)


def main():
    """主函数示例"""
    # 示例用法
    cleaner = DataCleaner()

    # 测试一些样本
    test_samples = [
        "This is a high quality sample of English text with proper grammar and meaningful content.",
        "function(x) { return x > 0 ? true : false; } var a = [1,2,3];",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Home | About Us | Contact | Enable Cookies | Copyright 2023",
        "https://example.com https://test.com https://demo.com https://site.com https://web.com https://app.com"
    ]

    print("数据清洗测试:")
    for i, sample in enumerate(test_samples, 1):
        is_quality = cleaner._is_high_quality(sample)
        status = "✓ 通过" if is_quality else "✗ 拒绝"
        print(f"样本 {i}: {status}")
        print(f"  文本: {sample[:50]}...")
        print()

    print("文件处理示例:")
    input_file = "data/processed/extracted.jsonl"
    output_file = "data/processed/cleaned.jsonl"
    print(f"cleaner.clean_file('{input_file}', '{output_file}')")


if __name__ == "__main__":
    main()