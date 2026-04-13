"""
语言识别模块 - 使用 FastText 进行语言分类
将混合语言的数据分流到不同的语言目录
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# FastText 导入
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    logging.warning("FastText 未安装，语言识别功能将不可用")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageSplitter:
    """语言分类器 - 使用 FastText 识别文本语言"""

    def __init__(self, model_path: str = 'models/lid.176.ftz',
                 target_languages: List[str] = None,
                 min_confidence: float = 0.5):
        """
        初始化语言分类器

        Args:
            model_path: FastText 模型路径
            target_languages: 目标语言列表，None 表示保留所有语言
            min_confidence: 最小置信度阈值
        """
        self.model_path = model_path
        self.target_languages = target_languages or ['en']
        self.min_confidence = min_confidence
        self.model = None

        self.stats = {
            'total': 0,
            'identified': 0,
            'failed': 0,
            'by_language': {},
            'filtered': 0
        }

        if FASTTEXT_AVAILABLE:
            self._load_model()
        else:
            logger.error("FastText 不可用，无法加载模型")

    def _load_model(self):
        """加载 FastText 模型"""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.warning(f"模型文件不存在: {self.model_path}")
                logger.info("将使用模拟的语言识别")
                self.model = None
                return

            self.model = fasttext.load_model(str(self.model_path))
            logger.info(f"成功加载 FastText 模型: {self.model_path}")

        except Exception as e:
            logger.error(f"加载 FastText 模型失败: {e}")
            self.model = None

    def predict_language(self, text: str) -> Tuple[str, float]:
        """
        预测文本语言

        Args:
            text: 输入文本

        Returns:
            (语言代码, 置信度)
        """
        if not self.model:
            # 使用模拟的语言识别（基于字符特征）
            return self._simulate_language_detection(text)

        try:
            # 预测语言
            predictions = self.model.predict(text, k=1)
            language = predictions[0][0].replace('__label__', '')
            confidence = predictions[0][1]

            return language, confidence

        except Exception as e:
            logger.warning(f"语言预测失败: {e}")
            return 'unknown', 0.0

    def _simulate_language_detection(self, text: str) -> Tuple[str, float]:
        """
        模拟语言检测（当 FastText 不可用时使用）
        基于简单的字符特征进行粗略判断

        Args:
            text: 输入文本

        Returns:
            (语言代码, 置信度)
        """
        if not text:
            return 'unknown', 0.0

        # 统计不同字符类型的比例
        total_chars = len(text)
        if total_chars == 0:
            return 'unknown', 0.0

        # 英语特征：主要包含拉丁字母、空格、常用标点
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        latin_ratio = latin_chars / total_chars if total_chars > 0 else 0

        # 中文字符范围
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

        # 简单的语言判断逻辑
        if chinese_ratio > 0.3:
            return 'zh', chinese_ratio
        elif latin_ratio > 0.5:
            return 'en', latin_ratio
        else:
            return 'unknown', 0.0

    def split_file_by_language(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        按语言分割文件

        Args:
            input_path: 输入 JSONL 文件路径
            output_dir: 输出目录（按语言创建子目录）

        Returns:
            分类统计信息
        """
        logger.info(f"开始语言分类: {input_path}")

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 按语言分组的数据
        language_data: Dict[str, List[Dict]] = {}

        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats['total'] += 1

                try:
                    item = json.loads(line)
                    text = item.get('text', '')

                    if not text:
                        self.stats['failed'] += 1
                        continue

                    # 预测语言
                    language, confidence = self.predict_language(text)

                    # 检查置信度
                    if confidence < self.min_confidence:
                        self.stats['failed'] += 1
                        continue

                    # 检查是否为目标语言
                    if self.target_languages and language not in self.target_languages:
                        self.stats['filtered'] += 1
                        continue

                    self.stats['identified'] += 1

                    # 添加语言和置信度信息
                    item['language'] = language
                    item['language_confidence'] = round(confidence, 4)

                    # 按语言分组
                    if language not in language_data:
                        language_data[language] = []

                    language_data[language].append(item)

                    # 更新语言统计
                    if language not in self.stats['by_language']:
                        self.stats['by_language'][language] = 0
                    self.stats['by_language'][language] += 1

                except json.JSONDecodeError:
                    self.stats['failed'] += 1
                    continue

        # 保存各语言的数据
        for language, data in language_data.items():
            language_output_path = f"{output_dir}/{language}/data.jsonl"
            Path(language_output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(language_output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            logger.info(f"保存了 {len(data)} 条 {language} 语言记录到 {language_output_path}")

        # 打印统计信息
        self._print_stats()

        return self.stats

    def filter_by_language(self, input_path: str, output_path: str,
                          target_language: str = 'en') -> Dict[str, Any]:
        """
        过滤特定语言的数据

        Args:
            input_path: 输入 JSONL 文件路径
            output_path: 输出 JSONL 文件路径
            target_language: 目标语言

        Returns:
            过滤统计信息
        """
        logger.info(f"开始过滤 {target_language} 语言数据: {input_path}")

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        filtered_data = []

        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get('text', '')

                    if not text:
                        continue

                    # 预测语言
                    language, confidence = self.predict_language(text)

                    # 只保留目标语言
                    if language == target_language and confidence >= self.min_confidence:
                        item['language'] = language
                        item['language_confidence'] = round(confidence, 4)
                        filtered_data.append(item)

                except json.JSONDecodeError:
                    continue

        # 保存过滤后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"保存了 {len(filtered_data)} 条 {target_language} 语言记录")

        return {
            'total_input': self.stats['total'],
            'filtered_output': len(filtered_data),
            'language': target_language
        }

    def _print_stats(self):
        """打印语言分类统计信息"""
        total = self.stats['total']
        identified = self.stats['identified']
        failed = self.stats['failed']
        filtered = self.stats['filtered']

        logger.info("=" * 50)
        logger.info("语言分类统计信息:")
        logger.info(f"  总记录数: {total}")
        logger.info(f"  识别成功: {identified} ({identified/total*100:.1f}%)")
        logger.info(f"  识别失败: {failed} ({failed/total*100:.1f}%)")
        logger.info(f"  被过滤: {filtered} ({filtered/total*100:.1f}%)")

        logger.info("\n语言分布:")
        for lang, count in sorted(self.stats['by_language'].items(),
                                 key=lambda x: x[1], reverse=True):
            logger.info(f"  {lang}: {count} ({count/total*100:.1f}%)")

        logger.info("=" * 50)


def download_fasttext_model(model_path: str = 'models/lid.176.ftz') -> bool:
    """
    下载 FastText 语言识别模型

    Args:
        model_path: 模型保存路径

    Returns:
        True 如果下载成功
    """
    try:
        import urllib.request
        import os

        model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始下载 FastText 模型: {model_url}")

        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r下载进度: {percent}%", end='')

        urllib.request.urlretrieve(model_url, model_path, progress_hook)
        print()

        logger.info(f"模型下载完成: {model_path}")
        return True

    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        return False


def main():
    """主函数示例"""
    # 示例用法
    splitter = LanguageSplitter(
        model_path='models/lid.176.ftz',
        target_languages=['en', 'zh'],
        min_confidence=0.5
    )

    # 测试语言检测
    print("语言检测测试:")

    test_texts = [
        "This is a sample text in English language for testing.",
        "这是一段中文测试文本，用于语言识别功能的测试。",
        "这是一个混合了 English and 中文 的 text sample。",
        "Ceci est un texte en français pour tester la détection de langue."
    ]

    for i, text in enumerate(test_texts, 1):
        lang, conf = splitter.predict_language(text)
        print(f"样本 {i}: 语言={lang}, 置信度={conf:.3f}")
        print(f"  文本: {text[:50]}...")
        print()

    print("语言分类示例:")
    input_file = "data/processed/deduplicated.jsonl"
    output_dir = "data/split_by_language"
    print(f"splitter.split_file_by_language('{input_file}', '{output_dir}')")

    print("\n语言过滤示例:")
    output_file = "data/processed/english_only.jsonl"
    print(f"splitter.filter_by_language('{input_file}', '{output_file}', 'en')")


if __name__ == "__main__":
    main()