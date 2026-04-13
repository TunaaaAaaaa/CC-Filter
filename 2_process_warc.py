"""
WARC 文件处理和文本提取模块
负责从 Common Crawl WARC 文件中提取纯文本内容
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any
from warcio.archiveiterator import ArchiveIterator
import trafilatura
import trafilatura.core

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WARCProcessor:
    """WARC 文件处理器"""

    def __init__(self, min_text_length: int = 100):
        """
        初始化 WARC 处理器

        Args:
            min_text_length: 最小文本长度，低于此长度的文本将被丢弃
        """
        self.min_text_length = min_text_length
        self.stats = {
            'total_records': 0,
            'html_responses': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'short_texts': 0
        }

    def process_warc_file(self, warc_path: str, output_path: str) -> Dict[str, int]:
        """
        处理单个 WARC 文件并保存提取的文本

        Args:
            warc_path: WARC 文件路径
            output_path: 输出 JSONL 文件路径

        Returns:
            处理统计信息
        """
        logger.info(f"开始处理 WARC 文件: {warc_path}")

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        extracted_data = []

        # 打开 WARC 文件（支持 gzip 压缩）
        open_func = gzip.open if warc_path.endswith('.gz') else open
        mode = 'rb' if warc_path.endswith('.gz') else 'r'

        try:
            with open_func(warc_path, mode) as stream:
                for record in ArchiveIterator(stream):
                    self.stats['total_records'] += 1

                    if record.rec_type == 'response':
                        # 获取内容类型
                        content_type = record.http_headers.get_header('Content-Type', '')

                        # 只处理 HTML 内容
                        if not content_type or 'text/html' not in content_type.lower():
                            continue

                        self.stats['html_responses'] += 1

                        # 尝试提取文本
                        result = self._extract_text_from_record(record)
                        if result:
                            extracted_data.append(result)
                        else:
                            self.stats['failed_extractions'] += 1

        except Exception as e:
            logger.error(f"处理 WARC 文件时出错: {e}")
            raise

        # 保存提取的数据
        self._save_extracted_data(extracted_data, output_path)

        # 打印统计信息
        self._print_stats()

        return self.stats

    def _extract_text_from_record(self, record) -> Dict[str, Any]:
        """
        从 WARC 记录中提取文本

        Args:
            record: WARC 记录对象

        Returns:
            提取的数据字典，失败返回 None
        """
        try:
            # 读取内容
            content = record.content_stream().read()

            # 使用 trafilatura 提取文本
            text = trafilatura.extract(
                content,
                include_comments=False,      # 不包含评论
                include_tables=False,        # 不包含表格
                no_fallback=False,           # 允许回退策略
                no_links=True               # 不包含链接文本
            )

            if not text:
                return None

            # 检查文本长度
            if len(text.strip()) < self.min_text_length:
                self.stats['short_texts'] += 1
                return None

            self.stats['successful_extractions'] += 1

            return {
                'url': record.rec_headers.get_header('WARC-Target-URI', ''),
                'text': text.strip(),
                'length': len(text.strip()),
                'date': record.rec_headers.get_header('WARC-Date', '')
            }

        except Exception as e:
            logger.warning(f"提取文本失败: {e}")
            return None

    def _save_extracted_data(self, data: list, output_path: str):
        """保存提取的数据到 JSONL 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"保存了 {len(data)} 条记录到 {output_path}")

    def _print_stats(self):
        """打印处理统计信息"""
        total = self.stats['total_records']
        html = self.stats['html_responses']
        success = self.stats['successful_extractions']
        failed = self.stats['failed_extractions']
        short = self.stats['short_texts']

        logger.info("=" * 50)
        logger.info("处理统计信息:")
        logger.info(f"  总记录数: {total}")
        logger.info(f"  HTML 响应: {html} ({html/total*100:.1f}%)")
        logger.info(f"  成功提取: {success} ({success/total*100:.1f}%)")
        logger.info(f"  提取失败: {failed} ({failed/total*100:.1f}%)")
        logger.info(f"  过短文本: {short} ({short/total*100:.1f}%)")
        logger.info(f"  最终产出率: {success/total*100:.1f}%")
        logger.info("=" * 50)


def main():
    """主函数示例"""
    # 示例用法
    processor = WARCProcessor(min_text_length=100)

    # 处理 WARC 文件
    warc_file = "data/raw/sample.warc.gz"
    output_file = "data/processed/extracted.jsonl"

    # 注意：实际使用时需要提供真实的 WARC 文件
    # processor.process_warc_file(warc_file, output_file)

    print("WARC 处理器已初始化")
    print("使用示例:")
    print(f"processor.process_warc_file('{warc_file}', '{output_file}')")


if __name__ == "__main__":
    main()