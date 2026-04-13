"""
C4 数据集处理模块
负责处理从 HuggingFace 下载的 C4 JSON 格式数据
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, List
import gzip

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class C4Processor:
    """C4 数据集处理器"""

    def __init__(self, min_text_length: int = 100):
        """
        初始化 C4 处理器

        Args:
            min_text_length: 最小文本长度，低于此长度的文本将被丢弃
        """
        self.min_text_length = min_text_length
        self.stats = {
            'total_records': 0,
            'successful_extractions': 0,
            'short_texts': 0,
            'empty_texts': 0
        }

    def process_c4_file(self, c4_path: str, output_path: str) -> Dict[str, int]:
        """
        处理单个 C4 JSON 文件并保存提取的文本

        Args:
            c4_path: C4 JSON 文件路径
            output_path: 输出 JSONL 文件路径

        Returns:
            处理统计信息
        """
        logger.info(f"开始处理 C4 文件: {c4_path}")

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        extracted_data = []

        # 判断是否为 gzip 压缩文件
        open_func = gzip.open if c4_path.endswith('.gz') else open
        mode = 'rt' if c4_path.endswith('.gz') else 'r'

        try:
            with open_func(c4_path, mode, encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # 解析 JSON 行
                        record = json.loads(line.strip())
                        self.stats['total_records'] += 1

                        # 提取文本内容
                        result = self._extract_text_from_c4_record(record)
                        if result:
                            extracted_data.append(result)

                        # 每处理 10000 行打印一次进度
                        if line_num % 10000 == 0:
                            logger.info(f"已处理 {line_num} 行...")

                    except json.JSONDecodeError as e:
                        logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"处理第 {line_num} 行时出错: {e}")
                        continue

        except Exception as e:
            logger.error(f"处理 C4 文件时出错: {e}")
            raise

        # 保存提取的数据
        self._save_extracted_data(extracted_data, output_path)

        # 打印统计信息
        self._print_stats()

        return self.stats

    def process_multiple_c4_files(self, c4_files: List[str], output_dir: str) -> Dict[str, int]:
        """
        处理多个 C4 文件

        Args:
            c4_files: C4 文件路径列表
            output_dir: 输出目录

        Returns:
            总处理统计信息
        """
        logger.info(f"开始处理 {len(c4_files)} 个 C4 文件")

        total_stats = {
            'total_records': 0,
            'successful_extractions': 0,
            'short_texts': 0,
            'empty_texts': 0,
            'files_processed': 0
        }

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for idx, c4_file in enumerate(c4_files, 1):
            logger.info(f"\n处理文件 {idx}/{len(c4_files)}: {c4_file}")

            # 为每个文件生成输出文件名
            file_name = Path(c4_file).stem
            output_file = f"{output_dir}/{file_name}_processed.jsonl"

            # 处理单个文件
            file_stats = self.process_c4_file(c4_file, output_file)

            # 累加统计信息
            for key in total_stats:
                if key in file_stats:
                    total_stats[key] += file_stats[key]
            total_stats['files_processed'] += 1

            # 重置当前文件统计
            self._reset_stats()

        # 打印总体统计
        self._print_total_stats(total_stats)

        return total_stats

    def _extract_text_from_c4_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 C4 记录中提取文本

        Args:
            record: C4 JSON 记录

        Returns:
            提取的数据字典，失败返回 None
        """
        try:
            # 获取文本内容
            text = record.get('text', '').strip()

            # 检查是否为空
            if not text:
                self.stats['empty_texts'] += 1
                return None

            # 检查文本长度
            if len(text) < self.min_text_length:
                self.stats['short_texts'] += 1
                return None

            self.stats['successful_extractions'] += 1

            return {
                'text': text,
                'url': record.get('url', ''),
                'timestamp': record.get('timestamp', ''),
                'length': len(text)
            }

        except Exception as e:
            logger.warning(f"提取文本失败: {e}")
            return None

    def _save_extracted_data(self, data: List[Dict], output_path: str):
        """保存提取的数据到 JSONL 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"保存了 {len(data)} 条记录到 {output_path}")

    def _print_stats(self):
        """打印处理统计信息"""
        total = self.stats['total_records']
        success = self.stats['successful_extractions']
        short = self.stats['short_texts']
        empty = self.stats['empty_texts']

        logger.info("=" * 50)
        logger.info("处理统计信息:")
        logger.info(f"  总记录数: {total}")
        logger.info(f"  成功提取: {success} ({success/total*100:.1f}%)")
        logger.info(f"  过短文本: {short} ({short/total*100:.1f}%)")
        logger.info(f"  空文本: {empty} ({empty/total*100:.1f}%)")
        logger.info(f"  最终产出率: {success/total*100:.1f}%")
        logger.info("=" * 50)

    def _print_total_stats(self, total_stats: Dict[str, int]):
        """打印总体处理统计信息"""
        total = total_stats['total_records']
        success = total_stats['successful_extractions']
        short = total_stats['short_texts']
        empty = total_stats['empty_texts']

        logger.info("\n" + "=" * 50)
        logger.info("总体处理统计信息:")
        logger.info(f"  处理文件数: {total_stats['files_processed']}")
        logger.info(f"  总记录数: {total}")
        logger.info(f"  成功提取: {success} ({success/total*100:.1f}%)")
        logger.info(f"  过短文本: {short} ({short/total*100:.1f}%)")
        logger.info(f"  空文本: {empty} ({empty/total*100:.1f}%)")
        logger.info(f"  最终产出率: {success/total*100:.1f}%")
        logger.info("=" * 50)

    def _reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_records': 0,
            'successful_extractions': 0,
            'short_texts': 0,
            'empty_texts': 0
        }

    def find_c4_files(self, directory: str, pattern: str = "c4-train.*.json") -> List[str]:
        """
        查找目录中的 C4 文件

        Args:
            directory: 搜索目录
            pattern: 文件匹配模式

        Returns:
            匹配的文件路径列表
        """
        import glob

        search_pattern = f"{directory}/{pattern}"
        files = sorted(glob.glob(search_pattern))

        logger.info(f"在 {directory} 中找到 {len(files)} 个匹配的 C4 文件")
        return files


def main():
    """主函数示例"""
    # 示例用法
    processor = C4Processor(min_text_length=100)

    # 查找 C4 文件
    c4_files = processor.find_c4_files("data/raw", "c4-train.*.json*")

    if not c4_files:
        print("未找到 C4 文件")
        return

    # 处理多个 C4 文件
    output_dir = "data/processed"
    processor.process_multiple_c4_files(c4_files, output_dir)

    print("C4 处理完成!")


if __name__ == "__main__":
    main()