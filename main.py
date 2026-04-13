"""
Mini-C4 分布式数据流水线 - 主入口
整合所有处理模块，完成从原始 WARC 到高质量数据集的完整流程
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import time

# 导入所有处理模块
from warcio.archiveiterator import ArchiveIterator  # 用于检查依赖

# 导入自定义模块
try:
    from importlib import import_module
    import importlib.util

    # 尝试导入各个处理模块
    _ = import_module('2_process_warc')
    _ = import_module('3_clean_data')
    _ = import_module('4_deduplicate')
    _ = import_module('5_split_lang')
    _ = import_module('6_quality_filter')

except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有处理模块都在同一目录下")
    sys.exit(1)

# 导入具体功能
from warcio.archiveiterator import ArchiveIterator
import trafilatura
import ray
from datasketch import MinHash, MinHashLSH

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MiniC4Pipeline:
    """Mini-C4 完整数据流水线"""

    def __init__(self, config: dict):
        """
        初始化流水线

        Args:
            config: 配置字典
        """
        self.config = config
        self.results = {}
        self.start_time = time.time()

        # 确保目录存在
        self._create_directories()

        # 初始化 Ray（如果需要）
        if config.get('use_ray', True):
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                logger.info("Ray 初始化成功")
            except Exception as e:
                logger.warning(f"Ray 初始化失败: {e}")

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            'data/raw',
            'data/processed',
            'data/split_by_language',
            'data/final',
            'models',
            'logs'
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        logger.info("目录结构创建完成")

    def run_full_pipeline(self, input_warc: str, output_file: str):
        """
        运行完整的流水线

        Args:
            input_warc: 输入 WARC 文件路径
            output_file: 最终输出文件路径
        """
        logger.info("=" * 60)
        logger.info("开始运行 Mini-C4 完整数据流水线")
        logger.info("=" * 60)

        try:
            # 阶段 1: WARC 处理
            logger.info("\n【阶段 1/5】WARC 文本提取...")
            self._step1_extract_text(input_warc)

            # 阶段 2: 数据清洗
            logger.info("\n【阶段 2/5】启发式数据清洗...")
            self._step2_clean_data()

            # 阶段 3: 去重
            logger.info("\n【阶段 3/5】MinHash 去重...")
            self._step3_deduplicate()

            # 阶段 4: 语言识别
            logger.info("\n【阶段 4/5】语言识别...")
            self._step4_language_split()

            # 阶段 5: 质量过滤
            logger.info("\n【阶段 5/5】质量过滤...")
            self._step5_quality_filter(output_file)

            # 完成
            self._print_final_summary()

        except Exception as e:
            logger.error(f"流水线运行失败: {e}")
            raise

    def _step1_extract_text(self, input_warc: str):
        """步骤 1: 从 WARC 提取文本"""
        try:
            # 动态导入模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("warc_processor", "2_process_warc.py")
            warc_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(warc_module)

            processor = warc_module.WARCProcessor(
                min_text_length=self.config.get('min_text_length', 100)
            )

            output_path = 'data/processed/extracted.jsonl'
            stats = processor.process_warc_file(input_warc, output_path)

            self.results['step1'] = {
                'name': 'WARC 文本提取',
                'input': input_warc,
                'output': output_path,
                'stats': stats
            }

            logger.info(f"步骤 1 完成: 提取了 {stats['successful_extractions']} 条文本记录")

        except Exception as e:
            logger.error(f"步骤 1 失败: {e}")
            raise

    def _step2_clean_data(self):
        """步骤 2: 数据清洗"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("data_cleaner", "3_clean_data.py")
            clean_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(clean_module)

            cleaner = clean_module.DataCleaner(
                min_word_length=self.config.get('min_word_length', 3),
                max_word_length=self.config.get('max_word_length', 20),
                symbol_ratio_threshold=self.config.get('symbol_ratio_threshold', 0.1)
            )

            input_path = self.results['step1']['output']
            output_path = 'data/processed/cleaned.jsonl'
            stats = cleaner.clean_file(input_path, output_path)

            self.results['step2'] = {
                'name': '数据清洗',
                'input': input_path,
                'output': output_path,
                'stats': stats
            }

            logger.info(f"步骤 2 完成: 保留了 {stats['passed']} 条清洗后记录")

        except Exception as e:
            logger.error(f"步骤 2 失败: {e}")
            raise

    def _step3_deduplicate(self):
        """步骤 3: 去重"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("deduplicator", "4_deduplicate.py")
            dedup_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dedup_module)

            deduplicator = dedup_module.DistributedDeduplicator(
                num_perm=self.config.get('num_perm', 128),
                threshold=self.config.get('dedup_threshold', 0.8),
                batch_size=self.config.get('batch_size', 1000)
            )

            input_path = self.results['step2']['output']
            output_path = 'data/processed/deduplicated.jsonl'
            stats = deduplicator.deduplicate_file(input_path, output_path)

            self.results['step3'] = {
                'name': 'MinHash 去重',
                'input': input_path,
                'output': output_path,
                'stats': stats
            }

            logger.info(f"步骤 3 完成: 去除了 {stats['duplicates']} 条重复记录")

        except Exception as e:
            logger.error(f"步骤 3 失败: {e}")
            raise

    def _step4_language_split(self):
        """步骤 4: 语言识别"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("lang_splitter", "5_split_lang.py")
            lang_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lang_module)

            splitter = lang_module.LanguageSplitter(
                model_path=self.config.get('fasttext_model', 'models/lid.176.ftz'),
                target_languages=self.config.get('target_languages', ['en']),
                min_confidence=self.config.get('min_lang_confidence', 0.5)
            )

            input_path = self.results['step3']['output']
            output_dir = 'data/split_by_language'
            stats = splitter.split_file_by_language(input_path, output_dir)

            self.results['step4'] = {
                'name': '语言识别',
                'input': input_path,
                'output': output_dir,
                'stats': stats
            }

            logger.info(f"步骤 4 完成: 识别了 {stats['identified']} 条记录的语言")

        except Exception as e:
            logger.error(f"步骤 4 失败: {e}")
            raise

    def _step5_quality_filter(self, output_file: str):
        """步骤 5: 质量过滤"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("quality_filter", "6_quality_filter.py")
            quality_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(quality_module)

            filter = quality_module.QualityFilter(
                model_path=self.config.get('kenlm_model', 'models/en.arpa.bin'),
                min_length=self.config.get('min_quality_length', 20),
                perplexity_threshold=self.config.get('perplexity_threshold', -6.0)
            )

            # 使用目标语言的数据
            target_lang = self.config.get('target_languages', ['en'])[0]
            input_path = f"data/split_by_language/{target_lang}/data.jsonl"

            # 检查文件是否存在
            if not Path(input_path).exists():
                logger.warning(f"目标语言文件不存在: {input_path}")
                # 使用上一步的输出
                input_path = self.results['step4']['input']

            stats = filter.filter_file(input_path, output_file)

            self.results['step5'] = {
                'name': '质量过滤',
                'input': input_path,
                'output': output_file,
                'stats': stats
            }

            logger.info(f"步骤 5 完成: 输出了 {stats['passed']} 条高质量记录")

        except Exception as e:
            logger.error(f"步骤 5 失败: {e}")
            raise

    def _print_final_summary(self):
        """打印最终汇总"""
        total_time = time.time() - self.start_time

        logger.info("\n" + "=" * 60)
        logger.info("Mini-C4 流水线执行完成")
        logger.info("=" * 60)

        logger.info(f"\n总执行时间: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")

        logger.info("\n各阶段处理结果:")

        # 计算总体留存率
        initial_count = self.results['step1']['stats']['successful_extractions']
        final_count = self.results['step5']['stats']['passed']
        overall_rate = (final_count / initial_count * 100) if initial_count > 0 else 0

        for step_key, step_data in self.results.items():
            stats = step_data['stats']
            logger.info(f"\n{step_data['name']}:")
            logger.info(f"  输入: {step_data['input']}")
            logger.info(f"  输出: {step_data['output']}")

            # 根据不同步骤显示不同的统计信息
            if 'successful_extractions' in stats:
                logger.info(f"  成功提取: {stats['successful_extractions']} 条")
            elif 'passed' in stats:
                logger.info(f"  通过记录: {stats['passed']} 条")
                logger.info(f"  拒绝记录: {stats['total'] - stats['passed']} 条")

        logger.info(f"\n总体统计:")
        logger.info(f"  初始记录数: {initial_count}")
        logger.info(f"  最终记录数: {final_count}")
        logger.info(f"  总体留存率: {overall_rate:.1f}%")
        logger.info("=" * 60)

        # 保存结果报告
        self._save_results_report()

    def _save_results_report(self):
        """保存结果报告"""
        report_path = 'logs/pipeline_report.json'

        report = {
            'pipeline_config': self.config,
            'processing_steps': {},
            'summary': {
                'total_time': time.time() - self.start_time,
                'initial_records': self.results['step1']['stats']['successful_extractions'],
                'final_records': self.results['step5']['stats']['passed'],
                'overall_retention_rate': (self.results['step5']['stats']['passed'] /
                                         self.results['step1']['stats']['successful_extractions'] * 100)
                                         if self.results['step1']['stats']['successful_extractions'] > 0 else 0
            }
        }

        for step_key, step_data in self.results.items():
            report['processing_steps'][step_key] = {
                'name': step_data['name'],
                'stats': step_data['stats']
            }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"结果报告已保存到: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Mini-C4 分布式数据流水线')

    parser.add_argument('--input', '-i', required=True, help='输入 WARC 文件路径')
    parser.add_argument('--output', '-o', default='data/final/final_data.jsonl',
                       help='输出文件路径 (默认: data/final/final_data.jsonl)')
    parser.add_argument('--config', '-c', help='配置文件路径 (JSON 格式)')

    parser.add_argument('--min-text-length', type=int, default=100,
                       help='最小文本长度 (默认: 100)')
    parser.add_argument('--min-word-length', type=int, default=3,
                       help='最小单词长度 (默认: 3)')
    parser.add_argument('--max-word-length', type=int, default=20,
                       help='最大单词长度 (默认: 20)')
    parser.add_argument('--symbol-ratio-threshold', type=float, default=0.1,
                       help='符号密度阈值 (默认: 0.1)')
    parser.add_argument('--num-perm', type=int, default=128,
                       help='MinHash 置换函数数量 (默认: 128)')
    parser.add_argument('--dedup-threshold', type=float, default=0.8,
                       help='去重相似度阈值 (默认: 0.8)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='批处理大小 (默认: 1000)')
    parser.add_argument('--target-languages', nargs='+', default=['en'],
                       help='目标语言列表 (默认: en)')
    parser.add_argument('--perplexity-threshold', type=float, default=-6.0,
                       help='困惑度阈值 (默认: -6.0)')
    parser.add_argument('--no-ray', action='store_true',
                       help='不使用 Ray 进行分布式处理')

    args = parser.parse_args()

    # 构建配置
    config = {
        'min_text_length': args.min_text_length,
        'min_word_length': args.min_word_length,
        'max_word_length': args.max_word_length,
        'symbol_ratio_threshold': args.symbol_ratio_threshold,
        'num_perm': args.num_perm,
        'dedup_threshold': args.dedup_threshold,
        'batch_size': args.batch_size,
        'target_languages': args.target_languages,
        'perplexity_threshold': args.perplexity_threshold,
        'use_ray': not args.no_ray
    }

    # 如果提供了配置文件，合并配置
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            config.update(file_config)
            logger.info(f"已加载配置文件: {args.config}")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")

    # 运行流水线
    try:
        pipeline = MiniC4Pipeline(config)
        pipeline.run_full_pipeline(args.input, args.output)
        logger.info("流水线执行成功!")
        return 0
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())