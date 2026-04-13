"""
C4 数据集处理流水线 - 主入口
整合所有处理模块，完成从 C4 JSON 到高质量数据集的完整流程
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class C4Pipeline:
    """C4 数据处理流水线"""

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

    def run_full_pipeline(self, input_pattern: str, output_file: str):
        """
        运行完整的流水线

        Args:
            input_pattern: 输入文件模式（如 "data/raw/c4-train.*.json"）
            output_file: 最终输出文件路径
        """
        logger.info("=" * 60)
        logger.info("开始运行 C4 数据处理流水线")
        logger.info("=" * 60)

        try:
            # 阶段 1: C4 数据预处理
            logger.info("\n【阶段 1/5】C4 数据预处理...")
            self._step1_process_c4(input_pattern)

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

    def _step1_process_c4(self, input_pattern: str):
        """步骤 1: 处理 C4 数据"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("c4_processor", "1_process_c4.py")
            c4_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(c4_module)

            processor = c4_module.C4Processor(
                min_text_length=self.config.get('min_text_length', 100)
            )

            # 查找匹配的文件
            import glob
            c4_files = sorted(glob.glob(input_pattern))

            if not c4_files:
                raise FileNotFoundError(f"未找到匹配的文件: {input_pattern}")

            logger.info(f"找到 {len(c4_files)} 个 C4 文件")

            # 合并所有处理后的文件到一个输出文件
            output_dir = 'data/processed'
            all_processed_data = []

            for c4_file in c4_files:
                logger.info(f"处理文件: {c4_file}")
                file_name = Path(c4_file).stem
                output_file = f"{output_dir}/{file_name}_processed.jsonl"
                stats = processor.process_c4_file(c4_file, output_file)

                # 读取处理后的数据以便后续合并
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_processed_data.append(json.loads(line))

            # 合并所有数据到一个文件
            merged_output = 'data/processed/c4_merged.jsonl'
            with open(merged_output, 'w', encoding='utf-8') as f:
                for item in all_processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            total_stats = {
                'total_records': sum(s['total_records'] for s in [processor.stats]),
                'successful_extractions': len(all_processed_data),
                'short_texts': sum(s['short_texts'] for s in [processor.stats]),
                'empty_texts': sum(s['empty_texts'] for s in [processor.stats])
            }

            self.results['step1'] = {
                'name': 'C4 数据预处理',
                'input': input_pattern,
                'output': merged_output,
                'stats': total_stats
            }

            logger.info(f"步骤 1 完成: 处理了 {len(c4_files)} 个文件，提取了 {total_stats['successful_extractions']} 条文本记录")

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
            output_path = 'data/processed/c4_cleaned.jsonl'
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
            output_path = 'data/processed/c4_deduplicated.jsonl'
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
        logger.info("C4 数据处理流水线执行完成")
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
        report_path = 'logs/c4_pipeline_report.json'

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
    parser = argparse.ArgumentParser(description='C4 数据集处理流水线')

    parser.add_argument('--input', '-i', default='data/raw/c4-train.*.json',
                       help='输入文件模式 (默认: data/raw/c4-train.*.json)')
    parser.add_argument('--output', '-o', default='data/final/c4_final_data.jsonl',
                       help='输出文件路径 (默认: data/final/c4_final_data.jsonl)')
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
        'perplexity_threshold': args.perplexity_threshold
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
        pipeline = C4Pipeline(config)
        pipeline.run_full_pipeline(args.input, args.output)
        logger.info("流水线执行成功!")
        return 0
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())