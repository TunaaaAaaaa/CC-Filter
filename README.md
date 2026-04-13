# Mini-C4 分布式数据流水线

基于 Ray 构建的微缩版 C4 (Colossal Clean Crawled Corpus) 数据集流水线，将原始网页数据转化为高质量的预训练数据。

## 🎯 项目目标

将杂乱无章的原始网页数据（Common Crawl）转化为低噪、去重、高质量的纯文本数据，使其达到可以直接输入大语言模型进行预训练的标准。

## 🏗️ 技术架构

### 核心技术栈

| 组件 | 选型 | 决策理由 |
|------|------|----------|
| WARC 处理 | warcio + trafilatura | warcio 是处理 WARC 标准的官方库；trafilatura 在提取正文方面有显著优势 |
| 分布式计算 | Ray | 提供简单的 Actor 模型，能将 MinHash 计算并行化到多核 CPU |
| 去重算法 | MinHash LSH | 利用 LSH 将复杂度从 O(n²) 降为 O(n)，适合大规模数据处理 |
| 质量评估 | KenLM + FastText | KenLM 计算困惑度速度快；FastText 进行语言识别 |

### 数据流水线

```
Raw WARC → Text Extraction → Heuristic Filtering → MinHash Deduplication → LangID & PPL Filtering → Final Dataset
```

## 📦 安装

### 环境要求

- Python 3.8+
- 建议 16GB+ 内存
- 建议多核 CPU（Ray 分布式计算）

### 安装步骤

1. 克隆仓库
```bash
git clone <repository-url>
cd CC-FIlter
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. （可选）下载预训练模型

FastText 语言模型：
```bash
python 5_split_lang.py  # 会自动下载提示
```

KenLM 语言模型：
```bash
python 6_quality_filter.py  # 会显示生成模型的命令
```

## 🚀 使用方法

### 快速开始

```bash
python main.py --input data/raw/sample.warc.gz --output data/final/final_data.jsonl
```

### 完整参数

```bash
python main.py --input <warc_file> \
               --output <output_file> \
               --min-text-length 100 \
               --min-word-length 3 \
               --max-word-length 20 \
               --symbol-ratio-threshold 0.1 \
               --num-perm 128 \
               --dedup-threshold 0.8 \
               --batch-size 1000 \
               --target-languages en zh \
               --perplexity-threshold -6.0
```

### 使用配置文件

创建 `config.json`：
```json
{
  "min_text_length": 100,
  "num_perm": 128,
  "dedup_threshold": 0.8,
  "target_languages": ["en"]
}
```

运行：
```bash
python main.py --input data/raw/sample.warc.gz --config config.json
```

## 📊 处理效果

### 数据漏斗统计

| 处理阶段 | 输入记录数 | 输出记录数 | 留存率 | 主要损耗原因 |
|---------|-----------|-----------|--------|------------|
| 原始 WARC | ~35,000 | ~10,000 | 28% | 非 HTML 响应、空内容 |
| 启发式清洗 | 10,000 | ~6,500 | 65% | 长度过短、符号密度过高 |
| 去重 (LSH) | 6,500 | ~4,800 | 73% | 转载文章、镜像站点 |
| 语言/质量过滤 | 4,800 | ~3,900 | 81% | 非英文内容、高困惑度 |
| **Total** | **35,000** | **3,900** | **~11%** | 最终产出率 |

### 过滤效果对比

✅ **高质量正文**（保留）
> "The James Webb Space Telescope has captured a new image of the Pillars of Creation..."
> KenLM Score: -4.82

❌ **导航栏噪声**（拒绝）
> "Home | About Us | Contact | Enable Cookies | Copyright 2023..."
> 触发: 短文本和关键词黑名单

❌ **代码片段**（拒绝）
> "function(x) { return x > 0 ? true : false; } var a = [1,2,3];"
> 触发: 符号密度 > 10%

## 🔧 模块说明

### 1. WARC 处理 (`2_process_warc.py`)
- 流式读取 WARC 压缩文件
- 使用 trafilatura 提取网页正文
- 过滤非 HTML 内容

### 2. 数据清洗 (`3_clean_data.py`)
- 启发式规则过滤低质量文本
- 符号密度检查
- 黑名单关键词过滤
- 重复字符检测

### 3. 去重 (`4_deduplicate.py`)
- 使用 Ray 并行计算 MinHash
- LSH 索引快速检测重复
- 支持大规模数据去重

### 4. 语言识别 (`5_split_lang.py`)
- FastText 语言模型识别
- 支持多语言分流
- 按置信度过滤

### 5. 质量过滤 (`6_quality_filter.py`)
- KenLM 困惑度计算
- 统计语言模型评估
- 基于质量阈值过滤

## 📈 性能指标

### 单机性能
- 处理速度: 1GB WARC / 5-8 分钟（16核 CPU）
- 内存使用: < 16GB
- 最终产出率: ~11%

### 扩展性
- 支持多机集群部署
- 水平扩展能力良好
- 瓶颈在于 LSH 索引构建

## ⚙️ 配置说明

### MinHash 参数
- `num_perm=128`: 置换函数数量（C4 标准）
- `threshold=0.8`: LSH 相似度阈值

### 困惑度过滤
- `threshold=-6.0`: 归一化 log score 阈值
- 高质量范围: > -5.0
- 中等质量: -5.0 ~ -6.0
- 低质量: < -6.5

### 清洗规则
- 符号密度阈值: 10%
- 最小文本长度: 100 字符
- 平均词长范围: 3-20 字符

## 🐛 故障排除

### 常见问题

1. **Ray 初始化失败**
   - 确保有足够内存
   - 检查端口占用
   - 使用 `--no-ray` 参数禁用 Ray

2. **模型文件不存在**
   - 下载 FastText 模型
   - 生成 KenLM 模型

3. **内存不足**
   - 减小 batch_size
   - 使用更少的 MinHash permutations

## 📝 项目结构

```
CC-FIlter/
├── main.py                  # 主入口
├── 2_process_warc.py        # WARC 处理
├── 3_clean_data.py          # 数据清洗
├── 4_deduplicate.py         # 去重
├── 5_split_lang.py          # 语言识别
├── 6_quality_filter.py      # 质量过滤
├── requirements.txt         # 依赖包
├── PLAN.md                  # 项目计划
├── README.md                # 项目说明
├── data/                    # 数据目录
├── models/                  # 模型目录
└── logs/                    # 日志目录
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

- [Common Crawl](https://commoncrawl.org/) - 原始数据源
- [Trafilatura](https://github.com/adbar/trafilatura) - 文本提取
- [Ray](https://ray.io/) - 分布式计算框架
- [FastText](https://fasttext.cc/) - 语言识别
- [KenLM](https://github.com/kpu/kenlm) - 语言模型

## 📧 联系方式

如有问题或建议，请创建 Issue 或联系维护者。

---

**注意**: 本项目主要用于教育和研究目的。实际生产环境使用时，请根据具体需求进行优化和调整。