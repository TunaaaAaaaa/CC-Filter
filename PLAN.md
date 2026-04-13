# Mini-C4 分布式数据流水线项目计划

## 项目概述
基于 Ray 构建微缩版 C4 (Colossal Clean Crawled Corpus) 数据集流水线，将原始网页数据转化为高质量训练数据。

## 技术架构

### 核心组件
- **WARC 处理**: warcio + trafilatura (文本提取)
- **分布式计算**: Ray (并行化 MinHash 计算)
- **去重算法**: MinHash LSH (模糊去重)
- **质量评估**: KenLM + FastText (困惑度 & 语言识别)

### 数据流水线
```
Raw WARC → Text Extraction → Heuristic Filtering → MinHash Deduplication → LangID & PPL Filtering → Final Dataset
```

## 实施计划

### 阶段一：基础架构搭建 (Week 1-2)
- [x] 创建项目结构和文档
- [x] 实现 WARC 文件解析和文本提取
- [ ] 添加启发式清洗规则
- [ ] 配置 Ray 分布式环境

### 阶段二：核心功能开发 (Week 3-4)
- [ ] 实现 MinHash 去重算法
- [ ] 集成 FastText 语言识别
- [ ] 实现 KenLM 困惑度过滤
- [ ] 开发主流程编排

### 阶段三：优化和测试 (Week 5-6)
- [ ] 性能调优和并行优化
- [ ] 参数调优 (MinHash perm数, 困惑度阈值等)
- [ ] 单元测试和集成测试
- [ ] 文档完善

## 技术规格

### MinHash 参数
- `num_perm=128`: C4 标准参数
- LSH 阈值: 0.8 (根据 Jaccard 相似度)

### 困惑度过滤
- 阈值: `-6.0` (归一化 log score)
- 语言模型: KenLM (en.arpa.bin)

### 语言识别
- 模型: FastText lid.176.ftz
- 目标语言: 英语

## 预期成果

### 数据质量指标
- 最终产出率: ~11% (从原始 WARC)
- 去重效果: 重复率降低 ~25%
- 质量提升: 高质量文本占比 > 80%

### 性能指标
- 处理速度: 1GB WARC / 5-8 分钟 (16核 CPU)
- 内存使用: < 16GB (单机)
- 可扩展性: 支持集群部署

## 文件结构
```
CC-FIlter/
├── PLAN.md                  # 项目计划 (本文件)
├── README.md                # 项目说明
├── requirements.txt         # Python 依赖
├── main.py                  # 主入口
├── 2_process_warc.py        # WARC 处理
├── 3_clean_data.py          # 数据清洗
├── 4_deduplicate.py         # 去重
├── 5_split_lang.py          # 语言识别
├── 6_quality_filter.py      # 质量过滤
├── models/                  # 模型目录
│   ├── lid.176.ftz         # FastText 模型
│   └── en.arpa.bin         # KenLM 模型
├── data/                    # 数据目录
│   ├── raw/                # 原始 WARC 文件
│   ├── processed/          # 处理后的数据
│   └── final/              # 最终输出
└── tests/                   # 测试目录
    └── test_pipeline.py    # 测试文件
```

## 风险和挑战

### 技术风险
1. **内存限制**: 大规模 LSH 索引可能超出单机内存
   - 解决方案: 实现基于 Redis 的分布式存储

2. **模型依赖**: KenLM 和 FastText 模型需要下载
   - 解决方案: 提供自动下载脚本

3. **计算资源**: MinHash 计算开销大
   - 解决方案: Ray 集群扩展和批处理优化

### 数据质量风险
1. **阈值设定**: 困惑度阈值需要根据数据调优
   - 解决方案: 提供阈值建议和自动调参脚本

2. **语言覆盖**: 主要针对英语优化
   - 解决方案: 支持多语言扩展

## 后续扩展
1. 支持更多语言的模型
2. 实现 GPU 加速
3. 添加 Web 管理界面
4. 集成到 MLOps 流水线

## 贡献指南
欢迎提交 Issue 和 Pull Request，请遵循项目规范和代码风格。

## 许可证
MIT License