# Dataset Crawler

一个用于从 Hugging Face Hub 爬取数据集元数据，并使用 LLM 进行智能分类和过滤的工具集。

[English](README.md) | 中文

## 项目概述

Dataset Crawler 是一个自动化工具集，用于：
- 从 Hugging Face Hub 爬取数据集元数据
- 处理和过滤数据集信息
- 使用 LLM 对数据集进行多领域分类（Agent、Finance、Benchmark 等）
- 生成可读的 Excel 报告

## 功能特性

### 核心功能

1. **数据爬取** (`1_crawler.py`)
   - 从 Hugging Face Hub 获取最新数据集元数据
   - 支持限制爬取数量
   - 保存为 pickle 格式

2. **数据处理** (`2_process.py`)
   - 处理爬取的元数据
   - 过滤和清洗数据
   - 转换为 JSONL 格式

3. **智能分类**
   - **Weekly 分类** (`weekly/weekly.py`): 对数据集进行多类别分类（Math、Code、Science、Medical、Legal、Finance、Other）
   - **Domain 分类** (`domain/domain.py`): 统一的域特定分类工具
     - `agent`: LLM agent 数据集粗粒度过滤
     - `agent_specific`: LLM agent 数据集细粒度高质量过滤
     - `finance`: 金融领域数据集过滤
   - **Benchmark 分类** (`benchmark/benchmark.py`): 识别和分类基准测试数据集

4. **工具集** (`utils/`)
   - JSONL ↔ Excel 转换
   - PKL ↔ JSONL 转换
   - 进度监控

## 项目结构

```
DatasetCrawler/
├── 1_crawler.py              # 数据爬取脚本
├── 2_process.py              # 数据处理脚本
├── config/                   # 配置文件
│   ├── api.json             # API 配置（base_url, api_key, model, temperature）
│   └── domains/             # Domain 分类配置
│       ├── agent.json
│       ├── agent_specific.json
│       └── finance.json
├── domain/                  # Domain 分类模块
│   ├── domain.py           # 统一的 domain 分类器
│   └── README.md
├── weekly/                  # Weekly 分类模块
│   └── weekly.py
├── benchmark/               # Benchmark 分类模块
│   └── benchmark.py
├── utils/                   # 工具函数
│   ├── json2xlsx.py
│   ├── pkl2jsonl.py
│   └── xlsx2jsonl.py
├── scripts/                 # 批处理脚本
│   ├── weekly/
│   │   └── weekly.sh
│   ├── domain/
│   │   └── domain.sh
│   ├── benchmark/
│   │   └── bench.sh
│   └── monitor_progress.sh
└── data/                    # 数据输出目录
    ├── weekly/
    ├── domain/
    └── benchmark/
```

## 快速开始

### 1. 配置 API

编辑 `config/api.json` 设置你的 API 配置：

```json
{
    "base_url": "http://your-api-endpoint/v1/chat/completions",
    "api_key": "your-api-key",
    "model": "gpt-4o-mini-2025-07-16",
    "temperature": 0.1
}
```

### 2. 数据爬取和处理

```bash
# 爬取数据集元数据
python 1_crawler.py -o "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" --limit 5000

# 处理数据
python 2_process.py -i "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl"
```

### 3. 分类任务

#### Weekly 分类

```bash
# Python 脚本
python weekly/weekly.py \
    -i "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl"

# 或使用批处理脚本
sbatch scripts/weekly/weekly.sh
```

#### Domain 分类

```bash
# Agent 粗粒度过滤
python domain/domain.py --domain agent \
    -i "data/domain/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_2025-01-01.jsonl"

# Agent 细粒度过滤（需要先运行 agent）
python domain/domain.py --domain agent_specific \
    -i "data/domain/2025-01-01/agent_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_specific_2025-01-01.jsonl" \
    --min_quality_score 6 \
    --min_confidence 0.7

# Finance 过滤
python domain/domain.py --domain finance \
    -i "data/domain/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/finance_2025-01-01.jsonl"

# 或使用批处理脚本
sbatch scripts/domain/domain.sh agent
sbatch scripts/domain/domain.sh agent_specific
sbatch scripts/domain/domain.sh finance
```

#### Benchmark 分类

```bash
# Python 脚本
python benchmark/benchmark.py \
    -i "data/benchmark/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/benchmark/2025-01-01/3_classified_2025-01-01.jsonl"

# 或使用批处理脚本
sbatch scripts/benchmark/bench.sh
```

### 4. 格式转换

```bash
# JSONL 转 Excel
python utils/json2xlsx.py \
    -i "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.xlsx"

# PKL 转 JSONL
python utils/pkl2jsonl.py \
    -i "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.jsonl"
```

## 完整工作流示例

### Weekly 工作流

```bash
# 1. 爬取数据
python 1_crawler.py -o "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" --limit 5000

# 2. 处理数据
python 2_process.py \
    -i "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl"

# 3. 分类
python weekly/weekly.py \
    -i "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl"

# 4. 转换为 Excel
python utils/json2xlsx.py \
    -i "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.xlsx"
```

或使用一键脚本：

```bash
sbatch scripts/weekly/weekly.sh
```

### Domain 工作流（Agent）

```bash
# 1. 爬取和处理（同上）
python 1_crawler.py -o "data/domain/2025-01-01/1_crawler_2025-01-01.pkl" --limit 5000
python 2_process.py \
    -i "data/domain/2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/domain/2025-01-01/2_process_2025-01-01.jsonl"

# 2. Agent 粗粒度过滤
python domain/domain.py --domain agent \
    -i "data/domain/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_2025-01-01.jsonl"

# 3. Agent 细粒度过滤
python domain/domain.py --domain agent_specific \
    -i "data/domain/2025-01-01/agent_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_specific_2025-01-01.jsonl" \
    --min_quality_score 6 \
    --min_confidence 0.7
```

## 配置说明

### API 配置 (`config/api.json`)

所有脚本的 API 参数（base_url, api_key, model, temperature）都从 `config/api.json` 读取，支持通过命令行参数覆盖。

### Domain 配置 (`config/domains/`)

每个 domain 的配置存储在独立的 JSON 文件中，包括：
- 标签/类别定义
- System prompt
- User prompt 模板
- 响应格式定义
- 过滤条件（min_confidence, min_quality_score 等）

## 主要特性

### 1. 统一的 Domain 分类系统

- 通过配置文件定义不同的分类域
- 支持扩展新的 domain（只需添加配置文件）
- 统一的接口和参数

### 2. Checkpoint 支持

所有分类脚本都支持 checkpoint，可以中断后继续处理：
- 自动保存处理进度
- 支持断点续传
- 避免重复处理

### 3. 并发处理

- 支持高并发 API 请求
- 可配置并发数量
- 自动重试机制

### 4. 批量处理

- 支持批量处理大量数据集
- 可配置批次大小
- 增量保存结果

## 依赖要求

```bash
pip install huggingface_hub openai aiofiles tqdm pandas openpyxl
```

## 输出格式

### JSONL 格式

每行一个 JSON 对象，包含：
- 原始数据集元数据
- 分类结果（confidence, tags, reasoning 等）
- 时间戳和模型信息

### Excel 格式

包含所有字段的表格，便于人工审查和筛选。

## 监控和调试

### 进度监控

```bash
# 监控处理进度
bash scripts/monitor_progress.sh
```

### 日志

所有脚本都输出详细的日志信息，包括：
- 处理进度
- 错误信息
- 统计信息

## 常见问题

### Q: 如何添加新的 domain？

A: 在 `config/domains/` 目录下创建新的 JSON 配置文件，定义标签、prompt 模板等。然后在 `domain.py` 的 `choices` 中添加新 domain（可选）。

### Q: 如何处理大量数据？

A: 使用 checkpoint 功能，支持中断后继续。也可以调整 `--batch_size` 和 `--max_concurrent` 参数。

### Q: API 调用失败怎么办？

A: 脚本内置了重试机制，会自动重试失败的请求。也可以调整 `--max_retries` 参数。

## 许可证

[添加许可证信息]

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

[添加联系方式]

