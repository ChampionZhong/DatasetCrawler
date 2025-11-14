# Dataset Crawler

A toolkit for crawling dataset metadata from Hugging Face Hub and performing intelligent classification and filtering using LLM.

[English](README.md) | [中文](README_zh.md)

## Overview

Dataset Crawler is an automated toolkit for:
- Crawling dataset metadata from Hugging Face Hub
- Processing and filtering dataset information
- Multi-domain classification using LLM (Agent, Finance, Benchmark, etc.)
- Generating readable Excel reports

## Features

### Core Functionality

1. **Data Crawling** (`1_crawler.py`)
   - Fetch latest dataset metadata from Hugging Face Hub
   - Support limiting the number of datasets to crawl
   - Save as pickle format

2. **Data Processing** (`2_process.py`)
   - Process crawled metadata
   - Filter and clean data
   - Convert to JSONL format

3. **Intelligent Classification**
   - **Weekly Classification** (`weekly/weekly.py`): Multi-category classification (Math, Code, Science, Medical, Legal, Finance, Other)
   - **Domain Classification** (`domain/domain.py`): Unified domain-specific classification tool
     - `agent`: Coarse-grained filtering for LLM agent datasets
     - `agent_specific`: Fine-grained high-quality filtering for LLM agent datasets
     - `finance`: Finance domain dataset filtering
   - **Benchmark Classification** (`benchmark/benchmark.py`): Identify and classify benchmark datasets

4. **Utilities** (`utils/`)
   - JSONL ↔ Excel conversion
   - PKL ↔ JSONL conversion
   - Progress monitoring

## Project Structure

```
DatasetCrawler/
├── 1_crawler.py              # Data crawling script
├── 2_process.py              # Data processing script
├── config/                   # Configuration files
│   ├── api.json             # API config (base_url, api_key, model, temperature)
│   └── domains/             # Domain classification configs
│       ├── agent.json
│       ├── agent_specific.json
│       └── finance.json
├── domain/                  # Domain classification module
│   ├── domain.py           # Unified domain classifier
│   └── README.md
├── weekly/                  # Weekly classification module
│   └── weekly.py
├── benchmark/               # Benchmark classification module
│   └── benchmark.py
├── utils/                   # Utility functions
│   ├── json2xlsx.py
│   ├── pkl2jsonl.py
│   └── xlsx2jsonl.py
├── scripts/                 # Batch processing scripts
│   ├── weekly/
│   │   └── weekly.sh
│   ├── domain/
│   │   └── domain.sh
│   ├── benchmark/
│   │   └── bench.sh
│   └── monitor_progress.sh
└── data/                    # Data output directory
    ├── weekly/
    ├── domain/
    └── benchmark/
```

## Quick Start

### 1. Configure API

Edit `config/api.json` to set your API configuration:

```json
{
    "base_url": "http://your-api-endpoint/v1/chat/completions",
    "api_key": "your-api-key",
    "model": "gpt-4o-mini-2025-07-16",
    "temperature": 0.1
}
```

### 2. Data Crawling and Processing

```bash
# Crawl dataset metadata
python 1_crawler.py -o "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" --limit 5000

# Process data
python 2_process.py -i "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl"
```

### 3. Classification Tasks

#### Weekly Classification

```bash
# Python script
python weekly/weekly.py \
    -i "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl"

# Or use batch script
sbatch scripts/weekly/weekly.sh
```

#### Domain Classification

```bash
# Agent coarse-grained filtering
python domain/domain.py --domain agent \
    -i "data/domain/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_2025-01-01.jsonl"

# Agent fine-grained filtering (requires running agent first)
python domain/domain.py --domain agent_specific \
    -i "data/domain/2025-01-01/agent_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_specific_2025-01-01.jsonl" \
    --min_quality_score 6 \
    --min_confidence 0.7

# Finance filtering
python domain/domain.py --domain finance \
    -i "data/domain/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/finance_2025-01-01.jsonl"

# Or use batch scripts
sbatch scripts/domain/domain.sh agent
sbatch scripts/domain/domain.sh agent_specific
sbatch scripts/domain/domain.sh finance
```

#### Benchmark Classification

```bash
# Python script
python benchmark/benchmark.py \
    -i "data/benchmark/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/benchmark/2025-01-01/3_classified_2025-01-01.jsonl"

# Or use batch script
sbatch scripts/benchmark/bench.sh
```

### 4. Format Conversion

```bash
# JSONL to Excel
python utils/json2xlsx.py \
    -i "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.xlsx"

# PKL to JSONL
python utils/pkl2jsonl.py \
    -i "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.jsonl"
```

## Complete Workflow Examples

### Weekly Workflow

```bash
# 1. Crawl data
python 1_crawler.py -o "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" --limit 5000

# 2. Process data
python 2_process.py \
    -i "data/weekly/weekly_2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl"

# 3. Classify
python weekly/weekly.py \
    -i "data/weekly/weekly_2025-01-01/2_processed_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl"

# 4. Convert to Excel
python utils/json2xlsx.py \
    -i "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.jsonl" \
    -o "data/weekly/weekly_2025-01-01/3_classified_2025-01-01.xlsx"
```

Or use the one-click script:

```bash
sbatch scripts/weekly/weekly.sh
```

### Domain Workflow (Agent)

```bash
# 1. Crawl and process (same as above)
python 1_crawler.py -o "data/domain/2025-01-01/1_crawler_2025-01-01.pkl" --limit 5000
python 2_process.py \
    -i "data/domain/2025-01-01/1_crawler_2025-01-01.pkl" \
    -o "data/domain/2025-01-01/2_process_2025-01-01.jsonl"

# 2. Agent coarse-grained filtering
python domain/domain.py --domain agent \
    -i "data/domain/2025-01-01/2_process_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_2025-01-01.jsonl"

# 3. Agent fine-grained filtering
python domain/domain.py --domain agent_specific \
    -i "data/domain/2025-01-01/agent_2025-01-01.jsonl" \
    -o "data/domain/2025-01-01/agent_specific_2025-01-01.jsonl" \
    --min_quality_score 6 \
    --min_confidence 0.7
```

## Configuration

### API Configuration (`config/api.json`)

All scripts read API parameters (base_url, api_key, model, temperature) from `config/api.json`, with support for overriding via command-line arguments.

### Domain Configuration (`config/domains/`)

Each domain's configuration is stored in a separate JSON file, including:
- Tag/category definitions
- System prompt
- User prompt template
- Response format definitions
- Filtering conditions (min_confidence, min_quality_score, etc.)

## Key Features

### 1. Unified Domain Classification System

- Define different classification domains through configuration files
- Support for extending new domains (just add a config file)
- Unified interface and parameters

### 2. Checkpoint Support

All classification scripts support checkpoints, allowing resumption after interruption:
- Automatic progress saving
- Support for resuming from checkpoints
- Avoid duplicate processing

### 3. Concurrent Processing

- Support for high-concurrency API requests
- Configurable concurrency level
- Automatic retry mechanism

### 4. Batch Processing

- Support for batch processing of large datasets
- Configurable batch size
- Incremental result saving

## Requirements

```bash
pip install huggingface_hub openai aiofiles tqdm pandas openpyxl
```

## Output Formats

### JSONL Format

Each line contains a JSON object with:
- Original dataset metadata
- Classification results (confidence, tags, reasoning, etc.)
- Timestamp and model information

### Excel Format

A table containing all fields for easy manual review and filtering.

## Monitoring and Debugging

### Progress Monitoring

```bash
# Monitor processing progress
bash scripts/monitor_progress.sh
```

### Logging

All scripts output detailed log information, including:
- Processing progress
- Error messages
- Statistics

## FAQ

### Q: How to add a new domain?

A: Create a new JSON configuration file in the `config/domains/` directory, define tags, prompt templates, etc. Then optionally add the new domain to the `choices` in `domain.py`.

### Q: How to handle large amounts of data?

A: Use the checkpoint feature, which supports resuming after interruption. You can also adjust the `--batch_size` and `--max_concurrent` parameters.

### Q: What if API calls fail?

A: The scripts have built-in retry mechanisms that automatically retry failed requests. You can also adjust the `--max_retries` parameter.

## Contributing

Issues and Pull Requests are welcome!

## Contact

championzhong@sjtu.edu.cn
