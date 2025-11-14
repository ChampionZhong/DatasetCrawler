# Domain Dataset Filtering

统一的域特定数据集过滤工具，支持通过配置文件定义不同的分类域。

## 功能

- **agent**: 粗粒度 LLM agent 数据集过滤
- **agent_specific**: 细粒度高质量 LLM agent 数据集过滤（需要先运行 agent）
- **finance**: 金融领域数据集过滤

## 使用方法

### Python 脚本

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
```

### Shell 脚本

```bash
# Agent 粗粒度过滤
sbatch scripts/domain/domain.sh agent

# Agent 细粒度过滤
sbatch scripts/domain/domain.sh agent_specific

# Finance 过滤
sbatch scripts/domain/domain.sh finance
```

## 配置

### API 配置

API 参数（base_url, api_key, model, temperature）从 `config/api.json` 读取。

### Domain 配置

每个 domain 的配置存储在 `config/domains/{domain}.json` 中，包括：
- 标签/类别定义
- System prompt
- User prompt 模板
- 响应格式定义
- 过滤条件

## 迁移指南

### 从旧脚本迁移

**旧方式：**
```bash
python domain/agent.py -i input.jsonl -o output.jsonl
python domain/agent_specific.py -i input.jsonl -o output.jsonl
python domain/finance.py -i input.jsonl -o output.jsonl
```

**新方式：**
```bash
python domain/domain.py --domain agent -i input.jsonl -o output.jsonl
python domain/domain.py --domain agent_specific -i input.jsonl -o output.jsonl
python domain/domain.py --domain finance -i input.jsonl -o output.jsonl
```

所有原有功能都已保留，包括：
- Checkpoint 支持
- 批量处理
- 并发控制
- 重试机制
- 增量保存

## 添加新 Domain

1. 在 `config/domains/` 目录下创建新的 JSON 配置文件
2. 定义标签、prompt 模板、响应格式等
3. 在 `domain.py` 的 `choices` 中添加新 domain（可选，用于验证）

