# KGEAR Pipeline Configuration Guide

This guide explains how to configure and use the KGEAR pipeline with different LLM models and parameters.

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Supported LLM Models](#supported-llm-models)
3. [Parameter Descriptions](#parameter-descriptions)
4. [Example Configurations](#example-configurations)
5. [Running Tests with Different Configs](#running-tests-with-different-configs)
6. [Python API Usage](#python-api-usage)

---

## Configuration File Structure

The pipeline uses YAML configuration files. The default configuration is `config.yaml`:

```yaml
# LLM Model Configuration
llm:
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device_map: "auto"

# Bridge Entity Configuration
bridge_entities:
  max_count: 3

# GNN Configuration
gnn:
  top_k: 50

# Path Extension Configuration
path_extension:
  fetch_limit: 2000
  return_limit: 1000

# Dataset Configuration
dataset:
  cwq_path: "data/cwq_100.json"

# Logging Configuration
logging:
  level: "INFO"
```

---

## Supported LLM Models

### 1. Qwen2.5-32B-Instruct (Default)

**Model ID**: `Qwen/Qwen2.5-32B-Instruct`

- **Size**: 32B parameters
- **Memory**: ~64GB GPU memory required (with bfloat16)
- **Performance**: High accuracy, slower inference
- **Best for**: Production use, complex reasoning

### 2. Llama-3.1-8B-Instruct

**Model ID**: `meta-llama/Llama-3.1-8B-Instruct`

- **Size**: 8B parameters
- **Memory**: ~16GB GPU memory required (with bfloat16)
- **Performance**: Faster inference, slightly lower accuracy
- **Best for**: Testing, resource-constrained environments

### Adding Other Models

The pipeline supports any HuggingFace model compatible with `AutoModelForCausalLM`. To use a different model:

1. Update `config.yaml` with the model ID
2. Ensure the model supports the same prompt format
3. Test with a small number of questions first

---

## Parameter Descriptions

### LLM Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `model_name` | HuggingFace model ID | `Qwen/Qwen2.5-32B-Instruct` | Must be a valid model ID |
| `device_map` | Device mapping strategy | `auto` | Use `auto` for multi-GPU |

### Bridge Entity Parameters

| Parameter | Description | Default | Range | Impact |
|-----------|-------------|---------|-------|--------|
| `max_count` | Number of bridge entities to extract (N) | 3 | 1-10 | Higher N = more paths explored but more latency |

**N Parameter Guidelines**:
- **N=1**: Minimal exploration, fastest, may miss answers
- **N=3**: Balanced (default per task.md)
- **N=5**: More thorough, higher coverage, slower
- **N>5**: Diminishing returns, significant latency increase

### GNN Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `top_k` | Top-K triplets from GNN results | 50 | 10-200 |

**Top-K Guidelines**:
- **K=20**: Minimal context, faster LLM calls
- **K=50**: Balanced (default)
- **K=100**: More context, may include noise

### Path Extension Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `fetch_limit` | Triplets to fetch from SPARQL | 2000 | 500-5000 |
| `return_limit` | Triplets after PPR ranking | 1000 | 100-2000 |

**Extension Guidelines**:
- Higher `fetch_limit` = more comprehensive but slower SPARQL queries
- Higher `return_limit` = more context for final generation but slower LLM calls

---

## Example Configurations

### Config 1: Default (Qwen 32B, N=3)

**File**: `config.yaml`

```yaml
llm:
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device_map: "auto"
bridge_entities:
  max_count: 3
gnn:
  top_k: 50
path_extension:
  fetch_limit: 2000
  return_limit: 1000
```

**Use case**: Production, high accuracy required

### Config 2: Llama 8B for Fast Testing

**File**: `config_llama3.1_8b.yaml`

```yaml
llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  device_map: "auto"
bridge_entities:
  max_count: 3
gnn:
  top_k: 50
path_extension:
  fetch_limit: 2000
  return_limit: 1000
```

**Use case**: Rapid prototyping, testing on limited GPU memory

### Config 3: High Exploration (N=5)

**File**: `config_N5.yaml`

```yaml
llm:
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  device_map: "auto"
bridge_entities:
  max_count: 5  # Increased from 3 to 5
gnn:
  top_k: 50
path_extension:
  fetch_limit: 2000
  return_limit: 1000
```

**Use case**: Maximizing answer coverage for difficult questions

### Config 4: Resource-Constrained

**File**: `config_minimal.yaml`

```yaml
llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  device_map: "auto"
bridge_entities:
  max_count: 2
gnn:
  top_k: 30
path_extension:
  fetch_limit: 1000
  return_limit: 500
```

**Use case**: Limited GPU/time, prefer speed over accuracy

---

## Running Tests with Different Configs

### Method 1: Copy Config File

```bash
# Use default config (Qwen 32B, N=3)
python tests/test_3_complex_questions.py

# Switch to Llama 8B
cp config_llama3.1_8b.yaml config.yaml
python tests/test_3_complex_questions.py

# Switch to N=5 configuration
cp config_N5.yaml config.yaml
python tests/test_3_complex_questions.py
```

### Method 2: Create Custom Config

```bash
# Create your own config
nano config_custom.yaml
# Edit parameters as needed

# Copy to active config
cp config_custom.yaml config.yaml

# Run tests
python tests/test_3_complex_questions.py
```

### Output Files

Results are automatically saved with model name and N in the filename:

```
results/json/test_3_complex_results_qwen2.5_32b_instruct_N3.json
results/json/test_3_complex_results_llama_3.1_8b_instruct_N3.json
results/json/test_3_complex_results_qwen2.5_32b_instruct_N5.json
```

This makes it easy to compare results across different configurations.

---

## Python API Usage

### Load Config and Initialize Pipeline

```python
import yaml
from src.kgear_pipeline import KGEARPipeline

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = KGEARPipeline(
    model_name=config['llm']['model_name'],
    device_map=config['llm']['device_map'],
    top_k=config['gnn']['top_k'],
    max_bridge_entities=config['bridge_entities']['max_count'],
    extension_fetch_limit=config['path_extension']['fetch_limit'],
    extension_return_limit=config['path_extension']['return_limit']
)
```

### Direct Initialization (No Config File)

```python
from src.kgear_pipeline import KGEARPipeline

# Use Llama 8B with N=5
pipeline = KGEARPipeline(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    top_k=50,
    max_bridge_entities=5,
    extension_fetch_limit=2000,
    extension_return_limit=1000
)
```

### Process Questions

```python
import json

# Load dataset
with open('data/cwq_100.json', 'r') as f:
    data = json.load(f)

# Process a question
sample = data['Questions'][0]
result = pipeline.process_question(
    question_id=sample['ID'],
    question=sample['question'],
    ground_truth_answer=sample['answer']
)

# Access results
print(f"Final answer: {result['final_answer']}")
print(f"Exact match: {result['stages']['final_answer']['exact_match']}")
print(f"Bridge entities: {result['stages']['llm_pregeneration']['bridge_entities']}")
```

### Batch Processing with Different Configs

```python
import yaml
import json

configs = [
    ('config.yaml', 'Qwen 32B N=3'),
    ('config_llama3.1_8b.yaml', 'Llama 8B N=3'),
    ('config_N5.yaml', 'Qwen 32B N=5')
]

for config_path, desc in configs:
    print(f"\n{'='*80}")
    print(f"Testing with: {desc}")
    print(f"{'='*80}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize pipeline
    pipeline = KGEARPipeline(
        model_name=config['llm']['model_name'],
        max_bridge_entities=config['bridge_entities']['max_count'],
        # ... other params
    )

    # Process questions
    # ...
```

---

## Performance Comparison

Expected performance with different configurations (based on 3 test questions):

| Configuration | Avg Latency | GPU Memory | Exact Match | Use Case |
|---------------|-------------|------------|-------------|----------|
| Qwen 32B, N=3 | ~15-20s | ~64GB | 67% | Production |
| Llama 8B, N=3 | ~8-12s | ~16GB | ~60% | Testing |
| Qwen 32B, N=5 | ~25-30s | ~64GB | ~70% | Max coverage |
| Llama 8B, N=2 | ~5-8s | ~16GB | ~50% | Fast prototyping |

*Note: Latency includes LLM inference, SPARQL queries, and PPR ranking.*

---

## Troubleshooting

### Out of Memory Error

**Problem**: GPU OOM when loading model

**Solutions**:
1. Use smaller model: `meta-llama/Llama-3.1-8B-Instruct`
2. Reduce batch size (already 1 for generation)
3. Use CPU offloading: set `device_map: "balanced"`

### Slow Inference

**Problem**: Each question takes too long

**Solutions**:
1. Reduce `top_k` to 30
2. Reduce `max_bridge_entities` to 2
3. Reduce `extension_return_limit` to 500
4. Use faster model (Llama 8B)

### Low Accuracy

**Problem**: Too many wrong answers

**Solutions**:
1. Increase `max_bridge_entities` to 5
2. Increase `top_k` to 100
3. Use larger model (Qwen 32B)
4. Increase `extension_fetch_limit` to 3000

---

## Best Practices

1. **Start with default config**: Test with `config.yaml` first
2. **Compare results**: Run same questions with multiple configs
3. **Monitor GPU memory**: Use `nvidia-smi` to track usage
4. **Save all results**: Results are auto-saved with config info in filename
5. **Document experiments**: Keep notes on which config works best for your use case

---

**Last Updated**: October 2025
