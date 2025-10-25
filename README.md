# KGEAR: Knowledge Graph Enhanced Answer Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**KGEAR** is a knowledge graph-based question answering system that combines GNN-based triplet retrieval with LLM reasoning and adaptive path extension for complex multi-hop questions.

## Overview

KGEAR addresses the challenge of answering complex questions over large knowledge graphs (Freebase) by:

1. **GNN-based Retrieval**: Pre-trained GNN model (SubgraphRAG) retrieves top-K relevant triplets
2. **LLM Pregeneration**: LLM judges if evidence is sufficient or identifies bridge entities for extension
3. **Adaptive Path Extension**: Two-stage filtering (PPR + GNN) expands from bridge entities when needed
4. **Final Answer Generation**: LLM generates answer from enriched evidence

### Key Features

- ✅ **Two-stage filtering**: PPR coarse filter (2000/entity) → GNN fine filter (50 globally)
- ✅ **Real GNN scoring**: Uses trained SubgraphRAG model, not heuristics
- ✅ **Adaptive extension**: Only extends when LLM judges initial evidence insufficient
- ✅ **Comprehensive metrics**: Tracks GT coverage, path coverage, token usage, and accuracy at each stage
- ✅ **vLLM integration**: Fast LLM inference with Qwen2.5-32B-Instruct

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KGEAR Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load Pre-computed GNN Results (50 triplets)            │
│     ├── SubgraphRAG trained GNN model                      │
│     └── Top-K selection from 2-hop neighborhood            │
│                                                             │
│  2. LLM Pregeneration (vLLM)                                │
│     ├── Judge: Can answer with current evidence?           │
│     ├── If YES → Direct answer (29% of questions)          │
│     └── If NO → Extract bridge entities (71% of questions) │
│                                                             │
│  3. Path Extension (if needed)                              │
│     ├── Stage 1: PPR pre-filter (2000 per bridge entity)   │
│     ├── Stage 2: GNN scoring (top 50 globally)             │
│     └── Merge with initial 50 → 100 total triplets         │
│                                                             │
│  4. Final Answer Generation                                 │
│     └── LLM generates answer from enriched evidence         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Results (CWQ-100)

### Current Performance (71/100 questions processed)

| Metric | Value |
|--------|-------|
| **Exact Match Accuracy** | 32.0% (31/97) |
| **Direct Answer Success** | 29 questions (62.1% accuracy) |
| **Extended Questions** | 68 questions (19.1% accuracy) |
| **GT Hit Rate (before extension)** | 60.6% (43/71) |
| **GT Hit Rate (after extension)** | 69.0% (49/71) |
| **Avg Path Coverage (before)** | 46.8% |
| **Avg Path Coverage (after)** | 47.6% |
| **Avg LLM Calls per Question** | 1.70 |
| **Avg Tokens per LLM Call** | 1,565 tokens |
| **Avg Processing Time** | 4.15s per question |

### Path Extension Impact

- **+6 questions** gained full GT coverage through path extension (+8.5 percentage points)
- **+1.5% path coverage** for extended questions
- Path extension helps **14.3%** of initially insufficient questions achieve GT hit

## Installation

### Prerequisites

```bash
# Python 3.8+
# PyTorch 2.0+
# CUDA 11.8+ (for GPU support)
```

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/KGEAR.git
cd KGEAR
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download pre-computed GNN results**:
```bash
# Place SubgraphRAG GNN results at:
# /path/to/SubgraphRAG/retrieve/cwq_Jun15-05:26:49/retrieval_result.pth
```

4. **Set up Freebase SPARQL endpoint**:
```bash
# Ensure Freebase endpoint is accessible at:
# http://localhost:3001/sparql
```

5. **Start vLLM server**:
```bash
# Start vLLM server with Qwen2.5-32B-Instruct
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
```

## Usage

### Quick Test (10 questions)

```bash
cd tests/
python test_pathExtension_gnn_10q.py
```

### Full CWQ-100 Evaluation

```bash
cd tests/
python test_pathExtension_gnn_cwq100.py
```

Results will be saved to `results/task2_gnn/`.

### Configuration

Edit pipeline parameters in `src/pipeline_pathExtension_gnn.py`:

```python
pipeline = KGEARPipelineWithGNN(
    top_k=50,                      # Initial GNN triplets
    max_bridge_entities=3,         # Max entities to extend
    ppr_limit_per_entity=2000,     # PPR coarse filter
    gnn_return_limit=50,           # GNN fine filter
    gnn_device='cuda:0'            # GNN model device
)
```

## Analysis Tools

### Ground Truth Hit Analysis

Compare GT hit rates before and after path extension:

```bash
cd analysis/
python analyze_gt_hit_before_after.py
```

### Top-100 vs Top-50 Coverage

Analyze whether increasing to 100 initial GNN triplets improves coverage:

```bash
cd analysis/
python analyze_top100_coverage.py
```

### Aggregate Metrics

Calculate aggregate statistics across all questions:

```bash
cd analysis/
python calculate_metrics.py
```

## Project Structure

```
KGEAR/
├── src/                                    # Core pipeline components
│   ├── pipeline_pathExtension_gnn.py      # Main pipeline
│   ├── gnn_result_loader.py               # GNN results loader
│   ├── gnn_scorer_real.py                 # Real GNN scorer
│   ├── bridge_extension_2hop_gnn.py       # PPR-based path extension
│   ├── llm_bridge_extractor_vllm.py       # vLLM interface
│   ├── prompt_templates.py                # Prompt templates
│   ├── metrics_evaluator.py               # Evaluation metrics
│   └── answer_validator.py                # Response validation
│
├── tests/                                  # Test scripts
│   ├── test_pathExtension_gnn_cwq100.py   # CWQ-100 evaluation
│   └── test_pathExtension_gnn_10q.py      # Quick 10-question test
│
├── analysis/                               # Analysis tools
│   ├── analyze_gt_hit_before_after.py     # Before/after comparison
│   ├── analyze_top100_coverage.py         # Coverage analysis
│   └── calculate_metrics.py               # Aggregate metrics
│
├── data/                                   # Datasets
│   └── cwq_100.json                       # CWQ-100 test set
│
├── docs/                                   # Documentation
│   ├── task.md                            # Original requirements
│   ├── TASK2_IMPLEMENTATION_SUMMARY.md    # Implementation details
│   └── CONFIGURATION_GUIDE.md             # Setup guide
│
└── results/                                # Results directory
    └── .gitkeep
```

## Metrics

KGEAR tracks 4 key metrics at each pipeline stage:

### 1. After GNN Pruning
- **Ground Truth Hit**: Whether GT answer appears in top-K triplets
- **Path Coverage**: Percentage of reasoning path triplets found

### 2. After LLM Pregeneration
- **Input/Output Tokens**: Token usage
- **Sufficiency Judgment**: Can answer directly or needs extension?
- **Bridge Entities**: Identified entities for extension

### 3. After Path Extension
- **Triplet Count**: Total triplets after extension
- **Ground Truth Hit**: Whether GT now appears
- **Path Coverage**: Updated coverage after extension

### 4. Final Answer
- **Exact Match**: Does predicted answer match ground truth?
- **Partial Match**: Partial overlap with ground truth

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- vLLM 0.2.0+
- SPARQLWrapper
- NetworkX
- NumPy

See `requirements.txt` for complete list.

## Citation

If you use KGEAR in your research, please cite:

```bibtex
@software{kgear2024,
  title={KGEAR: Knowledge Graph Enhanced Answer Retrieval},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/KGEAR}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **SubgraphRAG**: GNN model architecture and training
- **Qwen Team**: Qwen2.5-32B-Instruct LLM
- **vLLM**: Fast LLM inference framework
- **ComplexWebQuestions (CWQ)**: Benchmark dataset

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
