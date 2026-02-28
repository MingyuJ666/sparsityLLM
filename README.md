# Entrainment: Hidden State Sparsity Analysis for LLMs

This repository investigates the relationship between **hidden state sparsity** and model behavior in Large Language Models (LLMs). We study how sparsity patterns correlate with task difficulty, reasoning depth, and out-of-distribution (OOD) generalization.

## Overview

| Module | Purpose |
|--------|---------|
| `pretrain/` | Synthetic knowledge graph pretraining to study ID vs OOD sparsity |
| `cot/` | Chain-of-thought prompting with sparsity-based curriculum learning |
| `QA-bench/` | QA benchmarks analyzing sparsity across various domians |


## Key Findings

- **Sparsity as Uncertainty Signal**: Hidden state sparsity (L1 norm, effective rank, top-k energy) correlates with task difficulty
- **Curriculum Learning**: Ordering examples by sparsity improves few-shot learning performance

## Project Structure

```
entrainment/
├── pretrain/                    # Synthetic KG pretraining
│   ├── pretrain.py              # Main training script
│   └── llama-*/                 # Saved model checkpoints
│
├── cot/                         # Chain-of-Thought experiments
│   ├── cot.py                   # Main CoT inference script
│   ├── math_equivalence.py      # Answer matching utilities
│   ├── dataset/                 # Math-500 dataset
│   ├── math_utils/              # Math parsing utilities
│   └── utils/
│       ├── rank.py              # Sparsity-based ranking
│       └── retrieve_similar_examples.py
│
├── QA-bench/                    # QA benchmark analysis
│   ├── Longreason/              # Long-context reasoning
│   ├── robustness/              # MMLU-Pro adversarial noise
│   ├── rankmath/                # MATH-500 difficulty analysis
│   └── Knowledge_conflict/      # Knowledge conflict detection
│
├── SFT/                         # Supervised fine-tuning
│   └── toy_example_ce.py        # Multi-class CE toy verification
│
├── environment.yml              # Conda environment
├── requirements.txt             # Pip dependencies
└── draw.ipynb                   # Visualization notebook
```

## Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate sparsity
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```


## Quick Start

### 1. Pretrain: Synthetic Knowledge Graph Experiments

Train a small LLaMA model on synthetic knowledge graphs with latent rules:

```bash
cd pretrain
python pretrain.py \
    --llm_size llama-32-32 \
    --gpu_id 0 \
    --steps 1000 \
    --seed 42
```

This generates:
- Synthetic knowledge graph with deductive rules
- ID test set (training memory)
- OOD test sets (requires multi-hop reasoning)
- Sparsity comparison across difficulty levels

### 2. CoT: Curriculum Learning on MATH-500

Run chain-of-thought inference with sparsity-based curriculum:

```bash
cd cot
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot curriculum \
    --rank_metric l0_norm \
    --n_levels 5
```

Prompting strategies:
- `cot`: Zero-shot chain-of-thought
- `few-shot`: Random few-shot examples
- `curriculum`: Sparsity-ranked examples (easy→hard)
- `auto-shot`: Semantic similarity retrieval

### 3. QA-Bench: Robustness Analysis

```bash
cd QA-bench

# Long-context reasoning sparsity
python Longreason/analyze_length_sparsity.py

# MATH-500 accuracy vs sparsity
python rankmath/accuracy_vs_sparsity.py

# MMLU-Pro adversarial robustness
python robustness/analyze_mmlu_pro_area_difficulty.py
```

## Sparsity Metrics

We measure hidden state sparsity using multiple metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **L1 Norm** | $\sum_i \|h_i\|$ | Total activation magnitude |
| **L0 Norm** | $\sum_i \mathbb{1}[h_i \neq 0]$ | Number of active dimensions |
| **Top-k% Energy** | $\frac{\sum_{i \in \text{top-k}} h_i^2}{\sum_i h_i^2}$ | Energy concentration |
| **Effective Rank** | $\exp(-\sum_i p_i \log p_i) / d$ | Dimensionality utilization |



## Contact

For questions or issues, please open a GitHub issue.
