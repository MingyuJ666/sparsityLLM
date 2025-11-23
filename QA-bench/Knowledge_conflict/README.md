# Knowledge Conflict Analysis

This experiment analyzes the sparsity of model hidden states under **knowledge-conflicting** vs **non-conflicting** scenarios.

## Overview

The code compares how language models represent information when:
- **Conflicting**: The model's parametric knowledge contradicts the provided context
- **Non-conflicting**: The context aligns with the model's knowledge

## Files

- `judge.py`: Main analysis script
- `data.json`: Dataset with conflicting/non-conflicting question pairs
- `run.sh`: Convenience script to run experiments
- `*.pdf`: Generated visualization results

## Sparsity Metrics

The analysis computes and visualizes:
1. **L1 Norm**: Sum of absolute activation values
2. **Top-5% Energy**: Energy concentration in top 5% neurons
3. **Top-10% Energy**: Energy concentration in top 10% neurons
4. **Effective Rank**: Measure of representation diversity
5. **Hoyer Sparsity**: Normalized sparsity measure

## Usage

### Quick Start
```bash
bash run.sh
```

### Custom Run
```bash
python judge.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --max_samples 100 \
    --output_dir ./ \
    --hf_token YOUR_TOKEN  # optional
```

### Arguments
- `--model_name`: HuggingFace model name (default: `Qwen/Qwen2.5-3B-Instruct`)
- `--gpu_id`: GPU device ID (default: `0`)
- `--data_path`: Path to data.json (default: `./data.json`)
- `--output_dir`: Output directory for PDF (default: `./`)
- `--max_samples`: Number of samples to analyze (default: `100`)
- `--hf_token`: HuggingFace token (optional)

## Output

The script generates a PDF with 6 subplots:
- **(a-e)**: Individual sparsity metrics comparison
- **(f)**: Summary comparison across all metrics

Each plot shows:
- 🔴 **Red bars**: Conflicting scenarios
- 🔵 **Blue bars**: Non-conflicting scenarios



