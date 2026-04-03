# RankMath Quick Start

Scripts you need:

1. `hard.py`: sparsity statistics for different difficulty levels.
2. `accuracy_vs_sparsity.py`: compare accuracy versus sparsity.
3. `math_equivalence.py`: extracts/compares math answers (auto-imported by the other scripts).
4. `deepmath_sparsity.py`: analyze sparsity vs difficulty on the DeepMath-103K dataset.

## How to Run

### 1. Sparsity vs Difficulty in MATH500
```bash
python hard.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --gpu_id 0
```
Outputs `sparsity_vs_difficulty_model_comparison.pdf`.

### 2. Accuracy vs Sparsity in MATH500
```bash
python accuracy_vs_sparsity.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --gpu_id 0
```
Outputs `accuracy_vs_sparsity.pdf`.

> The scripts process the entire MATH-500 dataset by default. Use `--num_samples` / `--max_samples_per_level` to reduce workload.

### 3. DeepMath Sparsity Analysis
```bash
python deepmath_sparsity.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --gpu_id 0 \
  --num_samples 10000
```
Outputs `deepmath_sparsity_<model>.pdf`.

This script loads the [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) dataset and tests the hypothesis that harder math problems produce sparser hidden-state representations. It computes four sparsity metrics (L1 norm, top-5%/top-10% energy ratio, effective rank) on the last-token hidden state and reports Pearson/Spearman correlations against the continuous difficulty score.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model to analyze |
| `--gpu_id` | `3` | GPU device ID |
| `--num_samples` | `10000` | Number of samples (`None` = use all 103K) |
| `--layer` | `-1` | Hidden layer index (`-1` = last layer) |
| `--output_dir` | `./` | Directory for output PDF |

> Dataset parquets are cached locally in `.deepmath_cache/` (~2 GB) to avoid re-downloading.
