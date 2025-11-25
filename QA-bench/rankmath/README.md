# RankMath Quick Start

Scripts you need:

1. `hard.py`: sparsity statistics for different difficulty levels.
2. `accuracy_vs_sparsity.py`: compare accuracy versus sparsity.
3. `math_equivalence.py`: extracts/compares math answers (auto-imported by the other scripts).

## How to Run

### 1. Sparsity vs Difficulty
```bash
python hard.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --gpu_id 0
```
Outputs `sparsity_vs_difficulty_model_comparison.pdf`.

### 2. Accuracy vs Sparsity
```bash
python accuracy_vs_sparsity.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --gpu_id 0
```
Outputs `accuracy_vs_sparsity.pdf`.

> The scripts process the entire MATH-500 dataset by default. Use `--num_samples` / `--max_samples_per_level` to reduce workload.
