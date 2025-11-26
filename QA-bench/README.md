# QA-Bench Overview

This workspace collects all QA-related analysis/visualization scripts. Each subfolder focuses on a different robustness axis or dataset.

## Folders

| Folder | Description | Key Scripts |
|--------|-------------|-------------|
| `Longreason/` | Studies how long-context reasoning affects sparsity/accuracy using the LongReason benchmark. | `analyze_length_sparsity.py`, `token_level_context_comparison.py` |
| `robustness/` | Analyzes MMLU-Pro robustness under adversarial noise and exports the noisy prompts as a JSON dataset. | `analyze_mmlu_pro_area_difficulty.py`, `build_mmlu_robust_dataset.py` |
| `rankmath/` | Evaluates math QA accuracy/sparsity on MATH-500 (hard splits, accuracy vs. sparsity). | `hard.py`, `accuracy_vs_sparsity.py`, `math_equivalence.py` |
| `Knowledge_conflict/` | Compares sparsity between conflicting vs. non-conflicting knowledge segments. | `judge.py` |

Refer to each folder’s README for setup and execution details.

## Common Setup

```bash
conda create -n qa-bench python=3.10 -y
conda activate qa-bench
pip install -r requirements.txt   # torch, transformers, datasets, matplotlib, numpy, scipy, etc.
export HUGGING_FACE_HUB_TOKEN="hf_xxx"
```

> **Tip:** Many scripts assume GPU availability and may download large checkpoints (Qwen/Llama). Set `--gpu_id` and `--hf_token` where applicable.

