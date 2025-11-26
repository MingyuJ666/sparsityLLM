# QA-Bench Overview

This workspace collects all QA-related analysis/visualization scripts. Each subfolder focuses on a different robustness axis or dataset.

## Folders

| Folder | Description | Key Scripts |
|--------|-------------|-------------|
| `Longreason/` | Studies how long-context reasoning affects sparsity/accuracy using the LongReason benchmark. | `analyze_length_sparsity.py` |
| `robustness/` | Analyzes MMLU-Pro robustness under adversarial noise and exports the noisy prompts MMLU-Robust dataset. | `analyze_mmlu_pro_area_difficulty.py`, `build_mmlu_robust_dataset.py` |
| `rankmath/` | Evaluates math QA accuracy/sparsity on MATH-500 (hard splits, accuracy vs. sparsity). | `hard.py`, `accuracy_vs_sparsity.py` |
| `Knowledge_conflict/` | Compares sparsity between conflicting vs. non-conflicting knowledge segments. | `judge.py` |

Refer to each folder’s README for setup and execution details.


> **Tip:** Many scripts assume GPU availability and may download large checkpoints (Qwen/Llama). Set `--gpu_id` and `--hf_token` where applicable.

