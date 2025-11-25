# RankMath Experiments

This folder contains the scripts we use to evaluate sparsity and accuracy trends on the MATH-500 benchmark. Two entry points are provided:

- `hard.py` — measures sparsity statistics across difficulty levels.
- `accuracy_vs_sparsity.py` — correlates model accuracy with sparsity metrics.

Both rely on `math_equivalence.py` to extract and compare numeric answers from the model outputs.

## 1. Environment

```bash
conda create -n rankmath python=3.10 -y
conda activate rankmath
pip install -r requirements.txt   # transformers, datasets, matplotlib, seaborn, scipy, torch, vllm
```

You also need valid Hugging Face credentials because the scripts download Qwen/LLaMA checkpoints. Set the token once:

```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxx"
```

## 2. Sparsity vs Difficulty (`hard.py`)

This script computes sparsity metrics (L1 norm, top-k energy ratios) for each difficulty level in MATH-500.

```bash
cd /common/home/mg1998/deepsieve/entrainment/QA-bench/rankmath
python hard.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --gpu_id 0 \
  --num_samples 50        # optional: samples per level
```

Key outputs:

- Console summary of sparsity differences.
- PDF `sparsity_vs_difficulty_model_comparison.pdf` showing ICML-style box plots.
- JSON/CSV results (if you enable saving inside the script).

## 3. Accuracy vs Sparsity (`accuracy_vs_sparsity.py`)

This script first runs vLLM to generate answers for all samples, then compares accuracy trends with sparsity statistics.

```bash
python accuracy_vs_sparsity.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --gpu_id 0 \
  --max_samples_per_level 80
```

Outputs:

- Console table listing accuracy and sparsity for each level.
- Plot `accuracy_vs_sparsity.pdf` containing two subfigures (accuracy vs L1, accuracy vs top-10% energy).

> **Tip:** Both scripts can be long-running. Use `nohup ... &` or `tmux` when running on remote servers.

## 4. Answer Matching (`math_equivalence.py`)

`math_equivalence.py` cleans LaTeX answers, extracts boxed values, and performs a strict equality check. It is imported automatically; no separate run is needed.

## 5. Customization

- Switch to different models by changing `--model_name`.
- Adjust prompt templates, sampling parameters, or sparsity metrics directly inside the Python files.
- To add new metrics, modify `compute_sparsity_metrics` in the corresponding script and update the plotting logic.

## 6. Troubleshooting

| Symptom                                      | Fix                                                                 |
|----------------------------------------------|---------------------------------------------------------------------|
| `ValueError: CUDA out of memory`             | Reduce `--num_samples`, use a smaller model, or set `gpu_id` to a free GPU. |
| vLLM cannot find weights                     | Ensure the HF token has permission and `huggingface-cli login` succeeded. |
| Answer comparison always fails               | Check `math_equivalence.py` for parsing; enable verbose mode in `is_equiv`. |
| Plots look empty                             | Confirm the script actually processed data (e.g., no NaN warnings). |

## 7. Reproducing Figures

1. Run `hard.py` to regenerate sparsity-vs-difficulty plots (saved to PDF).
2. Run `accuracy_vs_sparsity.py` to regenerate accuracy-vs-sparsity plots.
3. Use the resulting PDFs in the paper/slide decks.

Feel free to open issues or add scripts here if you need additional analyses.

