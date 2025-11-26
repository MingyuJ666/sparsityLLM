# Robustness Benchmark (MMLU-Pro)

This folder contains two entry points:

1. `analyze_mmlu_pro_area_difficulty.py`: evaluate sparsity/accuracy across academic areas and adversarial noise levels on MMLU-Pro.
2. `build_mmlu_robust_dataset.py`: export the noise-augmented prompts as a JSON dataset (full areas/samples by default).

## Prerequisites

```bash
conda create -n robustness python=3.10 -y
conda activate robustness
pip install -r requirements.txt   # torch, transformers, datasets, matplotlib, scipy, numpy
export HUGGING_FACE_HUB_TOKEN="hf_xxx"    # required to download Qwen/Llama checkpoints
```

## Quick Start

```bash
python analyze_mmlu_pro_area_difficulty.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --gpu_id 0 \
  --num_samples 30 \
  --hf_token $HUGGING_FACE_HUB_TOKEN
```

Key flags:

- `--use_cot`: enable chain-of-thought prompting (default off).
- `--num_samples`: limit samples per academic area (default 20 in analyzer, “all” in dataset builder).
- `--gpu_id`: choose the CUDA device.

## Outputs

- Analyzer: console summary per noise level + `combined_heatmaps_<model>.pdf` (and `cot_...`).
- Dataset builder: `mmlu_robust_dataset.json`, containing normal/light/heavy prompts and expanded options.

## Tips

- If you run out of memory, reduce `--num_samples` or switch to a smaller model.
- When accuracy statistics look wrong, ensure `math_equivalence.py` in `../rankmath` is up to date for answer checking.
- The analyzer prints debug info for the first area—comment out the logs if too verbose.

