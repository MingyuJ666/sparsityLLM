# Robustness Benchmark (MMLU-Pro)

This folder contains two entry points:

1. `analyze_mmlu_pro_area_difficulty.py`: evaluate sparsity/accuracy across academic areas and adversarial noise levels on MMLU-Pro.
2. `build_mmlu_robust_dataset.py`: create the noise-augmented MMLU-Robust as a JSON dataset (full areas/samples by default).


```

## Quick Start

```bash
python analyze_mmlu_pro_area_difficulty.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --gpu_id 0 \
  --num_samples 30 \
  --hf_token $HUGGING_FACE_HUB_TOKEN
```
```bash
python build_mmlu_robust_dataset.py
```
Key flags:

- `--use_cot`: enable chain-of-thought prompting (default off).
- `--num_samples`: limit samples per academic area (default 20 in analyzer, “all” in dataset builder).
- `--gpu_id`: choose the CUDA device.



