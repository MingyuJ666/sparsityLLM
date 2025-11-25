# Robustness Benchmark (MMLU-Pro)

This folder contains one entry point: `analyze_mmlu_pro_area_difficulty.py`.  
It evaluates how sparsity metrics change across academic areas and adversarial noise levels on the MMLU-Pro dataset.

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

Important flags:

- `--use_cot`: enable chain-of-thought prompting (default off).
- `--num_samples`: limit samples per academic area.
- `--gpu_id`: choose the CUDA device.

## Outputs

- Console table summarizing sparsity metrics per noise level.
- Combined heatmaps saved as `combined_heatmaps_<model>.pdf` (and `cot_...` if CoT is enabled).
- Optional JSON statistics (enable inside the script if needed).

## Tips

- If you run out of memory, reduce `--num_samples` or switch to a smaller model.
- When accuracy statistics look wrong, ensure `math_equivalence.py` in `../rankmath` is up to date for answer checking.
- The script prints debug info for the first area—safe to ignore or comment out if too verbose.

