# LongReason Sparsity Studies

This folder contains experiments that analyze how context length affects sparsity for reasoning models on the LongReason benchmark.

## Scripts

1. `analyze_length_sparsity.py`  
   - Loads the LongReason dataset at multiple context lengths (8k/16k/32k/64k).  
   - Runs a CausalLM to generate answers, records layer-wise sparsity metrics (L1 norm, top-k energy, Hoyer, effective rank).  
   - Produces accuracy tables and a PDF line chart (`layerwise_sparsity_<model>.pdf`).

   Example:
   ```bash
   python analyze_length_sparsity.py \
     --model_name Qwen/Qwen2.5-3B-Instruct \
     --gpu_id 0 \
     --max_samples 20 \
     --output_dir ./figures \
     --hf_token $HUGGING_FACE_HUB_TOKEN
   ```

2. `token_level_context_comparison.py`  
   - Compares sparsity/token statistics at different truncation lengths for selected prompts.  
   - Useful for drilling down into specific samples.

## Requirements

```bash
conda create -n longreason python=3.10 -y
conda activate longreason
pip install -r requirements.txt   # torch, transformers, datasets, matplotlib, numpy
export HUGGING_FACE_HUB_TOKEN="hf_xxx"
```

## Outputs

- `layerwise_sparsity_<model>.pdf`: layer-wise sparsity trends for each context length.  
- Console logs: accuracy per split, per-sample sparsity info.  
- Additional figures (e.g., `last_hidden_state_comparison_*`) depending on the analysis script.

## Tips

- Reduce `--max_samples` if GPU memory/time is limited.  
- Use `--hf_token` to avoid rate limits when downloading private weights.  
- Random seed is fixed to 42; change inside the script if you need different runs.

