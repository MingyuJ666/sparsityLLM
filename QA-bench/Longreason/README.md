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



