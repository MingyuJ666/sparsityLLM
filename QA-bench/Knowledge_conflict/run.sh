#!/bin/bash


echo "=================================================="
echo "Knowledge Conflict Sparsity Analysis"
echo "=================================================="
echo ""


GPU_ID=0
MAX_SAMPLES=100
OUTPUT_DIR="./"



MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
)


for MODEL in "${MODELS[@]}"; do
    
    CMD="python judge.py \
        --model_name $MODEL \
        --gpu_id $GPU_ID \
        --max_samples $MAX_SAMPLES \
        --output_dir $OUTPUT_DIR"
    
    if [ ! -z "$HF_TOKEN" ]; then
        CMD="$CMD --hf_token $HF_TOKEN"
    fi
    
    echo "Command: $CMD"
    eval $CMD
    
done

echo "=================================================="
echo "All analyses completed!"
echo "Check the output directory for PDF results."
echo "=================================================="

