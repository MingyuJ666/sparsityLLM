# MATH-500 Problem Solving with vLLM

This script uses vLLM to solve MATH-500 problems with various prompting strategies, including Chain-of-Thought (CoT), few-shot learning, curriculum learning, and auto-shot retrieval.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)


## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate dpo

# Or install dependencies manually
conda install python=3.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers datasets huggingface-hub tqdm numpy
```

### Option 2: Using pip

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install core dependencies
pip install torch transformers vllm datasets huggingface-hub tqdm numpy
```

### HuggingFace Authentication

The script requires a HuggingFace token to access models. You can either:

1. **Set environment variable** (recommended):
   ```bash
   export HUGGING_FACE_HUB_TOKEN="your_token_here"
   ```

2. **Login via CLI**:
   ```bash
   huggingface-cli login
   ```

3. **Modify the script**: Update line 20 in `cot.py` with your token (not recommended for security).

## Dataset Preparation

The script expects a local Math-500 dataset in the following structure:

```
cot/
├── dataset/
│   └── Math-500/
│       ├── train.jsonl
│       └── test.jsonl
```

Each JSONL file should contain one JSON object per line with the following format:

```json
{
  "problem": "The problem statement...",
  "solution": "The step-by-step solution...",
  "answer": "The final answer...",
  "level": "Level 1"
}
```

**Note**: The script will automatically use the local dataset if `--use_local_dataset` is set (default: True).

## Usage

### Basic Command

```bash
cd cot
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot curriculum
```

### Running in Background (with screen)

For long-running experiments, use `screen`:

```bash
# Start a new screen session
screen -S math500_experiment

# Run your command
python cot.py --model_path Qwen/Qwen2.5-7B-Instruct --gpu_id 0 --use_cot curriculum

# Detach: Press Ctrl+A, then D
# Reattach: screen -r math500_experiment
```

## Arguments

### Required Arguments
- None (all have defaults)

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | `Qwen/Qwen2.5-7B-Instruct` | Model path or HuggingFace model name |
| `--gpu_id` | int | `2` | GPU ID to use |
| `--use_cot` | str | `curriculum` | Prompting strategy: `cot`, `few-shot`, `curriculum`, `auto-shot` |
| `--gpu_memory_utilization` | float | `0.5` | GPU memory utilization (0.0-1.0) |
| `--max_tokens` | int | `4096` | Maximum number of tokens to generate |
| `--num_few_shot` | int | `2` | Number of few-shot examples |
| `--example_pool_size` | int | `500` | Size of example pool (from training set) |
| `--rank_metric` | str | `l0_norm` | Sparsity metric for curriculum: `l0_norm`, `top10pct_ratio`, `effective_rank` |
| `--n_levels` | int | `5` | Number of difficulty levels (for curriculum learning) |
| `--local_dataset_path` | str | `./dataset/Math-500` | Path to local Math-500 dataset |
| `--sample_test_size` | int | `None` | Number of samples to randomly sample from test set |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--verbose` | flag | `False` | Print detailed response for each sample |
| `--print_freq` | int | `100` | Print detailed information every N samples |

## Examples

### Example 1: Basic CoT Prompting

```bash
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot cot \
    --gpu_memory_utilization 0.5
```

### Example 2: Few-Shot Learning

```bash
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot few-shot \
    --num_few_shot 3 \
    --example_pool_size 1000
```

### Example 3: Curriculum Learning (Default)

```bash
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot curriculum \
    --rank_metric l0_norm \
    --n_levels 5 \
    --example_pool_size 500
```

### Example 4: Auto-Shot Retrieval

```bash
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot auto-shot \
    --num_few_shot 2 \
    --example_pool_size 500
```

### Example 5: Quick Test with Small Sample

```bash
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot cot \
    --sample_test_size 10 \
    --verbose
```

### Example 6: Using Different Sparsity Metrics

```bash
# Using top10pct_ratio metric
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot curriculum \
    --rank_metric top10pct_ratio \
    --n_levels 5

# Using effective_rank metric
python cot.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --gpu_id 0 \
    --use_cot curriculum \
    --rank_metric effective_rank \
    --n_levels 5
```

### Example 7: Using Local Model Path

```bash
python cot.py \
    --model_path /path/to/local/model \
    --gpu_id 0 \
    --use_cot curriculum
```

## Prompting Strategies

### 1. `cot` (Chain-of-Thought)
- Zero-shot CoT prompting
- Asks the model to solve step-by-step
- No examples provided

### 2. `few-shot`
- Randomly selects N examples from the example pool
- Provides examples before the actual problem
- Uses `--num_few_shot` to control number of examples

### 3. `curriculum` (Default)
- Sorts examples by sparsity metrics (difficulty)
- Divides examples into difficulty levels
- Selects examples based on problem difficulty:
  - **Easy problems**: 1 same-level + 1 higher-level example
  - **Medium problems**: 2 same-level examples
  - **Hard problems**: 1 lower-level + 1 same-level example
- Uses `--rank_metric` to determine sorting metric
- Uses `--n_levels` to control number of difficulty levels

### 4. `auto-shot`
- Retrieves top-K most similar examples using semantic similarity
- Uses `--num_few_shot` to control number of retrieved examples

## Output

The script will:
1. Load and prepare the dataset
2. (If curriculum) Rank examples by sparsity and divide into difficulty levels
3. Load the model using vLLM
4. Generate responses for all test samples
5. Evaluate using `is_equiv` from `math_equivalence`
6. Print final accuracy statistics

Example output:
```
🚀 CUDA_VISIBLE_DEVICES set to 0
Loading local MATH-500 dataset (./dataset/Math-500)...
Test set size: 500

Loading vLLM model...
Model loading completed

Preparing prompts...
Preparation completed, total 500 samples

Starting batch inference...
Inference completed

Processing results
================================================================================
[Sample 1/500]
================================================================================
Problem: ...
Model response: ...
Predicted answer: ...
Ground truth: ...
Result: ✓ Correct
Current accuracy: 100.00% (1/1)
================================================================================

...

Evaluation completed!
================================================================================
Total samples: 500
Correct: 350
Final accuracy: 70.00%
================================================================================
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce GPU memory utilization
```bash
python cot.py --gpu_memory_utilization 0.3
```

### Issue: Dataset Not Found

**Error**: `FileNotFoundError: Local dataset file does not exist`

**Solution**: 
1. Check that `./dataset/Math-500/train.jsonl` and `./dataset/Math-500/test.jsonl` exist
2. Or specify custom path: `--local_dataset_path /path/to/dataset`

### Issue: Model Download Fails

**Error**: Authentication or network issues

**Solution**:
1. Set `HUGGING_FACE_HUB_TOKEN` environment variable
2. Or run `huggingface-cli login`
3. Check network connection

### Issue: CUDA Device Not Found

**Error**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**:
1. Check CUDA version compatibility
2. Ensure GPU is available: `nvidia-smi`
3. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Curriculum Learning Takes Too Long

**Solution**: 
1. Reduce `--example_pool_size` (e.g., from 500 to 100)
2. Use a smaller model for sparsity computation
3. Reduce `--sample_test_size` for quick testing

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'utils.rank'`

**Solution**: Ensure you're running from the `cot/` directory:
```bash
cd cot
python cot.py ...
```

## Code Structure

```
cot/
├── cot.py                          # Main script
├── math_equivalence.py             # Answer extraction and comparison
├── math_utils/
│   ├── parse.py                    # Math string parsing utilities
│   └── compare.py                  # Math equivalence comparison
└── utils/
    ├── rank.py                     # Sparsity-based ranking for curriculum learning
    └── retrieve_similar_examples.py # Semantic similarity retrieval
```

## Citation

If you use this code, please cite the relevant papers for:
- vLLM: [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Curriculum Learning: Based on sparsity metrics from hidden states
- MATH-500: The mathematical reasoning dataset

## License

[Specify your license here]

## Contact

[Your contact information]

