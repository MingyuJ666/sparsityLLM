"""
Tool module for ranking datasets by sparsity,
mainly used for Curriculum Learning: organizing training data in order from "easy examples to hard examples".
"""

import torch
import numpy as np
from tqdm import tqdm

def compute_sparsity_metrics(hidden_state):
    """Compute a set of sparsity metrics for a single hidden state"""
    h = hidden_state.squeeze()
    hidden_dim = h.shape[0]
    
    abs_h = h.abs()
    total_energy = abs_h.sum().item()
    
    # Top-K percentage (energy ratio of top 1%, 5%, 10%)
    top1pct_k = max(1, int(hidden_dim * 0.01))
    top5pct_k = max(1, int(hidden_dim * 0.05))
    top10pct_k = max(1, int(hidden_dim * 0.10))
    
    top1pct_energy = abs_h.topk(top1pct_k).values.sum().item()
    top5pct_energy = abs_h.topk(top5pct_k).values.sum().item()
    top10pct_energy = abs_h.topk(top10pct_k).values.sum().item()
    
    top1pct_ratio = top1pct_energy / total_energy if total_energy > 0 else 0
    top5pct_ratio = top5pct_energy / total_energy if total_energy > 0 else 0
    top10pct_ratio = top10pct_energy / total_energy if total_energy > 0 else 0
    
    # L0 "norm": proportion of elements greater than threshold (threshold = mean + 1 * std)
    mean_val = abs_h.mean()
    std_val = abs_h.std()
    threshold_std = mean_val + 1.0 * std_val
    l0_norm = (abs_h > threshold_std).float().mean().item()
    
    # Gini coefficient (measures distribution inequality, larger value means more sparse/uneven)
    sorted_h = torch.sort(abs_h)[0]
    n = len(sorted_h)
    index = torch.arange(1, n + 1).float().to(sorted_h.device)
    gini = (2 * (index * sorted_h).sum()) / (n * sorted_h.sum()) - (n + 1) / n
    gini = gini.item() if sorted_h.sum() > 0 else 0
    
    # Effective Rank (based on L1 entropy), normalized to [0, 1]
    normalized_l1 = abs_h / abs_h.sum()
    normalized_l1 = normalized_l1 + 1e-10
    effective1_rank_entropy = -(normalized_l1 * torch.log(normalized_l1)).sum()
    effective1_rank = torch.exp(effective1_rank_entropy).item()
    effective1_rank_ratio = effective1_rank / hidden_dim
    
    return {
        'l0_norm': l0_norm,
        'top1pct_ratio': top1pct_ratio,
        'top5pct_ratio': top5pct_ratio,
        'top10pct_ratio': top10pct_ratio,
        'gini': gini,
        'effective_rank': effective1_rank_ratio
    }

def get_last_hidden_state(model, tokenizer, text, device):
    """Get the hidden state corresponding to the last token of the text (last layer)"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_layer_hidden_state = outputs.hidden_states[-1]
        else:
            raise AttributeError("Cannot retrieve hidden states")
        
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1 
            batch_size = last_layer_hidden_state.size(0)
            
            last_token_states = last_layer_hidden_state[
                torch.arange(batch_size), 
                seq_lengths
            ]
        else:
            last_token_states = last_layer_hidden_state[:, -1, :]
        
        if last_token_states.size(0) == 1:
            last_token_states = last_token_states.squeeze(0)
    
    return last_token_states.detach().clone()

def compute_sample_sparsity(model, tokenizer, sample, device):
    """Compute sparsity metrics for a single sample"""
    problem = sample['problem']
   
    
    
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Provide the final answer in the end."},
        {"role": "user", "content": f"{problem}\n\nPlease provide the final answer in the end."}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Get last hidden state (hidden vector of the last token)
    hidden_state = get_last_hidden_state(model, tokenizer, formatted_prompt, device)
    
    # Compute sparsity metrics
    metrics = compute_sparsity_metrics(hidden_state)
    
    return metrics

def rank_examples_by_sparsity(example_pool, model_path, metric='l0_norm', verbose=True):
    """
    Rank the example pool by sparsity.
    
    Args:
        example_pool: Dataset object (e.g., Dataset from datasets.load_dataset)
        model_path: Model path or HuggingFace model name for computing sparsity
        metric: Sparsity metric for ranking, options include
                'l0_norm', 'top10pct_ratio', 'gini', 'effective_rank'
        verbose: Whether to print detailed information (loading progress, statistics, etc.)
    
    Returns:
        A new "sorted" dataset: arranged in order from "easy → hard".
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if verbose:
        print("\n" + "="*80)
        print("Loading model to compute sparsity for example pool and sort...")
        print("="*80)
    
    # Load model for computing sparsity
    if verbose:
        print(f"\nLoading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    device = model.device
    
    if verbose:
        print(f"Model loaded to: {device}")
    
    # Iterate through each sample and compute its sparsity metrics
    if verbose:
        print(f"\nComputing sparsity for {len(example_pool)} examples...")
    
    all_metrics = []
    
    iterator = tqdm(range(len(example_pool)), desc="Computing sparsity") if verbose else range(len(example_pool))
    
    for i in iterator:
        sample = example_pool[i]
        try:
            metrics = compute_sample_sparsity(model, tokenizer, sample, device)
            all_metrics.append({
                'index': i,
                'l0_norm': metrics['l0_norm'],
                'top10pct_ratio': metrics['top10pct_ratio'],
                'gini': metrics['gini'],
                'effective_rank': metrics['effective_rank']
            })
        except Exception as e:
            if verbose:
                print(f"\nSample {i} computation failed: {e}")
            all_metrics.append({
                'index': i,
                'l0_norm': 0,
                'top10pct_ratio': 1,
                'gini': 0,
                'effective_rank': 0
            })
    
    # Determine sorting direction (easy first or last) based on different metrics
    if metric == "l0_norm" or metric == "effective_rank":
        reverse_order = True  # High = easy
    else:
        reverse_order = False  # Low = easy
    
    # Sort by specified sparsity metric
    all_metrics.sort(key=lambda x: x[metric], reverse=reverse_order)
    
    # Statistics
    if verbose:
        metric_values = [m[metric] for m in all_metrics]
        print(f"\n{metric} statistics:")
        print(f"  Min: {min(metric_values):.6f}")
        print(f"  Max: {max(metric_values):.6f}")
        print(f"  Mean: {np.mean(metric_values):.6f}")
    
    # Rearrange dataset according to sorted indices
    sorted_indices = [m['index'] for m in all_metrics]
    sorted_scores = [m[metric] for m in all_metrics]
    sorted_pool = example_pool.select(sorted_indices)
    
    # Add corresponding sparsity scores as a column to the dataset for subsequent analysis
    sorted_pool = sorted_pool.add_column("sparsity_score", sorted_scores)
    
    if verbose:
        print(f"\n✅ Example pool sorted by {metric} (easy→hard)")
        print(f"✅ sparsity_score column added to dataset")
    
    # Release model GPU memory
    del model
    torch.cuda.empty_cache()
    
    return sorted_pool


def analyze_difficulty_levels(sorted_pool, n_levels=5, verbose=True):
    """
    Further divide "difficulty levels" on a dataset already sorted by sparsity, and provide thresholds for each level.
    
    Args:
        sorted_pool: Dataset sorted by sparsity from easy to hard (must contain sparsity_score column)
        n_levels: Number of difficulty levels to divide, default 5 levels
        verbose: Whether to print detailed information (statistics for each difficulty level)
    
    Returns:
        sorted_pool: Dataset with difficulty_level column added
        thresholds: Thresholds and statistics for each difficulty level
    """
    if 'sparsity_score' not in sorted_pool.column_names:
        raise ValueError("Dataset must contain sparsity_score column, please use rank_examples_by_sparsity first")
    
    n_samples = len(sorted_pool)
    scores = sorted_pool['sparsity_score']
    
    level_names = ["very_easy", "easy", "medium", "hard", "very_hard"]
    if n_levels != 5:
        level_names = [f"level_{i+1}" for i in range(n_levels)]
    
    # Divide difficulty levels based on sample position in sorted dataset (assuming data is "easy → hard")
    difficulty_levels = []
    samples_per_level = n_samples // n_levels
    
    for i in range(n_samples):
        level = min(i // samples_per_level + 1, n_levels)
        difficulty_levels.append(level)
    
    sorted_pool = sorted_pool.add_column("difficulty_level", difficulty_levels)
    
    # Statistics: range and mean of sparsity_score within each difficulty level
    thresholds = {}
    for level in range(1, n_levels + 1):
        level_indices = [i for i, l in enumerate(difficulty_levels) if l == level]
        level_scores = [scores[i] for i in level_indices]
        
        thresholds[level] = {
            'name': level_names[level - 1] if level <= len(level_names) else f"level_{level}",
            'count': len(level_scores),
            'min': min(level_scores),
            'max': max(level_scores),
            'mean': np.mean(level_scores)
        }
    
    if verbose:
        print(f"\n{'='*70}")
        print("Difficulty level division (based on sparsity_score, data sorted from easy→hard)")
        print(f"{'='*70}")
        print(f"{'Level':<6} {'Name':<12} {'Count':<8} {'Score Range':<32} {'Mean':<12}")
        print(f"{'-'*70}")
        for level, info in thresholds.items():
            print(f"{level:<6} {info['name']:<12} {info['count']:<8} [{info['min']:.6f}, {info['max']:.6f}]    {info['mean']:.6f}")
        print(f"{'='*70}")
        print(f"✅ difficulty_level column added (1=easiest, {n_levels}=hardest)")
    
    return sorted_pool, thresholds


def get_difficulty_level(score, thresholds):
    """Find the corresponding difficulty level in existing thresholds based on sparsity_score.
    
    If score is outside the range of all current thresholds, return the "closest" difficulty level:
    - score below all thresholds: return easiest level (level 1)
    - score above all thresholds: return hardest level (max level)
    """
    # First try exact match within threshold ranges
    for level, info in thresholds.items():
        if info['min'] <= score <= info['max']:
            return level, info['name']
    
    # If no exact match, find the "closest" level among all levels
    min_level = min(thresholds.keys())
    max_level = max(thresholds.keys())
    
    # If score is below the minimum of all levels, return easiest level
    if score < thresholds[min_level]['min']:
        return min_level, thresholds[min_level]['name']
    
    # If score is above the maximum of all levels, return hardest level
    if score > thresholds[max_level]['max']:
        return max_level, thresholds[max_level]['name']
    
    # Fallback: find the level with median sparsity_score closest to the score
    best_level = min_level
    best_dist = float('inf')
    for level, info in thresholds.items():
        mid = (info['min'] + info['max']) / 2
        dist = abs(score - mid)
        if dist < best_dist:
            best_dist = dist
            best_level = level
    
    return best_level, thresholds[best_level]['name']

