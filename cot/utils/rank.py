"""
使用 Sparsity 对 MATH-500 数据集进行排序的工具模块
用于 Curriculum Learning：从简单示例到困难示例
"""

import torch
import numpy as np
from tqdm import tqdm

def compute_sparsity_metrics(hidden_state):
    """计算稀疏性指标"""
    h = hidden_state.squeeze()
    hidden_dim = h.shape[0]
    
    abs_h = h.abs()
    total_energy = abs_h.sum().item()
    
    # Top-K 百分比
    top1pct_k = max(1, int(hidden_dim * 0.01))
    top5pct_k = max(1, int(hidden_dim * 0.05))
    top10pct_k = max(1, int(hidden_dim * 0.10))
    
    top1pct_energy = abs_h.topk(top1pct_k).values.sum().item()
    top5pct_energy = abs_h.topk(top5pct_k).values.sum().item()
    top10pct_energy = abs_h.topk(top10pct_k).values.sum().item()
    
    top1pct_ratio = top1pct_energy / total_energy if total_energy > 0 else 0
    top5pct_ratio = top5pct_energy / total_energy if total_energy > 0 else 0
    top10pct_ratio = top10pct_energy / total_energy if total_energy > 0 else 0
    
    # L0 范数
    mean_val = abs_h.mean()
    std_val = abs_h.std()
    threshold_std = mean_val + 1.0 * std_val
    l0_norm = (abs_h > threshold_std).float().mean().item()
    
    # Gini 系数
    sorted_h = torch.sort(abs_h)[0]
    n = len(sorted_h)
    index = torch.arange(1, n + 1).float().to(sorted_h.device)
    gini = (2 * (index * sorted_h).sum()) / (n * sorted_h.sum()) - (n + 1) / n
    gini = gini.item() if sorted_h.sum() > 0 else 0
    
    # Effective Rank (L1-based)
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
    """获取文本的最后一个 token 的 hidden state"""
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
            raise AttributeError("无法获取 hidden states")
        
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
    """计算单个样本的 sparsity metrics"""
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
    
    # 获取 last hidden state
    hidden_state = get_last_hidden_state(model, tokenizer, formatted_prompt, device)
    
    # 计算 sparsity metrics
    metrics = compute_sparsity_metrics(hidden_state)
    
    return metrics

def rank_examples_by_sparsity(example_pool, model_path, metric='l0_norm', verbose=True):
    """
    使用 sparsity 对示例池进行排序
    
    参数:
        example_pool: 数据集（例如从 datasets.load_dataset 加载）
        model_path: 模型路径或 HuggingFace 模型名
        metric: 排序指标，可选 'l0_norm', 'top10pct_ratio', 'gini', 'effective_rank'
        verbose: 是否打印详细信息
    
    返回:
        排序后的数据集（简单→困难）
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if verbose:
        print("\n" + "="*80)
        print("加载模型计算示例池的 sparsity 并排序...")
        print("="*80)
    
    # 加载模型
    if verbose:
        print(f"\n加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    device = model.device
    
    if verbose:
        print(f"模型已加载到: {device}")
    
    # 计算每个样本的 sparsity
    if verbose:
        print(f"\n计算 {len(example_pool)} 个示例的 sparsity...")
    
    all_metrics = []
    
    iterator = tqdm(range(len(example_pool)), desc="计算 sparsity") if verbose else range(len(example_pool))
    
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
                print(f"\n样本 {i} 计算失败: {e}")
            all_metrics.append({
                'index': i,
                'l0_norm': 0,
                'top10pct_ratio': 1,
                'gini': 0,
                'effective_rank': 0
            })
    
    # 确定排序方向
    if metric == "l0_norm" or metric == "effective_rank":
        reverse_order = True  # 高=简单
    else:
        reverse_order = False  # 低=简单
    
    # 排序
    all_metrics.sort(key=lambda x: x[metric], reverse=reverse_order)
    
    # 统计
    if verbose:
        metric_values = [m[metric] for m in all_metrics]
        print(f"\n{metric} 统计:")
        print(f"  最小值: {min(metric_values):.6f}")
        print(f"  最大值: {max(metric_values):.6f}")
        print(f"  平均值: {np.mean(metric_values):.6f}")
    
    # 重新排序数据集
    sorted_indices = [m['index'] for m in all_metrics]
    sorted_scores = [m[metric] for m in all_metrics]
    sorted_pool = example_pool.select(sorted_indices)
    
    # 将 sparsity score 添加到数据集中
    sorted_pool = sorted_pool.add_column("sparsity_score", sorted_scores)
    
    if verbose:
        print(f"\n✅ 示例池已按 {metric} 排序（简单→困难）")
        print(f"✅ sparsity_score 列已添加到数据集")
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    return sorted_pool


def analyze_difficulty_levels(sorted_pool, n_levels=5, verbose=True):
    """
    分析排序后的数据集，划分难度等级并给出阈值
    
    参数:
        sorted_pool: 已排序的数据集（需包含 sparsity_score 列，简单→困难）
        n_levels: 难度等级数量，默认5
        verbose: 是否打印详细信息
    
    返回:
        sorted_pool: 带有 difficulty_level 列的数据集
        thresholds: 各难度等级的阈值信息
    """
    if 'sparsity_score' not in sorted_pool.column_names:
        raise ValueError("数据集需包含 sparsity_score 列，请先使用 rank_examples_by_sparsity 排序")
    
    n_samples = len(sorted_pool)
    scores = sorted_pool['sparsity_score']
    
    level_names = ["very_easy", "easy", "medium", "hard", "very_hard"]
    if n_levels != 5:
        level_names = [f"level_{i+1}" for i in range(n_levels)]
    
    # 按位置划分难度等级（数据已按简单→困难排序）
    difficulty_levels = []
    samples_per_level = n_samples // n_levels
    
    for i in range(n_samples):
        level = min(i // samples_per_level + 1, n_levels)
        difficulty_levels.append(level)
    
    sorted_pool = sorted_pool.add_column("difficulty_level", difficulty_levels)
    
    # 计算每个难度等级的阈值
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
        print("难度等级划分 (基于 sparsity_score，数据已按简单→困难排序)")
        print(f"{'='*70}")
        print(f"{'等级':<6} {'名称':<12} {'数量':<8} {'分数范围':<32} {'均值':<12}")
        print(f"{'-'*70}")
        for level, info in thresholds.items():
            print(f"{level:<6} {info['name']:<12} {info['count']:<8} [{info['min']:.6f}, {info['max']:.6f}]    {info['mean']:.6f}")
        print(f"{'='*70}")
        print(f"✅ difficulty_level 列已添加 (1=最简单, {n_levels}=最难)")
    
    return sorted_pool, thresholds


def get_difficulty_level(score, thresholds):
    """根据 sparsity_score 返回难度等级
    
    如果 score 超出阈值范围，返回最近的难度级别：
    - score 低于所有阈值：返回最简单级别（level 1）
    - score 高于所有阈值：返回最难级别（max level）
    """
    # 首先尝试精确匹配
    for level, info in thresholds.items():
        if info['min'] <= score <= info['max']:
            return level, info['name']
    
    # 如果没有精确匹配，找最近的级别
    min_level = min(thresholds.keys())
    max_level = max(thresholds.keys())
    
    # score 低于最小阈值，返回最简单级别
    if score < thresholds[min_level]['min']:
        return min_level, thresholds[min_level]['name']
    
    # score 高于最大阈值，返回最难级别
    if score > thresholds[max_level]['max']:
        return max_level, thresholds[max_level]['name']
    
    # 兜底：找距离最近的级别
    best_level = min_level
    best_dist = float('inf')
    for level, info in thresholds.items():
        mid = (info['min'] + info['max']) / 2
        dist = abs(score - mid)
        if dist < best_dist:
            best_dist = dist
            best_level = level
    
    return best_level, thresholds[best_level]['name']

