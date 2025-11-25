"""
Analyze the relationship between difficulty levels (1-5) and sparsity in MATH-500 dataset
Hypothesis: Are the hardest problems (level 5) the sparsest?
"""

import os
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compute_sparsity_metrics(hidden_state):
    """Compute various sparsity metrics (following standard definitions)"""
    h = hidden_state.squeeze()
    hidden_dim = h.shape[0]
    
    # Convert to float32 to avoid numerical overflow
    h = h.float()
    
    # Check and handle abnormal values
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("Warning: hidden state contains NaN or Inf, using default values")
        return {
            'l1_norm': 0.0,
            'top1pct_ratio': 0.0,
            'top5pct_ratio': 0.0,
            'top10pct_ratio': 0.0
        }
    
    abs_h = h.abs()
    
    # Top-K Energy Ratio (standard: based on L2 energy)
    total_energy = (abs_h ** 2).sum().item()
    
    # Check if total_energy is valid
    if total_energy == 0 or np.isnan(total_energy) or np.isinf(total_energy):
        print(f"Warning: total_energy abnormal = {total_energy}, using default values")
        return {
            'l1_norm': abs_h.sum().item(),
            'top1pct_ratio': 0.0,
            'top5pct_ratio': 0.0,
            'top10pct_ratio': 0.0
        }
    
    top1pct_k = max(1, int(hidden_dim * 0.01))
    top5pct_k = max(1, int(hidden_dim * 0.05))
    top10pct_k = max(1, int(hidden_dim * 0.10))
    
    # Get top-K by absolute value, then square and sum
    top1pct_energy = (abs_h.topk(top1pct_k).values ** 2).sum().item()
    top5pct_energy = (abs_h.topk(top5pct_k).values ** 2).sum().item()
    top10pct_energy = (abs_h.topk(top10pct_k).values ** 2).sum().item()
    
    top1pct_ratio = top1pct_energy / total_energy
    top5pct_ratio = top5pct_energy / total_energy
    top10pct_ratio = top10pct_energy / total_energy
    
    # Final check of results
    if np.isnan(top10pct_ratio) or np.isinf(top10pct_ratio):
        print(f"Warning: top10pct_ratio = {top10pct_ratio}, top10={top10pct_energy}, total={total_energy}")
        top10pct_ratio = 0.0
    if np.isnan(top5pct_ratio) or np.isinf(top5pct_ratio):
        top5pct_ratio = 0.0
    if np.isnan(top1pct_ratio) or np.isinf(top1pct_ratio):
        top1pct_ratio = 0.0
    
    # L1 Norm (standard: sum of absolute values)
    l1_norm = abs_h.sum().item()
    
    return {
        'l1_norm': l1_norm,
        'top1pct_ratio': top1pct_ratio,
        'top5pct_ratio': top5pct_ratio,
        'top10pct_ratio': top10pct_ratio
    }

def get_last_hidden_state(model, tokenizer, text, device):
    """Get the hidden state of the last token"""
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
            raise AttributeError("Cannot find hidden states")
        
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
    """Compute sparsity metrics for a single sample's last hidden layer"""
    problem = sample['problem']
    
    messages = [
        {"role": "system", "content": "You are a helpful math tutor. Solve the math problem step by step and just provide the final answer."},
        {"role": "user", "content": f"{problem}\n\nPlease just give me the final answer."}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Get last hidden state
    hidden_state = get_last_hidden_state(model, tokenizer, formatted_prompt, device)
    
    # Compute sparsity metrics
    metrics = compute_sparsity_metrics(hidden_state)
    
    return metrics

def create_publication_figure_two_models(results_by_model, output_path):
    """Create publication-quality 2x3 visualization for model comparison (ICML standard)"""
    
    # ICML paper standard style settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.usetex': False
    })
    
    # Create 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    
    fig.suptitle('Sparsity vs Difficulty: Model Comparison', 
                 fontsize=15, fontweight='normal', y=0.98)
    
    # Color-blind friendly color scheme
    colormap_colors = ['#d4e4f7', '#88b3dd', '#4682b4', '#2c5985', '#1a3a5c']
    error_color = '#505050'
    
    # Define model and metric combinations for 6 subplots
    configs = [
        ('meta-llama/Llama-3.2-3B-Instruct', 'l1_norm', 'Llama 3.2 3B', '(c) L1 Norm - Llama 3.2 3B', True),
        ('Qwen/Qwen2.5-7B-Instruct', 'l1_norm', 'Qwen 2.5 7B', '(a) L1 Norm - Qwen 2.5 7B', True),
        ('meta-llama/Llama-3.1-8B-Instruct', 'l1_norm', 'Llama 3.1 8B', '(e) L1 Norm - Llama 3.1 8B', True),
        ('meta-llama/Llama-3.2-3B-Instruct', 'top10pct_ratio', 'Llama 3.2 3B', '(d) Top 10% Energy - Llama 3.2 3B', False),
        ('Qwen/Qwen2.5-7B-Instruct', 'top10pct_ratio', 'Qwen 2.5 7B', '(b) Top 10% Energy - Qwen 2.5 7B', False),
        ('meta-llama/Llama-3.1-8B-Instruct', 'top10pct_ratio', 'Llama 3.1 8B', '(f) Top 10% Energy - Llama 3.1 8B', False)
    ]
    
    for idx, (model_name, metric_name, model_display, title, reverse) in enumerate(configs):
        ax = axes[idx // 3, idx % 3]
        
        # Check if data exists for this model
        if model_name not in results_by_model:
            ax.text(0.5, 0.5, f'No data for {model_display}', 
                   ha='center', va='center', fontsize=11, transform=ax.transAxes)
            ax.set_title(title, fontsize=13, fontweight='normal', pad=10)
            continue
        
        results_by_level = results_by_model[model_name]
        levels = sorted(results_by_level.keys())
        
        # Calculate mean and standard error for each level
        means = []
        stds = []
        for level in levels:
            values = [r[metric_name] for r in results_by_level[level]]
            means.append(np.mean(values))
            stds.append(np.std(values) / np.sqrt(len(values)))
        
        # Use fixed 5 colors for 5 difficulty levels
        bar_colors = colormap_colors[:len(levels)]
        
        # Draw bar chart - ICML style
        x_pos = np.arange(len(levels))
        width = 0.65
        bars = ax.bar(x_pos, means, width=width, yerr=stds, 
                      color=bar_colors, alpha=0.85,
                      edgecolor='#2c3e50', linewidth=1.0,
                      error_kw={'linewidth': 1.5, 'ecolor': error_color, 
                               'elinewidth': 1.5, 'capsize': 4, 'capthick': 1.5})
        
        # Calculate correlation coefficient
        corr, p_value = stats.pearsonr(x_pos, means)
        
        # Check if trend matches expectation
        if reverse:
            expected = corr < 0
            trend_symbol = "✓" if expected else "✗"
        else:
            expected = corr > 0
            trend_symbol = "✓" if expected else "✗"
        
        # Add concise statistics box - ICML style
        textstr = f'r = {corr:.3f} {trend_symbol}'
        props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                    alpha=0.9, edgecolor='#cccccc', linewidth=1)
        
        if idx >= 3:  # Second row
            ax.text(0.03, 0.97, textstr, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='left',
                   bbox=props, family='monospace')
        else:  # First row
            ax.text(0.97, 0.97, textstr, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=props, family='monospace')
        
        # Set labels and title
        metric_label = 'L1 Norm' if metric_name == 'l1_norm' else 'Top 10% Energy Ratio'
        ax.set_xlabel('Difficulty Level', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='normal', pad=8)
        
        # Set x-axis ticks
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{l}' for l in levels], fontsize=11)
        
        # Grid
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, axis='y', color='#999999')
        ax.set_axisbelow(True)
        
        # Simplify borders
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color('#2c3e50')
        
        # Optimize y-axis range
        y_range = max(means) - min(means)
        if y_range > 0:
            padding = y_range * 0.25
            ax.set_ylim([min(means) - padding, max(means) + padding])
        
        # Optimize ticks
        ax.tick_params(axis='both', which='major', length=4, width=1, direction='out')
    
    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5, w_pad=2.0)
    
    # Save as high-quality PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                dpi=300, backend='pdf')
    plt.close()
    
    print(f"✓ Generated ICML standard figure: {output_path}")

def setup_environment():
    """Clean environment variables that might cause issues"""
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    vars_to_remove = [
        'HF_HUB_USER_AGENT', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN',
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
        'ALL_PROXY', 'all_proxy'
    ]
    
    for var in vars_to_remove:
        if var in os.environ:
            del os.environ[var]

def main():
    parser = argparse.ArgumentParser(description="Analyze relationship between difficulty and sparsity in MATH-500")
    parser.add_argument('--models', nargs='+', 
                        default=["Qwen/Qwen2.5-7B-Instruct", 
                                "meta-llama/Llama-3.2-3B-Instruct", 
                                "meta-llama/Llama-3.1-8B-Instruct"],
                        help='List of model names to test')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU device ID')
    parser.add_argument('--max_samples_per_level', type=int, default=None,
                        help='Maximum samples per level (None = use all)')
    parser.add_argument('--output_path', type=str, default='./sparsity_vs_difficulty_model_comparison.pdf',
                        help='Output path for visualization')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face token (optional)')
    args = parser.parse_args()
    
    # Environment setup
    setup_environment()
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"Using GPU: {args.gpu_id}")
    
    # Load dataset
    print("\n" + "="*80)
    print("Loading MATH-500 dataset...")
    print("="*80)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Total samples: {len(dataset)}")
    
    # Count distribution by level
    level_counts = defaultdict(int)
    for sample in dataset:
        level_counts[sample['level']] += 1
    
    print("\nLevel distribution:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} samples")
    
    # Organize samples by level
    samples_by_level = defaultdict(list)
    for i, sample in enumerate(dataset):
        level = sample['level']
        samples_by_level[level].append((i, sample))
    
    # Limit samples per level (optional)
    if args.max_samples_per_level is not None:
        for level in samples_by_level:
            samples_by_level[level] = samples_by_level[level][:args.max_samples_per_level]
    
    # Store results for all models
    results_by_model = {}
    
    # Test each model
    for model_name in args.models:
        print("\n" + "="*80)
        print(f"Testing model: {model_name}")
        print("="*80)
        
        # Load model
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        device = model.device
        print(f"Model loaded to: {device}")
        
        # Compute sparsity metrics for each level
        print("\nComputing sparsity metrics for each sample...")
        
        results_by_level = defaultdict(list)
        
        for level in sorted(samples_by_level.keys()):
            print(f"\nProcessing Level {level} ({len(samples_by_level[level])} samples)...")
            
            for idx, (sample_id, sample) in enumerate(samples_by_level[level]):
                try:
                    metrics = compute_sample_sparsity(model, tokenizer, sample, device)
                    
                    result = {
                        'sample_id': sample_id,
                        'level': level,
                        'subject': sample['subject'],
                        'problem_length': len(sample['problem']),
                        **metrics
                    }
                    
                    results_by_level[level].append(result)
                    
                    if (idx + 1) % 20 == 0:
                        print(f"  [{idx+1}/{len(samples_by_level[level])}] Completed")
                        
                except Exception as e:
                    print(f"  Sample {sample_id} (Level {level}) failed: {e}")
        
        # Save results for this model
        results_by_model[model_name] = results_by_level
        
        # Clean GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    # Statistical analysis (for each model)
    print("\n" + "="*80)
    print("Sparsity Metrics Statistics (by Model and Level)")
    print("="*80)
    
    metrics_names = ['l1_norm', 'top10pct_ratio']
    
    for model_name in args.models:
        results_by_level = results_by_model[model_name]
        print(f"\nModel: {model_name}")
        print("-" * 80)
        
        stats_summary = {}
        
        for level in sorted(results_by_level.keys()):
            print(f"\nLevel {level} ({len(results_by_level[level])} samples):")
            stats_summary[level] = {}
            
            for metric in metrics_names:
                values = [r[metric] for r in results_by_level[level]]
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                median_val = np.median(values)
                
                stats_summary[level][metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'median': median_val
                }
                
                print(f"  {metric}:")
                if metric == 'l1_norm':
                    print(f"    Mean: {mean_val:.2f} ± {std_val:.2f}")
                    print(f"    Median: {median_val:.2f}")
                    print(f"    Range: [{min_val:.2f}, {max_val:.2f}]")
                else:
                    print(f"    Mean: {mean_val:.6f} ± {std_val:.6f}")
                    print(f"    Median: {median_val:.6f}")
                    print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
        
        # Trend analysis
        print("\nTrend Analysis: Level vs Sparsity")
        for metric in metrics_names:
            print(f"\n{metric}:")
            means = [stats_summary[level][metric]['mean'] for level in sorted(stats_summary.keys())]
            
            if metric == 'l1_norm':
                # L1 norm: high=dense=easy, low=sparse=hard
                if means[0] > means[-1]:
                    trend = "Decreasing (Level 1 densest, Level 5 sparsest) ✓ Matches hypothesis"
                else:
                    trend = "Increasing (Level 1 sparsest, Level 5 densest) ✗ Does not match hypothesis"
            else:  # top10pct_ratio
                # Top10% ratio: high=concentrated energy=sparse=hard
                if means[0] < means[-1]:
                    trend = "Increasing (Level 1 dispersed, Level 5 concentrated) ✓ Matches hypothesis"
                else:
                    trend = "Decreasing (Level 1 concentrated, Level 5 dispersed) ✗ Does not match hypothesis"
            
            if metric == 'l1_norm':
                print(f"  Level 1→5 means: {[f'{m:.2f}' for m in means]}")
            else:
                print(f"  Level 1→5 means: {[f'{m:.6f}' for m in means]}")
            print(f"  Trend: {trend}")
        
        # Summary
        print("\nHypothesis Verification: Are the hardest problems (Level 5) the sparsest?")
        
        # Check key metrics
        l0_trend = stats_summary[1]['l1_norm']['mean'] > stats_summary[5]['l1_norm']['mean']
        top10_trend = stats_summary[1]['top10pct_ratio']['mean'] < stats_summary[5]['top10pct_ratio']['mean']
        
        if l0_trend and top10_trend:
            conclusion = "✓ Yes! Data supports hypothesis: Level 5 (hardest) is indeed sparser than Level 1 (easiest)"
        else:
            conclusion = "✗ Not fully supported. The relationship between difficulty and sparsity may be more complex"
        
        print(f"\n{conclusion}")
        print(f"\nKey Findings:")
        print(f"  - L1 Norm: Level 1 = {stats_summary[1]['l1_norm']['mean']:.2f}, Level 5 = {stats_summary[5]['l1_norm']['mean']:.2f}")
        print(f"    (Higher value = denser = easier, Level 1 {'>' if l0_trend else '<'} Level 5: {'✓ Matches' if l0_trend else '✗ Does not match'})")
        print(f"  - Top10% Ratio: Level 1 = {stats_summary[1]['top10pct_ratio']['mean']:.6f}, Level 5 = {stats_summary[5]['top10pct_ratio']['mean']:.6f}")
        print(f"    (Higher value = more concentrated = sparser = harder, Level 1 {'<' if top10_trend else '>'} Level 5: {'✓ Matches' if top10_trend else '✗ Does not match'})")
    
    print("\n" + "="*80)
    
    # Generate visualization
    print("\n" + "="*80)
    print("Generating publication-quality visualization (model comparison)...")
    print("="*80)
    
    create_publication_figure_two_models(results_by_model, args.output_path)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
