"""
Analyze the relationship between difficulty and sparsity in DeepMath-103K dataset
Using continuous difficulty score directly as ranking metric
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import requests
import random
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from scipy import stats


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


def load_deepmath_dataset(cache_dir=None):
    """Load DeepMath-103K dataset from parquet files directly"""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '.deepmath_cache')
    os.makedirs(cache_dir, exist_ok=True)

    parquet_urls = [
        f'https://huggingface.co/api/datasets/zwhe99/DeepMath-103K/parquet/default/train/{i}.parquet'
        for i in range(10)
    ]

    all_dfs = []
    for i, url in enumerate(parquet_urls):
        cache_file = os.path.join(cache_dir, f'train_{i}.parquet')

        if os.path.exists(cache_file):
            print(f"  Loading cached parquet {i+1}/10...")
            df = pd.read_parquet(cache_file)
        else:
            print(f"  Downloading parquet {i+1}/10...")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            df = pd.read_parquet(BytesIO(response.content))
            df.to_parquet(cache_file)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def compute_sparsity_metrics(hidden_state):
    """Compute various sparsity metrics including effective rank"""
    h = hidden_state.squeeze().float()
    hidden_dim = h.shape[0]

    if torch.isnan(h).any() or torch.isinf(h).any():
        return {'l1_norm': 0.0, 'top1pct_ratio': 0.0, 'top5pct_ratio': 0.0,
                'top10pct_ratio': 0.0, 'effective_rank': 0.0}

    abs_h = h.abs()
    total_energy = (abs_h ** 2).sum().item()

    if total_energy == 0 or np.isnan(total_energy) or np.isinf(total_energy):
        return {'l1_norm': abs_h.sum().item(), 'top1pct_ratio': 0.0,
                'top5pct_ratio': 0.0, 'top10pct_ratio': 0.0, 'effective_rank': 0.0}

    top1pct_k = max(1, int(hidden_dim * 0.01))
    top5pct_k = max(1, int(hidden_dim * 0.05))
    top10pct_k = max(1, int(hidden_dim * 0.10))

    top1pct_ratio = (abs_h.topk(top1pct_k).values ** 2).sum().item() / total_energy
    top5pct_ratio = (abs_h.topk(top5pct_k).values ** 2).sum().item() / total_energy
    top10pct_ratio = (abs_h.topk(top10pct_k).values ** 2).sum().item() / total_energy

    # Effective rank: exp(entropy of normalized squared values)
    # Higher effective rank = more uniform distribution = denser
    # Lower effective rank = more concentrated = sparser
    p = (abs_h ** 2) / total_energy  # normalized energy distribution
    p = p + 1e-10  # avoid log(0)
    entropy = -torch.sum(p * torch.log(p)).item()
    effective_rank = np.exp(entropy)

    return {
        'l1_norm': abs_h.sum().item(),
        'top1pct_ratio': top1pct_ratio,
        'top5pct_ratio': top5pct_ratio,
        'top10pct_ratio': top10pct_ratio,
        'effective_rank': effective_rank
    }


def get_last_hidden_state(model, tokenizer, text, device, layer=-1):
    """Get the hidden state of the last token at specified layer"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_layer = outputs.hidden_states[layer]
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            seq_len = attention_mask.sum(dim=1) - 1
            last_token = last_layer[torch.arange(last_layer.size(0)), seq_len]
        else:
            last_token = last_layer[:, -1, :]
        if last_token.size(0) == 1:
            last_token = last_token.squeeze(0)

    return last_token.detach().clone()


def create_scatter_figure(results, output_path, model_name):


    # NeurIPS style settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.usetex': False,
    })

    # NeurIPS full width: 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    model_short = model_name.split('/')[-1]

    # Add figure caption with dataset and model info
    n_samples = len(results)
    fig.suptitle('Sparsity Metrics vs. Problem Difficulty on DeepMath-103K', fontsize=12, fontweight='normal', y=0.95)

    difficulties = np.array([r['difficulty'] for r in results])

    # Define metrics: (key, ylabel, panel_label, expect_negative_corr)
    # expect_negative_corr=True means lower value = sparser (like L1, effective_rank)
    # expect_negative_corr=False means higher value = sparser (like top-k ratios)
    metrics_config = [
        ('l1_norm', 'L1 Norm', '(a)', True),
        ('top5pct_ratio', 'Top-5% Energy Ratio', '(b)', False),
        ('top10pct_ratio', 'Top-10% Energy Ratio', '(c)', False),
        ('effective_rank', 'Effective Rank', '(d)', True),
    ]

    # Color scheme
    colors = ['#2171b5', '#238b45', '#d94801', '#6a51a3']

    for idx, (metric, ylabel, panel_label, expect_neg) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]
        values = np.array([r[metric] for r in results])
        color = colors[idx]

        # Scatter plot with transparency
        ax.scatter(difficulties, values, alpha=0.35, s=12, c=color,
                   edgecolors='none', rasterized=True)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(difficulties, values)
        x_line = np.linspace(difficulties.min(), difficulties.max(), 100)
        y_line = slope * x_line + intercept

        # 95% confidence interval
        n = len(difficulties)
        y_pred = slope * difficulties + intercept
        residuals = values - y_pred
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
        x_mean = np.mean(difficulties)
        ss_x = np.sum((difficulties - x_mean)**2)
        ci = 1.96 * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)

        ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color=color)
        ax.plot(x_line, y_line, color='#c0392b', linewidth=1.5, linestyle='-')

        # Correlation coefficients
        pearson_r, pearson_p = stats.pearsonr(difficulties, values)
        spearman_r, spearman_p = stats.spearmanr(difficulties, values)

        # Check if trend matches hypothesis
        if expect_neg:
            trend_ok = pearson_r < 0
        else:
            trend_ok = pearson_r > 0


        # Stats text box
        stats_text = f'$r$ = {pearson_r:.3f}\n$\\rho$ = {spearman_r:.3f}'

        # Position stats box based on slope
        if slope > 0:
            text_x, text_y, ha, va = 0.03, 0.97, 'left', 'top'
        else:
            text_x, text_y, ha, va = 0.97, 0.97, 'right', 'top'

        ax.text(text_x, text_y, stats_text, transform=ax.transAxes,
                fontsize=8, va=va, ha=ha,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#cccccc', linewidth=0.5, alpha=0.92))

        # Panel label
        ax.text(-0.02, 1.08, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')

        # Labels
        ax.set_xlabel('Difficulty Score', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.3, color='#cccccc')
        ax.set_axisbelow(True)

        # Ticks
        ax.tick_params(axis='both', which='major', length=3, width=0.6,
                       direction='out', colors='#333333')

    # Tight layout with space for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.8, w_pad=1.5)

    # Save PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, backend='pdf')



    plt.close()
    print(f"✓ Generated figure: {output_path}")
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to analyze (None = use all)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Output directory for figures')
    parser.add_argument('--layer', type=int, default=-1,
                        help='Which layer to use (-1 for last, -2 for second-to-last, etc.)')
    args = parser.parse_args()

    # Generate filenames with model name
    model_short = args.model_name.split('/')[-1]
    output_pdf = os.path.join(args.output_dir, f'deepmath_sparsity_{model_short}.pdf')

    setup_environment()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"Using GPU: {args.gpu_id}")
    print(f"Using layer: {args.layer}")
    print(f"Output PDF: {output_pdf}")

    # Load dataset
    print("\n" + "="*80)
    print("Loading DeepMath-103K dataset...")
    print("="*80)
    dataset = load_deepmath_dataset()
    print(f"Total samples: {len(dataset)}")

    # Analyze difficulty distribution
    difficulties = dataset['difficulty'].values
    print(f"\nDifficulty statistics:")
    print(f"  Range: [{difficulties.min():.4f}, {difficulties.max():.4f}]")
    print(f"  Mean: {difficulties.mean():.4f}")
    print(f"  Std: {difficulties.std():.4f}")
    print(f"  Median: {np.median(difficulties):.4f}")

    # Sample selection
    if args.num_samples is None:
        sample_indices = list(range(len(dataset)))
        print(f"\nUsing ALL {len(sample_indices)} samples for analysis")
    else:
        random.seed(42)
        sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        print(f"\nSampled {len(sample_indices)} samples for analysis")

    # Load model
    print("\n" + "="*80)
    print(f"Loading model: {args.model_name}")
    print("="*80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded to: {device}")

    # Compute sparsity
    print("\n" + "="*80)
    print("Computing sparsity metrics...")
    print("="*80)

    results = []

    for idx, sample_idx in enumerate(sample_indices):
        sample = dataset.iloc[sample_idx]
        question = sample['question']
        difficulty = sample['difficulty']

        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve the math problem and just provide the final answer."},
            {"role": "user", "content": f"{question}\n\nPlease give me the final answer."}
        ]

        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt = f"Question: {question}\n\nAnswer:"

        try:
            hidden = get_last_hidden_state(model, tokenizer, prompt, device, args.layer)
            metrics = compute_sparsity_metrics(hidden)
            results.append({
                'sample_idx': sample_idx,
                'difficulty': float(difficulty),
                **metrics
            })
        except Exception as e:
            print(f"  Sample {sample_idx} failed: {e}")

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(sample_indices)}] done")

    print(f"\nSuccessfully processed {len(results)} samples")

   

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Statistical Analysis
    print("\n" + "="*80)
    print("Statistical Analysis (Continuous Difficulty)")
    print("="*80)

    all_diffs = [r['difficulty'] for r in results]
    all_l1 = [r['l1_norm'] for r in results]
    all_top5 = [r['top5pct_ratio'] for r in results]
    all_top10 = [r['top10pct_ratio'] for r in results]
    all_effrank = [r['effective_rank'] for r in results]

    # Compute correlations for all metrics
    metrics_analysis = [
        ('L1 Norm', all_l1, True),           # expect negative (lower = sparser)
        ('Top-5% Energy', all_top5, False),   # expect positive (higher = sparser)
        ('Top-10% Energy', all_top10, False), # expect positive (higher = sparser)
        ('Effective Rank', all_effrank, True), # expect negative (lower = sparser)
    ]

    print(f"\n{'Metric':<20} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 70)

    support_count = 0
    for name, values, expect_neg in metrics_analysis:
        corr_p, p_p = stats.pearsonr(all_diffs, values)
        corr_s, p_s = stats.spearmanr(all_diffs, values)

        # Check if supports hypothesis
        if expect_neg:
            supports = corr_p < 0 and p_p < 0.05
        else:
            supports = corr_p > 0 and p_p < 0.05

        if supports:
            support_count += 1

        marker = "✓" if supports else "✗"
        print(f"{name:<20} {corr_p:>11.4f} {p_p:>12.2e} {corr_s:>11.4f} {p_s:>12.2e}  {marker}")

    # Hypothesis test summary
    print("\n" + "="*80)
    print("Hypothesis Test: Harder problems -> Sparser representations?")
    print("="*80)

    print(f"\nMetrics supporting hypothesis: {support_count}/4")
    print("\nExpected trends for 'harder = sparser':")
    print("  - L1 Norm: negative correlation (lower activation sum)")
    print("  - Top-K Energy: positive correlation (more concentrated)")
    print("  - Effective Rank: negative correlation (fewer effective dimensions)")

    if support_count == 4:
        print("\n" + "="*80)
        print("✓ STRONGLY CONFIRMED: All metrics support sparsity hypothesis")
        print("="*80)
    elif support_count >= 2:
        print("\n" + "="*80)
        print(f"~ PARTIAL SUPPORT: {support_count}/4 metrics support hypothesis")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ NOT SUPPORTED: Insufficient evidence for sparsity hypothesis")
        print("="*80)

    # Generate figure
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80)
    create_scatter_figure(results, output_pdf, args.model_name)

    print("\n" + "="*80)
    print("Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
