"""
Compare sparsity of model hidden states under knowledge-conflicting vs non-conflicting scenarios
"""

import json
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def setup_environment():
    """Clean environment variables that might cause issues"""
    problematic_vars = []
    env_vars_to_check = [
        'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN', 'HF_HUB_USER_AGENT',
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
        'ALL_PROXY', 'all_proxy', 'USER', 'USERNAME'
    ]
    
    for var_name in env_vars_to_check:
        value = os.environ.get(var_name)
        if value:
            try:
                value.encode('latin-1')
            except UnicodeEncodeError as e:
                problematic_vars.append(var_name)
    
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    vars_to_remove = [
        'HF_HUB_USER_AGENT', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN',
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
        'ALL_PROXY', 'all_proxy'
    ]
    
    for var in vars_to_remove:
        if var in os.environ:
            del os.environ[var]

setup_environment()

def compute_sparsity_metrics(hidden_state):
    """Compute various sparsity metrics"""
    h = hidden_state.squeeze()
    hidden_dim = h.shape[0]
    
    abs_h = h.abs()
    
    # L1 Norm: sum of absolute values
    l1_norm = abs_h.sum().item()
    
    # Top-K percentage (based on L2 energy)
    total_energy = (abs_h ** 2).sum().item()
    
    top1pct_k = max(1, int(hidden_dim * 0.01))
    top5pct_k = max(1, int(hidden_dim * 0.05))
    top10pct_k = max(1, int(hidden_dim * 0.10))
    
    top1pct_energy = (abs_h.topk(top1pct_k).values ** 2).sum().item()
    top5pct_energy = (abs_h.topk(top5pct_k).values ** 2).sum().item()
    top10pct_energy = (abs_h.topk(top10pct_k).values ** 2).sum().item()
    
    top1pct_ratio = top1pct_energy / total_energy if total_energy > 0 else 0
    top5pct_ratio = top5pct_energy / total_energy if total_energy > 0 else 0
    top10pct_ratio = top10pct_energy / total_energy if total_energy > 0 else 0
    
    # Gini coefficient
    sorted_h = torch.sort(abs_h)[0]
    n = len(sorted_h)
    index = torch.arange(1, n + 1).float().to(sorted_h.device)
    gini = (2 * (index * sorted_h).sum()) / (n * sorted_h.sum()) - (n + 1) / n
    gini = gini.item()
    
    # Effective Rank (L1-based)
    normalized_l1 = abs_h / abs_h.sum()
    normalized_l1 = normalized_l1 + 1e-10
    effective1_rank_entropy = -(normalized_l1 * torch.log(normalized_l1)).sum()
    effective1_rank = torch.exp(effective1_rank_entropy).item()
    effective1_rank_ratio = effective1_rank / hidden_dim
    
    # Hoyer Sparsity
    l1_sum = abs_h.sum().item()
    l2_sum = torch.sqrt((abs_h**2).sum()).item()
    sqrt_n = np.sqrt(hidden_dim)
    hoyer = (sqrt_n - (l1_sum / (l2_sum + 1e-10))) / (sqrt_n - 1)
    
    return {
        'top1pct_ratio': top1pct_ratio,
        'top5pct_ratio': top5pct_ratio,
        'top10pct_ratio': top10pct_ratio,
        'l1_norm': l1_norm,
        'gini': gini,
        'effective1_rank': effective1_rank_ratio,
        'hoyer_sparsity': hoyer
    }

def analyze_prompt(context, question, model, tokenizer):
    """
    Analyze sparsity for given context and question
    Returns sparsity of the last token in the last layer
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    messages = [
        {"role": "system", "content": "You are a helpful Knowledge Conflict assistant. You should judge the knowledge in the sentence is correct or not. You should answer with ONLY a single letter: True or False. Do not explain or add any other text."},
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    
    input_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_token_state = last_hidden_state[:, -1, :]
        
        sparsity = compute_sparsity_metrics(last_token_state)
        sparsity['input_length'] = input_length
        
        return sparsity

def main():
    parser = argparse.ArgumentParser(description="Compare sparsity under knowledge conflict vs non-conflict scenarios")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                        help='Model name or path')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--data_path', type=str, default='./data.json',
                        help='Path to the knowledge conflict dataset')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to analyze')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face token (optional)')
    args = parser.parse_args()
    
    # Login to Hugging Face if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    
    print("="*80)
    print("Knowledge Conflict vs Non-Conflict Sparsity Comparison")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map={"": f"cuda:{args.gpu_id}"}
    )
    print("Model loaded\n")
    
    # Load data
    print(f"Loading data: {args.data_path}")
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Number of subjects: {len(data)}")
    
    # Flatten all samples
    all_samples = []
    for subject, samples in data.items():
        for sample in samples:
            sample['subject'] = subject
            all_samples.append(sample)
    
    print(f"Total samples: {len(all_samples)}\n")
    
    # Analyze samples
    max_samples = min(args.max_samples, len(all_samples))
    print(f"Analyzing first {max_samples} samples...\n")
    
    conflicting_results = []
    non_conflicting_results = []
    
    for i in range(max_samples):
        item = all_samples[i]
        
        print(f"[{i+1}/{max_samples}] [{item.get('subject', 'Unknown')}] {item['main_entity']}")
        
        try:
            # 1. Conflicting scenario
            conflict_sparsity = analyze_prompt(
                context=item['conflicting_knowledge'],
                question=item['question_about_conflicting_segments'],
                model=model,
                tokenizer=tokenizer
            )
            conflict_sparsity['type'] = 'conflict'
            conflict_sparsity['entity'] = item['main_entity']
            conflict_sparsity['subject'] = item.get('subject', 'Unknown')
            conflict_sparsity['sample_id'] = i
            conflicting_results.append(conflict_sparsity)
            
            # 2. Non-conflicting scenario
            non_conflict_sparsity = analyze_prompt(
                context=item['parametric_knowledge'],
                question=item['question_about_nonconflicting_segments'],
                model=model,
                tokenizer=tokenizer
            )
            non_conflict_sparsity['type'] = 'non_conflict'
            non_conflict_sparsity['entity'] = item['main_entity']
            non_conflict_sparsity['subject'] = item.get('subject', 'Unknown')
            non_conflict_sparsity['sample_id'] = i
            non_conflicting_results.append(non_conflict_sparsity)
            
            # Print comparison
            print(f"  Conflict    - L1: {conflict_sparsity['l1_norm']:.4f}, Hoyer: {conflict_sparsity['hoyer_sparsity']:.4f}, Gini: {conflict_sparsity['gini']:.4f}")
            print(f"  NonConflict - L1: {non_conflict_sparsity['l1_norm']:.4f}, Hoyer: {non_conflict_sparsity['hoyer_sparsity']:.4f}, Gini: {non_conflict_sparsity['gini']:.4f}")
            
            l1_diff = conflict_sparsity['l1_norm'] - non_conflict_sparsity['l1_norm']
            hoyer_diff = conflict_sparsity['hoyer_sparsity'] - non_conflict_sparsity['hoyer_sparsity']
            marker = "🔴" if abs(l1_diff) > 0.02 else "⚪"
            print(f"  {marker} Δ L1: {l1_diff:+.4f}, Δ Hoyer: {hoyer_diff:+.4f}\n")
            
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
    
    # Statistical analysis
    print("\n" + "="*80)
    print("Statistical Analysis")
    print("="*80)
    
    if len(conflicting_results) > 0 and len(non_conflicting_results) > 0:
        print(f"\nSuccessfully analyzed samples: {len(conflicting_results)}")
        
        metrics = ['l1_norm', 'top5pct_ratio', 'top10pct_ratio', 'gini', 'effective1_rank', 'hoyer_sparsity']
        
        print("\n" + "-"*80)
        print(f"{'Metric':<20} {'Conflict':>15} {'Non-Conflict':>15} {'Δ (C-NC)':>15}")
        print("-"*80)
        
        for metric in metrics:
            conflict_avg = np.mean([r[metric] for r in conflicting_results])
            non_conflict_avg = np.mean([r[metric] for r in non_conflicting_results])
            diff = conflict_avg - non_conflict_avg
            
            print(f"{metric:<20} {conflict_avg:>15.4f} {non_conflict_avg:>15.4f} {diff:>+15.4f}")
        
        print("\n" + "-"*80)
        print("Significance (|Δ L1| > 0.02)")
        print("-"*80)
        
        significant_diffs = 0
        for c_result, nc_result in zip(conflicting_results, non_conflicting_results):
            if abs(c_result['l1_norm'] - nc_result['l1_norm']) > 0.02:
                significant_diffs += 1
        
        print(f"Significant differences: {significant_diffs}/{len(conflicting_results)} ({significant_diffs/len(conflicting_results)*100:.1f}%)")
        
        print("\n" + "="*80)
        print("🎯 Conclusions")
        print("="*80)
        
        conflict_l1 = np.mean([r['l1_norm'] for r in conflicting_results])
        non_conflict_l1 = np.mean([r['l1_norm'] for r in non_conflicting_results])
        
        conflict_gini = np.mean([r['gini'] for r in conflicting_results])
        non_conflict_gini = np.mean([r['gini'] for r in non_conflicting_results])
        
        conflict_hoyer = np.mean([r['hoyer_sparsity'] for r in conflicting_results])
        non_conflict_hoyer = np.mean([r['hoyer_sparsity'] for r in non_conflicting_results])
        
        if conflict_l1 < non_conflict_l1:
            print(f"✅ Conflict is sparser (L1: {conflict_l1:.4f} < {non_conflict_l1:.4f})")
        else:
            print(f"✅ Non-conflict is sparser (L1: {non_conflict_l1:.4f} < {conflict_l1:.4f})")
        
        if conflict_gini > non_conflict_gini:
            print(f"✅ Conflict is sparser (Gini: {conflict_gini:.4f} > {non_conflict_gini:.4f})")
        else:
            print(f"✅ Non-conflict is sparser (Gini: {non_conflict_gini:.4f} > {conflict_gini:.4f})")
            
        if conflict_hoyer > non_conflict_hoyer:
            print(f"✅ Conflict is sparser (Hoyer: {conflict_hoyer:.4f} > {non_conflict_hoyer:.4f})")
        else:
            print(f"✅ Non-conflict is sparser (Hoyer: {non_conflict_hoyer:.4f} > {conflict_hoyer:.4f})")
        
        # Visualization - ICML standard
        print("\n" + "="*80)
        print("Generating visualization (ICML standard)...")
        print("="*80)
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.0,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'text.usetex': False,
            'axes.unicode_minus': False
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Sparsity Comparison: Conflict vs Non-Conflict', 
                     fontsize=14, fontweight='normal', y=0.98)
        
        conflict_color = '#2c5985'
        non_conflict_color = '#88b3dd'
        
        metrics_display = [
            ('l1_norm', '(a) L1 Norm\n(Lower = Sparser)'),
            ('hoyer_sparsity', '(b) Hoyer Sparsity\n(Higher = Sparser)'),
            ('top5pct_ratio', '(c) Top 5% Energy\n(Higher = Sparser)'),
            ('top10pct_ratio', '(d) Top 10% Energy\n(Higher = Sparser)'),
            ('effective1_rank', '(e) Effective Rank\n(Lower = Sparser)'),
        ]
        
        for idx, (metric, title) in enumerate(metrics_display):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            conflict_vals = [r[metric] for r in conflicting_results]
            non_conflict_vals = [r[metric] for r in non_conflicting_results]
            
            bp = ax.boxplot([conflict_vals, non_conflict_vals], 
                           labels=['Conflict', 'Non-Conflict'],
                           patch_artist=True,
                           showmeans=True,
                           widths=0.4,
                           medianprops=dict(color='black', linewidth=1.5),
                           meanprops=dict(marker='D', markerfacecolor='white', 
                                        markeredgecolor='black', markersize=5),
                           boxprops=dict(linewidth=1.2),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2))
            
            bp['boxes'][0].set_facecolor(conflict_color)
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor(non_conflict_color)
            bp['boxes'][1].set_alpha(0.7)
            
            conflict_mean = np.mean(conflict_vals)
            non_conflict_mean = np.mean(non_conflict_vals)
            diff = conflict_mean - non_conflict_mean
            
            ax.set_title(title, fontsize=11, fontweight='normal', pad=8)
            ax.set_ylabel('Value', fontsize=11)
            ax.tick_params(axis='x', labelsize=10)
            
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, axis='y')
            ax.set_axisbelow(True)
            
            textstr = f'Δ = {diff:+.4f}'
            props = dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor='#cccccc', linewidth=1)
            ax.text(0.5, 0.97, textstr, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', 
                   horizontalalignment='center', bbox=props)
            
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(1.0)
                ax.spines[spine].set_color('#2c3e50')
        
        # Summary bar chart
        ax = axes[1, 2]
        metrics_for_bar = ['top5pct_ratio', 'hoyer_sparsity', 'top10pct_ratio', 'effective1_rank']
        metric_labels = ['Top5%', 'Hoyer', 'Top10%', 'EffRank']
        
        conflict_means = [np.mean([r[m] for r in conflicting_results]) for m in metrics_for_bar]
        non_conflict_means = [np.mean([r[m] for r in non_conflicting_results]) for m in metrics_for_bar]
        
        x = np.arange(len(metrics_for_bar))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, conflict_means, width, label='Conflict', 
                      color=conflict_color, alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, non_conflict_means, width, label='Non-Conflict', 
                      color=non_conflict_color, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Metrics', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('(f) Summary', fontsize=11, fontweight='normal', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.legend(frameon=True, edgecolor='#cccccc', fancybox=False, 
                 framealpha=0.9, loc='upper left')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, axis='y')
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color('#2c3e50')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save PDF
        model_short_name = args.model_name.split('/')[-1].replace('-', '_')
        pdf_file = os.path.join(args.output_dir, f'knowledge_conflict_sparsity_comparison_{model_short_name}.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', 
                   dpi=300, backend='pdf')
        print(f"\n✅ PDF saved: {pdf_file}")
        
        plt.close()

if __name__ == "__main__":
    main()
