"""
Analyze the impact of different context lengths on layer-wise sparsity
Using LongReason dataset with different length versions: 8k, 16k, 32k, 64k
Generate line charts to show the trend of sparsity changes across layers
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random

# ============================================================
# Environment Setup
# ============================================================
def setup_environment():
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

# HF token will be provided via command line argument

from datasets import load_dataset

# ============================================================
# Command Line Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the impact of different context lengths on sparsity")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--max_samples", type=int, default=30,
                        help="Number of samples to analyze per dataset version")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Output directory (default: current directory)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token (optional)")
    return parser.parse_args()

# Global variables (initialized in main)
model = None
tokenizer = None
args = None

# ============================================================
# Calculate Sparsity Metrics
# ============================================================
def compute_sparsity_metrics(hidden_state):
    """Calculate multiple sparsity metrics"""
    h = hidden_state.squeeze()
    hidden_dim = h.shape[0]
    
    abs_h = h.abs()
    
    # Top-K Energy Ratio (standard: based on L2 energy, not L1)
    # Correct formula: top-K of |h|, then square and sum, divided by total L2 energy
    squared_h = abs_h ** 2
    total_energy = squared_h.sum().item()
    
    top1pct_k = max(1, int(hidden_dim * 0.01))
    top5pct_k = max(1, int(hidden_dim * 0.05))
    top10pct_k = max(1, int(hidden_dim * 0.10))
    top7pct_k = max(1, int(hidden_dim * 0.07))
    
    # Get top-K absolute values, then square and sum
    top1pct_energy = (abs_h.topk(top1pct_k).values ** 2).sum().item()
    top5pct_energy = (abs_h.topk(top5pct_k).values ** 2).sum().item()
    top10pct_energy = (abs_h.topk(top10pct_k).values ** 2).sum().item()
    top7pct_energy = (abs_h.topk(top7pct_k).values ** 2).sum().item()
    
    top1pct_ratio = top1pct_energy / total_energy if total_energy > 0 else 0
    top5pct_ratio = top5pct_energy / total_energy if total_energy > 0 else 0
    top10pct_ratio = top10pct_energy / total_energy if total_energy > 0 else 0
    top7pct_ratio = top7pct_energy / total_energy if total_energy > 0 else 0
    
    # L1 Norm
    l1_norm = float(abs_h.sum().item())
    
    # Hoyer Sparsity: (√n - ||x||₁/||x||₂) / (√n - 1)
    abs_np = abs_h.cpu().numpy()
    l1_val = float(np.sum(abs_np))
    l2_val = float(np.sqrt(np.sum(abs_np ** 2)))
    n = len(abs_np)
    if l2_val == 0:
        hoyer = 0.0
    else:
        hoyer = float((np.sqrt(n) - l1_val / l2_val) / (np.sqrt(n) - 1))
    
    # Effective1 Rank (L1-based)
    normalized_l1 = abs_h / abs_h.sum()
    normalized_l1 = normalized_l1 + 1e-10
    effective1_rank_entropy = -(normalized_l1 * torch.log(normalized_l1)).sum()
    effective1_rank = torch.exp(effective1_rank_entropy).item()
    effective1_rank_ratio = effective1_rank / hidden_dim
    
    return {
        'top1pct_ratio': top1pct_ratio,
        'top5pct_ratio': top5pct_ratio,
        'top7pct_ratio': top7pct_ratio,
        'top10pct_ratio': top10pct_ratio,
        'l1_norm': l1_norm,
        'hoyer': hoyer,
        'effective1_rank': effective1_rank_ratio
    }

def analyze_question(prompt_text, correct_answer=None):
    """Analyze layer-wise sparsity for a single question"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. For each multiple choice question, you must answer with ONLY a single letter: A, B, C, D, or E. Do not explain or add any other text."},
        {"role": "user", "content": f"{prompt_text}\n\nAnswer with only the letter:"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    
    # Get input length
    input_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get hidden states of all layers (last token)
        all_hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors
        
        # Calculate sparsity for each layer
        layer_sparsity = []
        for layer_idx, hidden_state in enumerate(all_hidden_states):
            last_token_state = hidden_state[:, -1, :]
            sparsity = compute_sparsity_metrics(last_token_state)
            sparsity['layer'] = layer_idx
            layer_sparsity.append(sparsity)
        
        # Generate answer
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode generated answer
        generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Extract the first letter as the predicted answer
        predicted_answer = None
        for char in generated_text:
            if char.upper() in ['A', 'B', 'C', 'D', 'E']:
                predicted_answer = char.upper()
                break
        
        return {
            'input_length': input_length,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'is_correct': (predicted_answer == correct_answer) if (predicted_answer and correct_answer) else None,
            'layer_sparsity': layer_sparsity
        }

# ============================================================
# Main Experiment
# ============================================================
def main():
    global model, tokenizer, args
    
    # Parse command line arguments
    args = parse_args()
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    
    # Fix random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"Set GPU: {args.gpu_id}\n")
    print(f"Random seed fixed: {seed}\n")
    
    print("="*80)
    print("Analyze the impact of different context lengths on sparsity")
    print(f"Model: {args.model_name}")
    print(f"Samples per dataset: {args.max_samples}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded\n")
    
    # Define dataset versions to test
    dataset_splits = {
        '8k': '8k',
        '16k': '16k',
        '32k': '32k',
        '64k': '64k',
    }
    
    all_results = {}
    
    for split_name, split_key in dataset_splits.items():
        print(f"\n{'='*80}")
        print(f"Loading LongReason dataset - {split_name} version...")
        print(f"{'='*80}")
        
       
        ds = load_dataset("lz1bytedance/LongReason", split=split_key)
        
        results = []
        max_samples = min(args.max_samples, len(ds))
        
        print(f"Analyzing {max_samples} samples...\n")
        
        for i in range(max_samples):
            item = ds[i]
            # Use 'prompt' field instead of 'question' to include full long context
            prompt = "This is a background: \n"+item['background'] + "\nThis is a question: please answer the question based on the background.\n"+item['question']
            
            # Get correct answer (assuming 'answer' field in dataset)
            correct_answer = item.get('answer', None)
            
            print(f"[{i+1}/{max_samples}] {split_name}")
            
            try:
                result = analyze_question(prompt, correct_answer)
                result['split'] = split_name
                result['sample_id'] = i
                results.append(result)
                
                correct_mark = "✓" if result['is_correct'] else "✗" if result['is_correct'] is not None else "?"
                print(f"  Input length: {result['input_length']} tokens")
                print(f"  Pred: {result['predicted_answer']}, Truth: {result['correct_answer']} {correct_mark}")
                
                # Print sparsity of the last layer
                last_layer_sparsity = result['layer_sparsity'][-1]
                print(f"  Last layer sparsity - L1: {last_layer_sparsity['l1_norm']:.2f}, Top-5%: {last_layer_sparsity['top5pct_ratio']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        all_results[split_name] = results
            
       
    
    # ============================================================
    # Statistical Analysis
    # ============================================================
    print("\n" + "="*80)
    print("Statistical Analysis")
    print("="*80)
    
    # Accuracy Statistics
    print("\nAccuracy Statistics")
    print("-"*80)
    print(f"{'Split':<12} {'N':>5} {'Correct':>8} {'Accuracy':>10} {'AvgLen':>10}")
    print("-"*80)
    
    split_order = ['8k', '16k', '32k', '64k']
    for split_name in split_order:
        if split_name in all_results and len(all_results[split_name]) > 0:
            results = all_results[split_name]
            n = len(results)
            correct_count = sum([1 for r in results if r.get('is_correct') == True])
            accuracy = correct_count / n if n > 0 else 0
            avg_len = np.mean([r['input_length'] for r in results])
            
            print(f"{split_name:<12} {n:>5} {correct_count:>8} {accuracy:>10.2%} {avg_len:>10.1f}")
    
    # ============================================================
    # Visualization - Layer-wise Sparsity Line Chart
    # ============================================================
    print("\nGenerating Layer-wise Sparsity Line Chart...")
    model_short_name = args.model_name.split('/')[-1].replace('-', '_')
    
    # Color configuration - using academic paper color scheme
    colors = {
        '8k': '#1f77b4',     # Dark Blue
        '16k': '#2ca02c',    # Forest Green
        '32k': '#d62728',    # Brick Red
        '64k': '#ff7f0e'     # Orange
    }
    
    # Calculate average sparsity per layer for each split
    layer_avg_metrics = {}
    num_layers = None
    
    for split_name in split_order:
        if split_name in all_results and len(all_results[split_name]) > 0:
            results = all_results[split_name]
            
            # Get number of layers
            if num_layers is None:
                num_layers = len(results[0]['layer_sparsity'])
            
            # Initialize metric lists for each layer
            layer_metrics = {i: {'l1_norm': [], 'top5pct_ratio': [], 'top7pct_ratio': [], 'top10pct_ratio': [], 'hoyer': []} 
                           for i in range(num_layers)}
            
            # Collect metrics for each layer of each sample
            for result in results:
                for layer_data in result['layer_sparsity']:
                    layer_idx = layer_data['layer']
                    layer_metrics[layer_idx]['l1_norm'].append(layer_data['l1_norm'])
                    layer_metrics[layer_idx]['top5pct_ratio'].append(layer_data['top5pct_ratio'])
                    layer_metrics[layer_idx]['top7pct_ratio'].append(layer_data['top7pct_ratio'])
                    layer_metrics[layer_idx]['top10pct_ratio'].append(layer_data['top10pct_ratio'])
                    layer_metrics[layer_idx]['hoyer'].append(layer_data['hoyer'])
            
            # Calculate averages
            layer_avg = {
                'l1_norm': [np.mean(layer_metrics[i]['l1_norm']) for i in range(num_layers)],
                'top5pct_ratio': [np.mean(layer_metrics[i]['top5pct_ratio']) for i in range(num_layers)],
                'top7pct_ratio': [np.mean(layer_metrics[i]['top7pct_ratio']) for i in range(num_layers)],
                'top10pct_ratio': [np.mean(layer_metrics[i]['top10pct_ratio']) for i in range(num_layers)],
                'hoyer': [np.mean(layer_metrics[i]['hoyer']) for i in range(num_layers)]
            }
            
            layer_avg_metrics[split_name] = layer_avg
    
    # Create 1x5 subplots (in a row)
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    
    metrics_to_plot = [
        ('l1_norm', 'L1 Norm'),
        ('top5pct_ratio', 'Top-5% Energy'),
        ('top7pct_ratio', 'Top-7% Energy'),
        ('top10pct_ratio', 'Top-10% Energy'),
        ('hoyer', 'Hoyer Sparsity')
    ]
    
    layers = list(range(num_layers))
    
    for idx, (metric_key, metric_title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Plot lines for each split
        for split_name in split_order:
            if split_name in layer_avg_metrics:
                values = layer_avg_metrics[split_name][metric_key]
                ax.plot(layers, values, 
                          color=colors[split_name], 
                       linewidth=2.8, 
                       marker='o', 
                       markersize=6,
                       label=f'{split_name.upper()}',
                       alpha=0.9)
        
        ax.set_xlabel('Layer', fontsize=18, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=18, fontweight='bold')
        ax.set_title(f'{metric_title}', fontsize=20, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.0)
        ax.tick_params(axis='both', labelsize=14)
        
        # Set more professional borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        # Add zoomed view - show details of the last 5 layers
        if num_layers >= 5:
            last_n_layers = 5
            zoom_start = num_layers - last_n_layers
            
            # Create inset zoomed plot - placed at bottom right to avoid overlap with legend
            axins = inset_axes(ax, width="38%", height="38%", loc='lower right',
                              bbox_to_anchor=(0, 0, 0.95, 0.95), bbox_transform=ax.transAxes)
            
            # Plot the last few layers in the zoomed plot
            for split_name in split_order:
                if split_name in layer_avg_metrics:
                    values = layer_avg_metrics[split_name][metric_key]
                    last_layers = layers[zoom_start:]
                    last_values = values[zoom_start:]
                    axins.plot(last_layers, last_values, 
                              color=colors[split_name], 
                              linewidth=2.5, 
                              marker='o', 
                              markersize=5,
                              alpha=0.9)
            
            # Set style for zoomed plot
            axins.set_xlim(zoom_start - 0.5, num_layers - 0.5)
            axins.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            axins.tick_params(axis='both', labelsize=11)
            axins.set_facecolor('#f9f9f9')
            axins.set_title(f'Last {last_n_layers} Layers', fontsize=12, fontweight='bold', pad=5)
            
            # Add border highlight
            for spine in axins.spines.values():
                spine.set_linewidth(1.8)
                spine.set_color('#d62728')
                spine.set_linestyle('-')
        
        # Add legend (placed inside top left, avoiding lines)
        ax.legend(fontsize=15, loc='upper left', frameon=True, framealpha=0.98, edgecolor='gray')
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f'layerwise_sparsity_{model_short_name}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Layer-wise sparsity line chart saved: {output_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

