"""
Analyze the relationship between accuracy and sparsity across different difficulty levels in MATH-500
For Qwen2.5-7B model
"""

import os
import sys
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from vllm import LLM, SamplingParams

# Import answer matching tools
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from math_equivalence import is_equiv, get_answer

def compute_sparsity_metrics(hidden_state):
    """Compute sparsity metrics (following standard definitions)"""
    h = hidden_state.squeeze()
    hidden_dim = h.shape[0]
    
    # Convert to float32 to avoid numerical overflow
    h = h.float()
    
    # Check and handle abnormal values
    if torch.isnan(h).any() or torch.isinf(h).any():
        print("Warning: hidden state contains NaN or Inf, using default values")
        return {
            'l1_norm': 0.0,
            'top10pct_ratio': 0.0
        }
    
    abs_h = h.abs()
    
    # L1 Norm (standard: sum of absolute values)
    l1_norm = abs_h.sum().item()
    
    # Top-10% Energy Ratio (standard: based on L2 energy)
    total_energy = (abs_h ** 2).sum().item()
    
    # Check if total_energy is valid
    if total_energy == 0 or np.isnan(total_energy) or np.isinf(total_energy):
        print(f"Warning: total_energy abnormal = {total_energy}, using default values")
        return {
            'l1_norm': l1_norm,
            'top10pct_ratio': 0.0
        }
    
    top10pct_k = max(1, int(hidden_dim * 0.10))
    # Get top-K by absolute value, then square and sum
    top10pct_energy = (abs_h.topk(top10pct_k).values ** 2).sum().item()
    top10pct_ratio = top10pct_energy / total_energy
    
    # Check result
    if np.isnan(top10pct_ratio) or np.isinf(top10pct_ratio):
        print(f"Warning: top10pct_ratio = {top10pct_ratio}, using default value")
        top10pct_ratio = 0.0
    
    return {
        'l1_norm': l1_norm,
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

def create_accuracy_sparsity_plot(level_stats, output_path):
    """Create accuracy vs sparsity plot - ICML standard"""
    
    # ICML paper standard style settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.usetex': False
    })
    
    # Create 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    fig.suptitle('Accuracy vs Sparsity: Qwen 2.5 7B on MATH-500', 
                 fontsize=15, fontweight='normal', y=1.02)
    
    levels = sorted(level_stats.keys())
    accuracies = [level_stats[l]['accuracy'] for l in levels]
    l1_norms = [level_stats[l]['l1_norm'] for l in levels]
    top10_ratios = [level_stats[l]['top10pct_ratio'] for l in levels]
    
    # Colors
    line_color = '#2c5985'
    marker_color = '#4682b4'
    
    # Subplot (a): Accuracy vs L1 Norm
    ax1 = axes[0]
    ax1.plot(levels, accuracies, '-o', color=line_color, 
             markerfacecolor=marker_color, markeredgecolor='black',
             markeredgewidth=1, linewidth=2.5, markersize=10, label='Accuracy')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(levels, l1_norms, '-s', color='#d9534f',
                  markerfacecolor='#ff6b6b', markeredgecolor='black',
                  markeredgewidth=1, linewidth=2.5, markersize=8, 
                  label='L1 Norm', alpha=0.8)
    
    # Calculate correlation coefficient
    corr_l1, _ = stats.pearsonr(accuracies, l1_norms)
    
    ax1.set_xlabel('Difficulty Level', fontsize=13)
    ax1.set_ylabel('Accuracy', fontsize=13, color=line_color)
    ax1_twin.set_ylabel('L1 Norm', fontsize=13, color='#d9534f')
    ax1.set_title('(a) Accuracy vs L1 Norm', fontsize=13, pad=10)
    ax1.set_xticks(levels)
    ax1.set_xticklabels([f'{l}' for l in levels])
    ax1.tick_params(axis='y', labelcolor=line_color)
    ax1_twin.tick_params(axis='y', labelcolor='#d9534f')
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, axis='y')
    ax1.set_axisbelow(True)
    
    # Add correlation coefficient
    textstr = f'r = {corr_l1:.3f}'
    props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                alpha=0.9, edgecolor='#cccccc', linewidth=1)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
           fontsize=11, verticalalignment='top', bbox=props, family='monospace')
    
    # Simplify borders
    for spine in ['top']:
        ax1.spines[spine].set_visible(False)
        ax1_twin.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax1.spines[spine].set_linewidth(1.0)
        ax1.spines[spine].set_color('#2c3e50')
    ax1_twin.spines['right'].set_linewidth(1.0)
    ax1_twin.spines['right'].set_color('#2c3e50')
    
    # Subplot (b): Accuracy vs Top 10% Energy
    ax2 = axes[1]
    ax2.plot(levels, accuracies, '-o', color=line_color,
             markerfacecolor=marker_color, markeredgecolor='black',
             markeredgewidth=1, linewidth=2.5, markersize=10, label='Accuracy')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(levels, top10_ratios, '-s', color='#2d7a3e',
                  markerfacecolor='#3d9a4f', markeredgecolor='black',
                  markeredgewidth=1, linewidth=2.5, markersize=8,
                  label='Top 10% Energy', alpha=0.8)
    
    # Calculate correlation coefficient
    corr_top10, _ = stats.pearsonr(accuracies, top10_ratios)
    
    ax2.set_xlabel('Difficulty Level', fontsize=13)
    ax2.set_ylabel('Accuracy', fontsize=13, color=line_color)
    ax2_twin.set_ylabel('Top 10% Energy Ratio', fontsize=13, color='#2d7a3e')
    ax2.set_title('(b) Accuracy vs Top 10% Energy', fontsize=13, pad=10)
    ax2.set_xticks(levels)
    ax2.set_xticklabels([f'{l}' for l in levels])
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2_twin.tick_params(axis='y', labelcolor='#2d7a3e')
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.7, axis='y')
    ax2.set_axisbelow(True)
    
    # Add correlation coefficient
    textstr = f'r = {corr_top10:.3f}'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
           fontsize=11, verticalalignment='top', bbox=props, family='monospace')
    
    # Simplify borders
    for spine in ['top']:
        ax2.spines[spine].set_visible(False)
        ax2_twin.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax2.spines[spine].set_linewidth(1.0)
        ax2.spines[spine].set_color('#2c3e50')
    ax2_twin.spines['right'].set_linewidth(1.0)
    ax2_twin.spines['right'].set_color('#2c3e50')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300, backend='pdf')
    plt.close()
    
    print(f"\n✓ Generated plot: {output_path}")
    print(f"\nCorrelation coefficients:")
    print(f"  Accuracy vs L1 Norm: r = {corr_l1:.4f}")
    print(f"  Accuracy vs Top 10% Energy: r = {corr_top10:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze accuracy vs sparsity relationship in MATH-500")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU device ID')
    parser.add_argument('--max_samples_per_level', type=int, default=None,
                        help='Maximum samples per level (None = use all)')
    parser.add_argument('--output_path', type=str, default='./accuracy_vs_sparsity.pdf',
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
    
    # Organize samples by level
    samples_by_level = defaultdict(list)
    for i, sample in enumerate(dataset):
        level = sample['level']
        samples_by_level[level].append((i, sample))
    
    if args.max_samples_per_level is not None:
        for level in samples_by_level:
            samples_by_level[level] = samples_by_level[level][:args.max_samples_per_level]
    
    print("\nLevel distribution:")
    for level in sorted(samples_by_level.keys()):
        print(f"  Level {level}: {len(samples_by_level[level])} samples")
    
    # ============================================================
    # Step 1: Batch inference using vLLM
    # ============================================================
    print("\n" + "="*80)
    print("Step 1: Batch inference using vLLM")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load vLLM model
    print("\nLoading vLLM model...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        dtype=torch.float16,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        max_model_len=2048
    )
    print("vLLM model loaded")
    
    # Sampling parameters (greedy decoding)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )
    
    # Prepare all prompts
    all_prompts = []
    all_metadata = []  # Store (level, sample_id, ground_truth)
    
    for level in sorted(samples_by_level.keys()):
        print(f"\nPreparing prompts for Level {level}...")
        for sample_id, sample in samples_by_level[level]:
            problem = sample['problem']
            solution = sample['solution']
            
            # Extract ground truth answer using math_equivalence.py tools
            ground_truth_answer = get_answer(solution)
            
            messages = [
                {"role": "system", "content": "You are a helpful math tutor. Solve the math problem step by step and provide the final answer in \\boxed{}."},
                {"role": "user", "content": f"{problem}\n\nPlease solve this step by step and put your final answer in \\boxed{{}}."}
            ]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            all_prompts.append(formatted_prompt)
            all_metadata.append((level, sample_id, ground_truth_answer))
    
    print(f"\nPreparation complete, total {len(all_prompts)} prompts")
    
    # Batch inference
    print("\nStarting batch inference...")
    outputs = llm.generate(all_prompts, sampling_params)
    print("Inference complete")
    
    # Clean up vLLM model
    del llm
    torch.cuda.empty_cache()
    
    # ============================================================
    # Step 2: Compute sparsity using LLM + answer matching
    # ============================================================
    print("\n" + "="*80)
    print("Step 2: Computing sparsity and evaluating accuracy")
    print("="*80)
    
    # Load regular LLM model (for sparsity computation)
    print("\nLoading model (for sparsity computation)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    device = model.device
    print(f"Model loaded to: {device}")
    
    # Process each sample
    results_by_level = defaultdict(list)
    
    for i, output in enumerate(outputs):
        level, sample_id, ground_truth_answer = all_metadata[i]
        generated_text = output.outputs[0].text.strip()
        
        # Use math_equivalence.py tools for answer matching
        # is_equiv automatically extracts answer from generated_text and compares with ground_truth_answer
        is_correct = is_equiv(generated_text, ground_truth_answer, verbose=False)
        
        # Compute sparsity (using prompt's hidden state)
        try:
            prompt = all_prompts[i]
            hidden_state = get_last_hidden_state(model, tokenizer, prompt, device)
            metrics = compute_sparsity_metrics(hidden_state)
            
            result = {
                'sample_id': sample_id,
                'level': level,
                'is_correct': is_correct,
                'generated_text': generated_text,
                'ground_truth': ground_truth_answer,
                **metrics
            }
            
            results_by_level[level].append(result)
            
        except Exception as e:
            print(f"  Sample {sample_id} (Level {level}) processing failed: {e}")
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(outputs)} samples")
    
    # Clean up model
    del model
    torch.cuda.empty_cache()
    
    # ============================================================
    # Statistical analysis
    # ============================================================
    print("\n" + "="*80)
    print("Statistical Analysis Results")
    print("="*80)
    
    level_stats = {}
    
    for level in sorted(results_by_level.keys()):
        results = results_by_level[level]
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r['is_correct'])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # Calculate sparsity statistics
        l1_norms = [r['l1_norm'] for r in results]
        top10_ratios = [r['top10pct_ratio'] for r in results]
        
        level_stats[level] = {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'l1_norm': np.mean(l1_norms),
            'l1_norm_std': np.std(l1_norms),
            'top10pct_ratio': np.mean(top10_ratios),
            'top10pct_ratio_std': np.std(top10_ratios)
        }
        
        print(f"\nLevel {level}:")
        print(f"  Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        print(f"  L1 Norm: {level_stats[level]['l1_norm']:.2f} ± {level_stats[level]['l1_norm_std']:.2f}")
        print(f"  Top 10% Energy: {level_stats[level]['top10pct_ratio']:.6f} ± {level_stats[level]['top10pct_ratio_std']:.6f}")
    
    # ============================================================
    # Generate visualization
    # ============================================================
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80)
    
    create_accuracy_sparsity_plot(level_stats, args.output_path)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
