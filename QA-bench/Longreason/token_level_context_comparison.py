"""
Compare token-level sparsity between normal and long-context questions
Left: Normal question (original)
Right: Long-context question (32k)
"""

import json
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
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


def configure_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 120,
        "axes.facecolor": "#F7F9FC",
        "axes.edgecolor": "#CED4DD",
        "axes.linewidth": 1.4,
        "axes.titleweight": "semibold",
        "axes.labelweight": "semibold",
        "axes.labelsize": 20,
        "axes.titlesize": 24,
        "axes.titlepad": 16,
        "axes.grid": True,
        "grid.color": "#D5DBE5",
        "grid.linewidth": 0.8,
        "grid.linestyle": "--",
        "grid.alpha": 0.55,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
        "savefig.dpi": 450,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False
    })


configure_plot_style()

# HF token will be provided via command line argument

from datasets import load_dataset

# ============================================================
# Command Line Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Compare last-token sparsity between normal and long-context questions")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model name")
    parser.add_argument("--gpu_id", type=int, default=2,
                        help="GPU ID to use")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of samples to collect")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Output directory (default: current directory)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face token (optional)")
    return parser.parse_args()

# ============================================================
# Compute Sparsity Metrics
# ============================================================
def compute_sparsity_metrics(hidden_state):
    """Calculate multiple sparsity metrics"""
    h = hidden_state.squeeze()
    hidden_dim = h.shape[0]
    
    abs_h = h.abs()
    
    # Top-K Energy Ratio (standard: based on L2 energy, not L1)
    squared_h = abs_h ** 2
    total_energy = squared_h.sum().item()
    
    top10pct_k = max(1, int(hidden_dim * 0.10))
    top5pct_k = max(1, int(hidden_dim * 0.05))
    
    # Get top-K absolute values, then square and sum
    top10pct_energy = (abs_h.topk(top10pct_k).values ** 2).sum().item()
    top5pct_energy = (abs_h.topk(top5pct_k).values ** 2).sum().item()
    
    top10pct_ratio = top10pct_energy / total_energy if total_energy > 0 else 0
    top5pct_ratio = top5pct_energy / total_energy if total_energy > 0 else 0
    
    # L1 Norm
    l1_norm = float(abs_h.sum().item())
    
    return {
        'top10pct_ratio': top10pct_ratio,
        'top5pct_ratio': top5pct_ratio,
        'l1_norm': l1_norm
    }

def get_last_token_representation(prompt_text, model, tokenizer):
    """Get only the last token's representation and sparsity"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. For each multiple choice question, you must answer with ONLY a single letter: A, B, C, D, or E. "},
        {"role": "user", "content": f"{prompt_text}\n\n Answer with only the letter:"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # Last layer
        
        # Get only the last token's representation
        last_token_hidden = last_hidden_state[0, -1, :]  # shape: (hidden_dim,)
        
        # Calculate sparsity
        sparsity = compute_sparsity_metrics(last_token_hidden.unsqueeze(0))
        
        return last_token_hidden.cpu().numpy(), sparsity, input_ids.shape[1]

def visualize_hidden_state_comparison(normal_data, long_data, model_name, output_path, sample_indices=None):
    """Visualize hidden state distribution and statistics for multiple samples"""

    n_samples = len(normal_data['sparsity'])
    
    if sample_indices is None:
        sample_indices = list(range(n_samples))
    
    # Adjust layout based on number of samples
    if n_samples <= 2:
        nrows, ncols = 1, n_samples
        width_per_sample = 8
        height_per_sample = width_per_sample * 3 / 5
    else:
        # For 3 or more samples, use 2-row layout
        nrows = 2
        ncols = (n_samples + 1) // 2
        width_per_sample = 8
        height_per_sample = width_per_sample * 3 / 5
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_per_sample * ncols, height_per_sample * nrows), sharey=False)
    axes = axes.flatten() if n_samples > 1 else np.array([axes])

    # 不显示主标题
    # fig.suptitle(
    #     f"Last-Token Hidden State Distribution · {model_name}",
    #     fontsize=24,
    #     fontweight="bold",
    #     y=0.98
    # )

    palette_main = {
        "normal": "#E59B9B",
        "normal_edge": "#D87373",
        "long": "#89AEE6",
        "long_edge": "#5F88C4"
    }

    metrics_layout = {
        "L1 Norm": (0.10, 0.75, 0.20, 0.15, "l1_norm"),
        "Top5% Energy": (0.10, 0.45, 0.20, 0.15, "top5pct_ratio"),
        "Top10% Energy": (0.10, 0.15, 0.20, 0.15, "top10pct_ratio")
    }

    for sample_idx in range(n_samples):
        hidden_normal = normal_data['repr'][sample_idx]
        hidden_long = long_data['repr'][sample_idx]
        
        sparsity_normal = normal_data['sparsity'][sample_idx]
        sparsity_long = long_data['sparsity'][sample_idx]
        
        ax_main = axes[sample_idx]

        combined = np.concatenate([hidden_normal, hidden_long])
        percentile_low, percentile_high = np.percentile(combined, [0.5, 99.5])
        if percentile_low == percentile_high:
            percentile_low -= 1e-3
            percentile_high += 1e-3
        bins = np.linspace(percentile_low, percentile_high, 100)

        ax_main.hist(
            hidden_normal,
            bins=bins,
            color=palette_main["normal"],
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            label="Original"
        )
        ax_main.hist(
            hidden_long,
            bins=bins,
            color=palette_main["long"],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
            label="32k Context"
        )

        ax_main.axvline(0, color="#77818F", linestyle="--", linewidth=1.1, alpha=0.75)
        # 不显示子图标题
        # ax_main.set_title(f"Sample {sample_indices[sample_idx] + 1}", fontsize=18, fontweight="bold")
        ax_main.set_xlabel("Hidden Activation Value", fontsize=20)
        ax_main.set_ylabel("Frequency", fontsize=20)
        ax_main.tick_params(axis='both', labelsize=16)
        ax_main.legend(frameon=False, fontsize=16, loc="upper right")
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)

        metrics_pairs = {
            "Original": sparsity_normal,
            "32k": sparsity_long
        }

        for label, (x0, y0, w, h, key) in metrics_layout.items():
            inset = ax_main.inset_axes([x0, y0, w, h])
            values = [metrics_pairs["Original"][key], metrics_pairs["32k"][key]]
            
            # 检查并处理 NaN 或 Inf 值
            if any(np.isnan(v) or np.isinf(v) for v in values):
                print(f"警告: Sample {sample_idx} 的 {label} 包含 NaN 或 Inf 值: {values}")
                values = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in values]
            
            bar_colors = [palette_main["normal"], palette_main["long"]]
            inset.bar([0, 1], values, color=bar_colors, width=0.6)
            inset.set_xticks([0, 1])
            inset.set_xticklabels(["Orig", "32k"], fontsize=12)
            inset.set_title(label, fontsize=14, pad=4, fontweight="semibold")
            ymin = min(values)
            ymax_metric = max(values)
            if ymin == ymax_metric:
                ymin *= 0.9
                ymax_metric *= 1.1 if ymax_metric != 0 else 0.1
            margin = (ymax_metric - ymin) * 0.25 if ymax_metric != ymin else 0.1
            inset.set_ylim(ymin - margin, ymax_metric + margin)
            inset.tick_params(axis='y', labelsize=11)
            inset.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            for spine in inset.spines.values():
                spine.set_color("#B5BFCC")
                spine.set_linewidth(1.0)

    plt.tight_layout(rect=(0, 0, 1, 1), w_pad=3.5)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"可视化已保存 (PDF): {output_path}")
    
    plt.close()

# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()
    
    # 固定随机种子以确保可复现性
    seed = 60
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"设置 GPU: {args.gpu_id}\n")
    print(f"随机种子已固定: {seed}\n")
    
    print("="*80)
    print("对比普通问题和长文本问题的最后token稀疏性")
    print(f"模型: {args.model_name}")
    print(f"样本数量: {args.n_samples}")
    print("="*80)
    
    # 加载模型
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("模型加载完成\n")
    
    # 加载数据集
    print("加载数据集...")
    ds_normal = load_dataset("lz1bytedance/LongReason", split='8k')
    ds_long = load_dataset("lz1bytedance/LongReason", split='32k')
    
    # 随机选择样本索引
    dataset_size = len(ds_normal)
    sample_indices = random.sample(range(dataset_size), args.n_samples)
    print(f"\n随机选择的样本索引: {sample_indices}")
    
    # 收集多个样本的最后token表示
    print(f"\n收集 {args.n_samples} 个样本的最后token表示...")
    
    last_hiddens_normal = []
    last_hiddens_long = []
    sparsity_normal_list = []
    sparsity_long_list = []
    lens_normal = []
    lens_long = []
    
    from tqdm import tqdm
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="处理样本")):
        # 处理 Normal 版本
        item_normal = ds_normal[sample_idx]
        prompt_normal = "This is a background: \n" + item_normal['background'] + "\nThis is a question: please answer the question based on the background.\n" + item_normal['question']
        
        hidden_n, sparsity_n, len_n = get_last_token_representation(prompt_normal, model, tokenizer)
        last_hiddens_normal.append(hidden_n)
        sparsity_normal_list.append(sparsity_n)
        lens_normal.append(len_n)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 处理 Long 版本
        item_long = ds_long[sample_idx]
        prompt_long = "This is a background: \n" + item_long['background'] + "\nThis is a question: please answer the question based on the background.\n" + item_long['question']
        
        if (item_normal['question'] != item_long['question']):
            print(f"Normal: {item_normal['question']}")
            print(f"Long: {item_long['question']}")
            raise ValueError("Normal and Long questions are not the same")
        else:
            print(f"Normal: {item_normal['question']}")
            print(f"Long: {item_long['question']}")
            print("Normal and Long questions are the same")
        
        hidden_l, sparsity_l, len_l = get_last_token_representation(prompt_long, model, tokenizer)
        last_hiddens_long.append(hidden_l)
        sparsity_long_list.append(sparsity_l)
        lens_long.append(len_l)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    # 转换为数组
    last_hiddens_normal = np.array(last_hiddens_normal)  # (n_samples, hidden_dim)
    last_hiddens_long = np.array(last_hiddens_long)      # (n_samples, hidden_dim)
    
    print(f"\n收集完成！")
    print(f"  Normal: {last_hiddens_normal.shape}")
    print(f"  Long: {last_hiddens_long.shape}")
    print(f"  平均长度 - Normal: {np.mean(lens_normal):.0f} tokens")
    print(f"  平均长度 - Long: {np.mean(lens_long):.0f} tokens")
    
    # 统计对比
    l1_normal_mean = np.mean([s['l1_norm'] for s in sparsity_normal_list])
    l1_long_mean = np.mean([s['l1_norm'] for s in sparsity_long_list])
    top5_normal_mean = np.mean([s['top5pct_ratio'] for s in sparsity_normal_list])
    top5_long_mean = np.mean([s['top5pct_ratio'] for s in sparsity_long_list])
    top10_normal_mean = np.mean([s['top10pct_ratio'] for s in sparsity_normal_list])
    top10_long_mean = np.mean([s['top10pct_ratio'] for s in sparsity_long_list])
    
    print(f"\n稀疏性统计:")
    print(f"  L1 Norm - Normal: {l1_normal_mean:.2f}, Long: {l1_long_mean:.2f} (差异: {l1_long_mean-l1_normal_mean:+.2f})")
    print(f"  Top5% - Normal: {top5_normal_mean:.4f}, Long: {top5_long_mean:.4f} (差异: {top5_long_mean-top5_normal_mean:+.4f})")
    print(f"  Top10% - Normal: {top10_normal_mean:.4f}, Long: {top10_long_mean:.4f} (差异: {top10_long_mean-top10_normal_mean:+.4f})")
    
    # 准备数据
    normal_data = {
        'repr': last_hiddens_normal,
        'sparsity': sparsity_normal_list,
        'lens': lens_normal
    }
    
    long_data = {
        'repr': last_hiddens_long,
        'sparsity': sparsity_long_list,
        'lens': lens_long
    }
    
    # 生成对比可视化
    print("\n" + "="*80)
    print("生成隐藏状态向量可视化...")
    print("="*80)
    model_short_name = args.model_name.split('/')[-1].replace('-', '_')
    output_path = os.path.join(args.output_dir, f'last_hidden_state_comparison_{model_short_name}.pdf')
    
    visualize_hidden_state_comparison(normal_data, long_data, args.model_name, output_path, sample_indices)
    
    print("\n完成！")

if __name__ == "__main__":
    main()

