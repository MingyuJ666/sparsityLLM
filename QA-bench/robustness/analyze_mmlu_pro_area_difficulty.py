import os
import argparse
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.set_grad_enabled(False)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from datasets import load_dataset

def check_env_vars():
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
    
    return problematic_vars

problematic = check_env_vars()
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
vars_to_remove = [
    'HF_HUB_USER_AGENT', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN',
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
    'ALL_PROXY', 'all_proxy'
]

for var in vars_to_remove:
    if var in os.environ:
        del os.environ[var]

# HF token will be provided via command line argument 

def compute_sparsity_metrics(hidden_state):
    """Compute various sparsity metrics"""
    if len(hidden_state.shape) == 3:
        hidden_state = hidden_state[:, -1, :]  # (batch_size, hidden_dim)
    elif len(hidden_state.shape) == 1:
        hidden_state = hidden_state.unsqueeze(0)  # (1, hidden_dim)
    
    h = hidden_state.detach().cpu().float().numpy()
    
    metrics = {}
    
    for i in range(h.shape[0]):
        sample_h = h[i]  # (hidden_dim,)
        abs_h = np.abs(sample_h)
        
        # 1. L1 norm (standard: sum of absolute values)
        l1_norm = abs_h.sum()
        
        # 2. Top-K energy ratio (standard: based on L2 energy)
        sorted_abs = np.sort(abs_h)[::-1]
        total_energy = (sorted_abs ** 2).sum()
        
        top1pct = max(1, int(0.01 * len(sorted_abs)))
        top5pct = max(1, int(0.05 * len(sorted_abs)))
        top10pct = max(1, int(0.10 * len(sorted_abs)))
        
        top1pct_energy = (sorted_abs[:top1pct] ** 2).sum() / total_energy if total_energy > 0 else 0
        top5pct_energy = (sorted_abs[:top5pct] ** 2).sum() / total_energy if total_energy > 0 else 0
        top10pct_energy = (sorted_abs[:top10pct] ** 2).sum() / total_energy if total_energy > 0 else 0
        
        # 3. Gini coefficient
        sorted_h = np.sort(abs_h)
        n = len(sorted_h)
        cumsum = np.cumsum(sorted_h)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_h)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0
        
        # 4. Hoyer sparsity
        l2_norm = np.sqrt(np.sum(abs_h ** 2))
        sqrt_n = np.sqrt(len(abs_h))
        hoyer = (sqrt_n - l1_norm / l2_norm) / (sqrt_n - 1) if l2_norm > 0 else 0
        
        # 5. Kurtosis
        kurtosis = stats.kurtosis(abs_h)
        
        # 6. Effective Rank
        squared = sample_h ** 2
        normalized = squared / squared.sum() if squared.sum() > 0 else squared
        entropy = -(normalized * np.log(normalized + 1e-10)).sum()
        effective_rank = np.exp(entropy) / len(sample_h)
        
        # 7. Participation Ratio
        sum_sq = (sample_h ** 2).sum()
        sum_fourth = (sample_h ** 4).sum()
        participation_ratio = (sum_sq ** 2) / sum_fourth / len(sample_h) if sum_fourth > 0 else 0
        
        # Store metrics
        if i == 0:
            for key in ['l1_norm', 'top1pct_energy', 'top5pct_energy', 'top10pct_energy',
                       'gini', 'hoyer', 'kurtosis', 'effective_rank', 'participation_ratio']:
                metrics[key] = []
        
        metrics['l1_norm'].append(l1_norm)
        metrics['top1pct_energy'].append(top1pct_energy)
        metrics['top5pct_energy'].append(top5pct_energy)
        metrics['top10pct_energy'].append(top10pct_energy)
        metrics['gini'].append(gini)
        metrics['hoyer'].append(hoyer)
        metrics['kurtosis'].append(kurtosis)
        metrics['effective_rank'].append(effective_rank)
        metrics['participation_ratio'].append(participation_ratio)
    
    # Calculate average
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics

def apply_chat_template(tokenizer, question_text: str, use_cot: bool = False) -> str:
    """应用chat模板格式化问题"""
    if use_cot:
        # CoT prompt
        system_content = "You are a helpful assistant. Answer multiple choice questions by selecting the correct option. Think step by step, then respond with ONLY the letter of your choice at the end."
        system_simple = "You are a helpful assistant. Answer multiple choice questions by selecting the correct option. Think step by step, then respond with ONLY the letter of your choice at the end."
    else:
        # Direct prompt
        system_content = "You are a helpful QA assistant. Answer multiple choice questions by selecting the correct option (A, B, C, D.....). Respond with ONLY the letter of your choice."
        system_simple = "You are a helpful assistant. Answer multiple choice questions by selecting the correct option. Respond with ONLY the letter of your choice."
    
    messages = [
        {
            "role": "system", 
            "content": system_content
        },
        {
            "role": "user", 
            "content": question_text
        }
    ]
    
    # 使用tokenizer的chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return formatted
    else:
        # 如果没有chat template，使用简单格式
        return f"System: {system_simple}\n\nUser: {question_text}\n\nAssistant:"

def generate_answer(tokenizer, model, prompt: str, use_cot: bool = False) -> str:
    """生成模型回答，只输出A-J选项"""
    model.eval()
    
    # 应用chat模板
    formatted_prompt = apply_chat_template(tokenizer, prompt, use_cot)
    
    inputs = tokenizer(formatted_prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    
    with torch.no_grad():
        # 生成回答，限制长度为1个token
        generated = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # 获取生成的新token
        generated_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 确保答案是A-J中的一个
        valid_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        if generated_text.upper() in valid_options:
            return generated_text.upper()
        else:
            # 如果不是标准答案格式，尝试提取第一个字母
            for char in generated_text.upper():
                if char in valid_options:
                    return char
            return 'A'  # 默认返回A

def get_last_hidden_state_for_input(tokenizer, model, text: str, use_cot: bool = False) -> torch.Tensor:
    model.eval()
    
    # 应用chat模板
    formatted_text = apply_chat_template(tokenizer, text, use_cot)
    
    inputs = tokenizer(formatted_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    
    # 构建完整的inputs字典
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_layer_hidden_state = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            last_layer_hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
            last_layer_hidden_state = outputs.decoder_hidden_states[-1]
        else:
            raise AttributeError(f"Cannot find hidden states in {type(outputs)}")
        
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

def format_category_name(category):
    """格式化类别名称，使其更易读"""
    # 将下划线替换为空格，并首字母大写
    return category.replace('_', ' ').title()


def create_combined_heatmaps(avg_results, selected_areas, model_short_name, use_cot=False):
    """创建五个稀疏性指标的组合热力图"""
    
    # 创建5个子图的组合热力图，使用更美观的布局
    fig, axes = plt.subplots(1, 5, figsize=(30, 7))
    
    # 设置整体样式
    plt.style.use('default')
    fig.patch.set_facecolor('white')
    
    noise_labels = ['Normal', 'Light Noise', 'Heavy Noise']
    metrics = [
        ('l1_norm', 'L1 Norm'),
        ('top5pct_energy', 'Top-5% Energy'), 
        ('top10pct_energy', 'Top-10% Energy'),
        ('effective_rank', 'Effective Rank'),
        ('hoyer', 'Hoyer Sparsity')
    ]
    
    # 为每个指标创建热力图
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        
        # 准备热力图数据
        heatmap_data = []
        area_labels = []
        
        for area in selected_areas:
            area_labels.append(area)
            row_data = []
            for noise_type in ['normal', 'light_noise', 'heavy_noise']:
                if metric in avg_results[area][noise_type]:
                    value = avg_results[area][noise_type][metric]
                    row_data.append(value)
                    # 调试：打印第一个area的数据
                    if idx == 0 and area == selected_areas[0]:
                        print(f"  DEBUG: {area} - {noise_type} - {metric}: {value:.4f}")
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
        
        heatmap_data = np.array(heatmap_data)
        
        # 调试：验证数据形状
        if idx == 0:
            print(f"\n=== 热力图数据形状验证 ===")
            print(f"  数据形状: {heatmap_data.shape} (应该是 {len(area_labels)} 个领域 × 3 个噪声级别)")
            print(f"  X轴标签顺序: {noise_labels}")
            print(f"  数据列顺序: normal(0), light_noise(1), heavy_noise(2)")
            print(f"  第一行数据 ({area_labels[0]}): {heatmap_data[0]}")
        
        # 创建热力图，使用沉稳的颜色方案
        im = ax.imshow(heatmap_data, cmap='Blues', aspect='auto', interpolation='nearest')
        
        # 设置刻度和标签，加粗字体
        ax.set_xticks(range(len(noise_labels)))
        ax.set_yticks(range(len(area_labels)))
        ax.set_xticklabels(noise_labels, fontsize=14, fontweight='bold')
        
        # 只在第一个子图显示y轴标签
        if idx == 0:
            ax.set_yticklabels(area_labels, fontsize=13, fontweight='bold')
        else:
            ax.set_yticklabels([])
        
        # 设置标题，更大更粗的字体 - 减小pad让标题往下移
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        
        # 添加数值标注，更大的字体
        for i in range(len(area_labels)):
            for j in range(len(noise_labels)):
                value = heatmap_data[i, j]
                # 根据颜色方案调整文字颜色
                text_color = 'white' if value > np.percentile(heatmap_data, 60) else 'black'
                ax.text(j, i, f'{value:.3f}',
                       ha="center", va="center", color=text_color, 
                       fontsize=12, fontweight='bold')
        
        # 添加颜色条，调整样式
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.025)
        cbar.ax.tick_params(labelsize=11, width=1.5)
        cbar.outline.set_linewidth(1.5)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
    
    # 调整子图间距，使布局更美观
    plt.tight_layout(pad=3.0, w_pad=2.0)
    
    # 添加整体标题
    fig.suptitle('Sparsity Metrics Analysis by Academic Area and Adversarial Noise Level', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Save as PDF
    prefix = "cot_" if use_cot else ""
    combined_output_path = f'./{prefix}combined_heatmaps_{model_short_name}.pdf'
    plt.savefig(combined_output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    print(f"Combined heatmap saved: {combined_output_path}")
    
    plt.close()
    return combined_output_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze MMLU-Pro with adversarial noise across academic areas')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help='Model name or path')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples per area (default: 20)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--use_cot', default=False, action='store_true',
                        help='Use Chain-of-Thought prompting instead of direct prompting (default: False)')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face token (optional)')
    args = parser.parse_args()
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    
    model_name = args.model_name
    num_samples_per_area = args.num_samples
    use_cot = args.use_cot
    
    # 设置GPU
    if args.gpu_id is not None:
        device_map = {"": args.gpu_id}
        print(f"使用GPU {args.gpu_id}")
    else:
        device_map = "auto"
        print("使用自动设备分配")
    
    print(f"加载模型: {model_name}")
    print(f"每个领域样本数: {num_samples_per_area}")
    print(f"Prompt类型: {'CoT (Chain-of-Thought)' if use_cot else 'Direct'}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map
    )
    print(f"模型加载完成: {model_name}")
    
    # 设置pad_token如果不存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载MMLU-Pro数据集
    print("Loading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    validation_data = ds['test']
    
    print(f"Total validation samples: {len(validation_data)}")
    
    # 分析数据集结构
    categories = list(set(validation_data['category']))
    print(f"Total categories: {len(categories)}")
    
    # 按领域组织数据（噪声级别将作为难度级别）
    area_data = defaultdict(list)
    
    # 处理所有样本，按领域分类
    print("按领域组织样本...")
    for sample in validation_data:
        category = sample['category']
        area = format_category_name(category)  # 直接使用数据集中的类别名称
        area_data[area].append(sample)
    
    # 显示领域统计
    print("\n领域样本分布:")
    print("="*60)
    for area in sorted(area_data.keys()):
        count = len(area_data[area])
        print(f"  {area}: {count} samples")
    
    # 选择要分析的领域（样本数量足够的）
    selected_areas = []
    for area in area_data.keys():
        total_samples = len(area_data[area])
        if total_samples >= 30:  # 降低门槛以包含更多领域
            selected_areas.append(area)
    
    # 确保包含主要学科领域
    priority_areas = ['Math', 'Physics', 'Chemistry', 'Biology', 'Computer Science', 'Psychology', 'Economics', 'Business']
    for priority_area in priority_areas:
        if priority_area in area_data and priority_area not in selected_areas:
            total_samples = len(area_data[priority_area])
            if total_samples >= 10:  # 更低的门槛用于优先领域
                selected_areas.append(priority_area)
    
    print(f"\n选择的领域进行分析: {selected_areas}")
    
    # 存储所有结果
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    accuracy_results = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    
    # 对每个选定领域进行分析 - 对所有问题应用不同噪声级别作为难度
    for area in selected_areas:
        print(f"\n### 分析领域: {area} ###")
        print("="*80)
        
        samples = area_data[area]
        
        if len(samples) == 0:
            continue
            
        # 限制样本数量 - 使用命令行参数指定
        sample_size = min(num_samples_per_area, len(samples))
        selected_samples = samples[:sample_size]
        
        print(f"分析 {area}: {sample_size} samples")
        
        for idx, sample in enumerate(selected_samples):
            question = sample['question']
            options = sample['options']
            correct_answer = sample['answer']
            
            # 找到正确答案的索引
            if len(correct_answer) == 1 and correct_answer.upper() in 'ABCDEFGHIJ':
                correct_answer_idx = ord(correct_answer.upper()) - ord('A')
            else:
                correct_answer_idx = sample.get('answer_index', 0)
            
            print(f"\nSample {idx+1}/{sample_size}")
            print(f"Question: {question[:100]}...")
            
            # 创建不同噪声级别的变体
            contexts = create_context_variants(question, options, correct_answer_idx, sample['category'])
            
            # 打印每个样本的所有难度级别的完整文本
            print("\n" + "="*100)
            print(f"【{area}】样本 {idx+1}/{sample_size} - 各难度级别的完整文本")
            print("="*100)
            for noise_name, context_text in [('Normal', contexts['normal']), 
                                               ('Light Noise', contexts['light_noise']), 
                                               ('Heavy Noise', contexts['heavy_noise'])]:
                print(f"\n{'*'*50}")
                print(f"【{noise_name}】")
                print(f"{'*'*50}")
                print(context_text)
                print(f"{'*'*50}")
                print(f"文本长度: {len(context_text)} 字符")
            print("="*100 + "\n")
            
            context_names = ['Normal', 'Light Noise', 'Heavy Noise']
            sparsity_keys = ['normal', 'light_noise', 'heavy_noise']
            
            for context_key, name, key in zip(['normal', 'light_noise', 'heavy_noise'], context_names, sparsity_keys):
                prompt = contexts[context_key]
                
                # 调试：打印当前处理的噪声级别
                if idx == 0 and area == selected_areas[0]:
                    print(f"    Processing: {name} (key={key})")
                
                # 获取隐藏状态并计算稀疏性指标
                hidden_state = get_last_hidden_state_for_input(tokenizer, model, prompt, use_cot)
                sparsity = compute_sparsity_metrics(hidden_state)
                all_results[area][key]['sparsity'].append(sparsity)
                
                # 调试：打印稀疏性值
                if idx == 0 and area == selected_areas[0]:
                    print(f"      L1 norm: {sparsity['l1_norm']:.4f}, EffRank: {sparsity['effective_rank']:.4f}")
                
                # 只在非 CoT 模式下生成答案并统计准确率
                if not use_cot:
                    predicted_answer = generate_answer(tokenizer, model, prompt, use_cot)
                    is_correct = predicted_answer == correct_answer
                    
                    # 更新准确率统计
                    accuracy_results[area][key]['total'] += 1
                    if is_correct:
                        accuracy_results[area][key]['correct'] += 1
                    
                    print(f"{name}: {predicted_answer} ({'✓' if is_correct else '✗'})")
                else:
                    print(f"{name}: representation extracted (CoT mode)")
            
            print("-" * 60)
    
    # 计算平均指标
    print("\n计算平均指标...")
    avg_results = defaultdict(lambda: defaultdict(dict))
    
    for area in selected_areas:
        for noise_type in ['normal', 'light_noise', 'heavy_noise']:
            if len(all_results[area][noise_type]['sparsity']) > 0:
                sparsity_data = all_results[area][noise_type]['sparsity']
                
                # 计算每个指标的平均值，确保包含所有需要的指标
                for metric in ['l1_norm', 'top5pct_energy', 'top10pct_energy', 'effective_rank', 'hoyer']:
                    values = [d[metric] for d in sparsity_data]
                    avg_results[area][noise_type][metric] = np.mean(values)
                
                # 只在非 CoT 模式下计算准确率
                if not use_cot:
                    correct = accuracy_results[area][noise_type]['correct']
                    total = accuracy_results[area][noise_type]['total']
                    avg_results[area][noise_type]['accuracy'] = correct / total * 100 if total > 0 else 0
    
    # 打印所有领域的平均结果
    print(f"\n=== 跨领域平均结果 ===")
    for noise_type in ['normal', 'light_noise', 'heavy_noise']:
        l1_values = []
        effrank_values = []
        hoyer_values = []
        acc_values = []
        top5_values = []
        top10_values = []
        
        for area in selected_areas:
            if noise_type in avg_results[area]:
                l1_values.append(avg_results[area][noise_type].get('l1_norm', 0))
                effrank_values.append(avg_results[area][noise_type].get('effective_rank', 0))
                hoyer_values.append(avg_results[area][noise_type].get('hoyer', 0))
                if not use_cot:
                    acc_values.append(avg_results[area][noise_type].get('accuracy', 0))
                top5_values.append(avg_results[area][noise_type].get('top5pct_energy', 0))
                top10_values.append(avg_results[area][noise_type].get('top10pct_energy', 0))
        
        if l1_values:
            avg_l1 = sum(l1_values) / len(l1_values)
            avg_effrank = sum(effrank_values) / len(effrank_values)
            avg_hoyer = sum(hoyer_values) / len(hoyer_values)
            avg_top5 = sum(top5_values) / len(top5_values)
            avg_top10 = sum(top10_values) / len(top10_values)
            
            if use_cot:
                # CoT 模式：不显示准确率
                print(f"  {noise_type:15s}: L1={avg_l1:.4f}, EffRank={avg_effrank:.4f}, Hoyer={avg_hoyer:.4f}, Top5%={avg_top5:.4f}, Top10%={avg_top10:.4f}")
            else:
                # Direct 模式：显示准确率
                avg_acc = sum(acc_values) / len(acc_values)
                print(f"  {noise_type:15s}: L1={avg_l1:.4f}, EffRank={avg_effrank:.4f}, Hoyer={avg_hoyer:.4f}, Top5%={avg_top5:.4f}, Top10%={avg_top10:.4f}, Acc={avg_acc:.1f}%")
    
    # 生成热力图
    print("\n生成热力图...")
    model_short_name = model_name.split('/')[-1].replace('-', '_')
    
    # 生成组合热力图
    create_combined_heatmaps(avg_results, selected_areas, model_short_name, use_cot)
    print("分析完成！")

def add_adversarial_noise_to_content(text, noise_level="light"):
    """只对问题内容添加对抗噪声，保护选项标识符"""

    
    # 分离问题内容和选项部分
    lines = text.split('\n')
    question_lines = []
    option_lines = []
    answer_line = ""
    
    in_options = False
    for line in lines:
        stripped = line.strip()
        # 检测选项行 (A) (B) (C) 等
        if stripped.startswith('(') and len(stripped) > 2 and stripped[1] in 'ABCDEFGHIJ' and stripped[2] == ')':
            in_options = True
            option_lines.append(line)
        elif stripped == "Answer:":
            answer_line = line
        elif not in_options:
            question_lines.append(line)
        else:
            option_lines.append(line)
    
    # 只对问题内容应用噪声
    question_text = '\n'.join(question_lines)
    noisy_question = apply_noise_to_text(question_text, noise_level)
    
    # 重新组合，保持选项不变
    result_lines = [noisy_question] if noisy_question.strip() else []
    result_lines.extend(option_lines)
    if answer_line:
        result_lines.append(answer_line)
    
    return '\n'.join(result_lines)

def apply_char_level_noise(text, noise_rate=0.10):
    """字符级别的噪声 - 包含替换、交换、删除、数字化（不添加垃圾文本）"""
    import random
    char_substitutions = {
        'a': ['@', 'α'], 'b': ['β'], 'c': ['/'], 'd': ['δ'], 'e': ['3', 'ε'], 'f': ['φ'], 'g': ['γ'], 'h': ['η'], 'i': ['1', '!'], 
        'j': ['ϳ'], 'k': ['κ'], 'l': ['1', '|'], 'm': ['μ'], 'n': ['η'], 'o': ['0', 'ο'], 'p': ['ρ'], 'q': ['ϙ'], 'r': ['я'], 's': ['$'], 't': ['7'], 'u': ['υ'], 'v': ['ϑ'], 'w': ['ω'], 'x': ['×'], 'y': ['γ'], 'z': ['ζ']
    }
    
    # 第一步：字符替换
    noisy_text = ""
    for char in text:
        if char.lower() in char_substitutions and random.random() < noise_rate:
            replacement = random.choice(char_substitutions[char.lower()])
            noisy_text += replacement.upper() if char.isupper() else replacement
        else:
            noisy_text += char
    
    # 第二步：单词级别的破坏（交换、删除、数字化）
    words = noisy_text.split()
    processed_words = []
    
    for word in words:
        if len(word) > 2 and random.random() < noise_rate:
            # 随机选择一种破坏方式
            corruption_type = random.choice(['swap', 'delete', 'digitize'])
            
            if corruption_type == 'swap' and len(word) > 3:
                # 交换两个相邻字符
                pos = random.randint(0, len(word) - 2)
                word_list = list(word)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                word = ''.join(word_list)
            
            elif corruption_type == 'delete' and len(word) > 3:
                # 删除一个字符
                pos = random.randint(1, len(word) - 2)  # 不删除首尾
                word = word[:pos] + word[pos+1:]
            
            elif corruption_type == 'digitize':
                # 替换为随机数字（666, 888, 999等）
                digit_replacements = ['666', '888', '999', '123', '777']
                # 随机替换单词中的一个字符为数字串
                if len(word) > 2:
                    pos = random.randint(1, len(word) - 1)
                    digit = random.choice(digit_replacements)
                    word = word[:pos] + digit + word[pos+1:]
        
        processed_words.append(word)
    
    # 不再添加垃圾文本，直接返回
    return ' '.join(processed_words)



def apply_noise_to_text(text, noise_level):
    """对纯文本内容应用对抗噪声"""
    if noise_level == "normal":
        # Normal：完全干净，无噪声
        return text
    elif noise_level == "light":
        # Light：10%字符噪声（字符替换、交换、删除、数字化）
        return apply_char_level_noise(text, noise_rate=0.10)
    
    return text

def create_context_variants(question, options, correct_answer_idx, category):
    """为MMLU问题创建三种噪声级别的变体"""
    import random
    
    # 基础问题格式 - 使用简单的format函数
    def simple_format_mmlu_question(question, options):
        formatted = f"{question}\n\n"
        for i, option in enumerate(options):
            formatted += f"({chr(65+i)}) {option}\n"
        formatted += "\nAnswer:"
        return formatted
    
    # 创建修改版选项的通用函数
    def create_modified_options(original_options, num_to_add):
        """创建修改版选项：原始选项 + 指定数量的修改版本"""
        modified_options = list(original_options)  # 先添加所有原始选项
        
        # 从原始选项中随机选择要修改的选项
        indices_to_modify = random.sample(range(len(original_options)), min(num_to_add, len(original_options)))
        
        for idx in indices_to_modify:
            opt = original_options[idx]
            modified_opt = opt
            
            # 随机选择一种修改方式
            modification_type = random.choice(['add_word', 'change_word', 'add_punctuation', 'repeat_word'])
            
            if modification_type == 'add_word':
                # 添加一个干扰词
                distractor_words = ['also', 'maybe', 'possibly', 'or', 'approximately', 'nearly', 'about', 'typically', 'generally']
                word = random.choice(distractor_words)
                words = modified_opt.split()
                if len(words) > 0:
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, word)
                    modified_opt = ' '.join(words)
            
            elif modification_type == 'change_word':
                # 改变一个词
                words = modified_opt.split()
                if len(words) > 1:
                    pos = random.randint(0, len(words) - 1)
                    # 简单的同义词替换
                    replacements = {
                        'the': 'a', 'is': 'was', 'are': 'were', 'in': 'at', 
                        'on': 'upon', 'for': 'to', 'with': 'by', 'an': 'the'
                    }
                    if words[pos].lower() in replacements:
                        words[pos] = replacements[words[pos].lower()]
                    modified_opt = ' '.join(words)
            
            elif modification_type == 'add_punctuation':
                # 添加标点符号
                punctuations = [',', ';', '...']
                punct = random.choice(punctuations)
                words = modified_opt.split()
                if len(words) > 1:
                    insert_pos = random.randint(1, len(words))
                    words.insert(insert_pos, punct)
                    modified_opt = ' '.join(words)
            
            elif modification_type == 'repeat_word':
                # 重复一个词
                words = modified_opt.split()
                if len(words) > 0:
                    pos = random.randint(0, len(words) - 1)
                    words.insert(pos + 1, words[pos])
                    modified_opt = ' '.join(words)
            
            modified_options.append(modified_opt)
        
        return modified_options
    
    # 1. Normal - 10个原始选项，无噪声
    base_question = simple_format_mmlu_question(question, options)
    normal_question = add_adversarial_noise_to_content(base_question, "normal")
    
    # 2. Light - 15个选项（原10个 + 5个修改版），10%字符噪声
    light_options = create_modified_options(options, 5)
    light_base_question = simple_format_mmlu_question(question, light_options)
    light_noise_question = add_adversarial_noise_to_content(light_base_question, "light")
    
    # 3. Heavy - 20个选项（原10个 + 10个修改版），10%字符噪声
    heavy_options = create_modified_options(options, 10)
    heavy_base_question = simple_format_mmlu_question(question, heavy_options)
    heavy_noise_question = add_adversarial_noise_to_content(heavy_base_question, "light")
    
    return {
        'normal': normal_question,
        'light_noise': light_noise_question,
        'heavy_noise': heavy_noise_question
    }

if __name__ == "__main__":
    main()
