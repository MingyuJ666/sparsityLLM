import os
import time
import re
import argparse
import random
import gc
import torch
from tqdm import tqdm

vars_to_remove = [
    'HF_HUB_USER_AGENT', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN',
    'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
    'ALL_PROXY', 'all_proxy'
]
for var in vars_to_remove:
    if var in os.environ:
        del os.environ[var]

from huggingface_hub import login
login(token="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")

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from utils.retrieve_similar_examples import retrieve_topk
from utils.rank import rank_examples_by_sparsity, analyze_difficulty_levels, get_difficulty_level

# 导入数学答案等价性判断模块
from math_equivalence import get_answer, is_equiv

def parse_args():
    parser = argparse.ArgumentParser(description="使用 vLLM 解决 MATH-500 问题")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="模型路径或名称（本地 vLLM 模型）")
    parser.add_argument("--gpu_id", type=int, default=2,
                        help="GPU 编号")
    parser.add_argument("--test_ratio", type=float, default=0.8, help="测试集比例")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                        help="GPU 显存利用率")
    parser.add_argument("--use_cot", type=str, default='curriculum', choices=['cot', 'few-shot', 'curriculum', 'auto-shot'],
                        help="是否使用 Chain-of-Thought prompting")
    parser.add_argument("--num_few_shot", type=int, default=2,
                        help="Few-shot 示例数量")
    parser.add_argument("--example_pool_size", type=int, default=500,
                        help="示例池大小（从训练集中选择的样本数量）")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="是否打印每个样本的详细回复")
    parser.add_argument("--print_freq", type=int, default=100,
                        help="打印详细信息的频率（每N个样本）")
    parser.add_argument("--rank_metric", type=str, default="l0_norm",
                        choices=["l0_norm", "top10pct_ratio", "effective_rank"],
                        help="示例池排序指标（用于 curriculum learning）")
    parser.add_argument("--dataset", type=str, default="math500",
                        choices=["math500"])
    parser.add_argument("--use_local_dataset", action="store_true", default=True,
                        help="使用本地 Math-500 数据集（./dataset/Math-500/）")
    parser.add_argument("--local_dataset_path", type=str, default="./dataset/Math-500",
                        help="本地数据集路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，用于固定 few-shot 样本选择，确保实验可复现")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="生成的最大 token 数量")
    parser.add_argument("--sample_test_size", type=int, default=None,
                        help="从测试集中随机抽样的样本数量（None 表示使用全部测试集）")
    parser.add_argument("--n_levels", type=int, default=5,
                        help="难度等级数量（用于 curriculum learning）")
    return parser.parse_args()





def prepare_prompt_math500(item, tokenizer, use_cot='zero-shot', example_pool=None, num_few_shot=1, dataset_type='math500', 
                           model_path=None, rank_metric=None, return_messages=False, difficulty_thresholds=None):
    # 根据数据集类型选择正确的字段
   
    problem = item.get('problem', item.get('question', ''))
    ground_truth = item.get('answer', item.get('solution', ''))
    level = item.get('level', 'N/A')

    if use_cot == 'cot':
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and provide your final answer in the end of the solution."},
            {"role": "user", "content": f"{problem}\n\n Please solve this step by step and provide your final answer."}
        ]
    if use_cot == 'auto-shot':
        num_auto_shot = num_few_shot
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and provide your final answer in the end of the solution."}
        ]
        
        similar_examples = retrieve_topk(problem, example_pool, top_k=num_auto_shot)
        for score, rank_idx, example in similar_examples:
            print(score, rank_idx, example)
            messages.append({"role": "user", "content": f"{example.get('problem', '')}\n\nPlease solve this step by step and provide your final answer."})
            messages.append({"role": "assistant", "content": example.get('solution', '')})
        messages.append({"role": "user", "content": f"{problem}\n\nPlease solve this step by step and provide your final answer."})
    
    elif use_cot == 'few-shot':
       
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and provide your final answer in the end of the solution."}
        ]
       
        num_examples = min(num_few_shot, len(example_pool))
        random_indices = random.sample(range(len(example_pool)), num_examples)
        
        for idx in random_indices:
            random_example = example_pool[idx]
            question_example = random_example.get("problem", "")
            answer_example = random_example.get("solution", "")
            messages.append({"role": "user", "content": f"{question_example}\n\nPlease solve this step by step and provide your final answer."})
            messages.append({"role": "assistant", "content": answer_example})
        
        # 添加实际问题
        messages.append({"role": "user", "content": f"{problem}\n\nPlease solve this step by step and provide your final answer."})
        
    elif use_cot == 'curriculum':
        sparsity_score = item.get('sparsity_score', 0)
        difficulty_level, difficulty_level_name = get_difficulty_level(sparsity_score, difficulty_thresholds)
        
        top_k = min(20, len(example_pool))
        similar_examples = retrieve_topk(problem, example_pool, top_k=top_k)
        
        max_level = max(difficulty_thresholds.keys())
        
        # 自动根据 max_level 划分三个难度区间
        easy_threshold = max_level // 3                    # 简单区间上限
        medium_threshold = (max_level * 2) // 3            # 中等区间上限
        # 简单问题：difficulty_level <= easy_threshold
        # 中等问题：easy_threshold < difficulty_level <= medium_threshold
        # 难问题：difficulty_level > medium_threshold
        
        if difficulty_level < easy_threshold:
            # 简单问题：只需要 1 个同级别的相似例子 和一个更高级的相似例子
            question1, answer1 = '', ''
            for score, rank_idx, example in similar_examples:
                if example.get('difficulty_level', 0) == difficulty_level:
                    question1 = example.get('problem', '')
                    answer1 = example.get('solution', '')
                    break
            question2, answer2 = '', ''
            for score, rank_idx, example in similar_examples:
                if example.get('difficulty_level', 0) == difficulty_level + 1:
                    question2 = example.get('problem', '')
                    answer2 = example.get('solution', '')
                    break
            
            
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and provide your final answer in the end of the solution."},
                {"role": "user", "content": f"{question1}\n\nPlease solve this problem step by step and provide your final answer."},
                {"role": "assistant", "content": answer1},
                {"role": "user", "content": f"{question2}\n\nPlease solve this problem step by step and provide your final answer."},
                {"role": "assistant", "content": answer2},
                {"role": "user", "content": f"{problem}\n\nPlease solve this problem step by step and provide your final answer."},
            ]
        
        elif difficulty_level < medium_threshold:
            # 中等问题：1 个同级别的相似例子 和一个更高级的相似例子
            examples_found = []
            for score, rank_idx, example in similar_examples:
                if example.get('difficulty_level', 0) == max(1, difficulty_level):
                    examples_found.append(example)
                    if len(examples_found) >= 2:
                        break
            # 补足到 2 个
            if len(examples_found) < 2:
                for score, rank_idx, example in similar_examples:
                    if example not in examples_found :
                        examples_found.append(example)
                        if len(examples_found) >= 2:
                            break
            
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and provide your final answer in the end of the solution."},
            ]
            for ex in examples_found:
                messages.append({"role": "user", "content": f"{ex.get('problem', '')}\n\nPlease solve this problem step by step and provide your final answer."})
                messages.append({"role": "assistant", "content": ex.get('solution', '')})
            messages.append({"role": "user", "content": f"{problem}\n\nPlease solve this problem step by step and provide your final answer."})
        
        else:
            # 难问题 (level 4-5)：低一级 + 同级别，循序渐进
            lower_level = max(1, difficulty_level - 1)
            
            question1, answer1 = '', ''
            for score, rank_idx, example in similar_examples:
                if example.get('difficulty_level', 0) == lower_level:
                    question1 = example.get('problem', '')
                    answer1 = example.get('solution', '')
                    break
            
            question2, answer2 = '', ''
            for score, rank_idx, example in similar_examples:
                if example.get('difficulty_level', 0) == difficulty_level:
                    if example.get('problem', '') != question1:
                        question2 = example.get('problem', '')
                        answer2 = example.get('solution', '')
                        break
            
            # Fallback
            if not question1 and similar_examples:
                question1 = similar_examples[0][2].get('problem', '')
                answer1 = similar_examples[0][2].get('solution', '')
            if not question2:
                for score, rank_idx, example in similar_examples:
                    if example.get('problem', '') != question1:
                        question2 = example.get('problem', '')
                        answer2 = example.get('solution', '')
                        break
            
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and provide your final answer in the end of the solution."},
                {"role": "user", "content": f"{question1}\n\nPlease solve this problem step by step and provide your final answer."},
                {"role": "assistant", "content": answer1},
                {"role": "user", "content": f"{question2}\n\nPlease solve this problem step by step and provide your final answer."},
                {"role": "assistant", "content": answer2},
                {"role": "user", "content": f"{problem}\n\nPlease solve this problem step by step and provide your final answer."},
            ]
    
    # 如果使用 OpenAI API，直接返回 messages
    if return_messages:
        return messages, ground_truth, problem
    
    # 否则使用 tokenizer 格式化
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
        
    return formatted_prompt, ground_truth, problem

def load_local_math500(dataset_path, split='test'):
    """加载本地 Math-500 数据集"""
    file_path = os.path.join(dataset_path, f"{split}.jsonl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"本地数据集文件不存在: {file_path}")
    
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    
    # 转换为 HuggingFace Dataset 格式以保持兼容性
    return Dataset.from_list(data_list)


def main():
    args = parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"🚀 已设置 CUDA_VISIBLE_DEVICES={args.gpu_id}")
    

    
    # 加载数据集
    if args.dataset == 'math500' and args.use_local_dataset:
        print(f"加载本地 MATH-500 数据集 ({args.local_dataset_path})...")

        train_data = load_local_math500(args.local_dataset_path, split='train')
        test_data = load_local_math500(args.local_dataset_path, split='test')
        print(f"测试集大小: {len(test_data)}")
        
        if args.sample_test_size is not None:
            sample_size = min(args.sample_test_size, len(test_data))
            sample_indices = random.sample(range(len(test_data)), sample_size)
            test_data = test_data.select(sample_indices)
            print(f"抽样后测试集大小: {len(test_data)}")
        
       
        example_pool_size = min(args.example_pool_size, len(train_data))
        example_pool = train_data.select(range(example_pool_size))

    if args.use_cot == 'curriculum':
        example_pool = rank_examples_by_sparsity(example_pool, args.model_path, metric=args.rank_metric)
        example_pool, difficulty_thresholds = analyze_difficulty_levels(example_pool, n_levels=args.n_levels, verbose=True)
        test_data = rank_examples_by_sparsity(test_data, args.model_path, metric=args.rank_metric)
        
        # 强制清理显存
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"显存已清理，当前可用: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB")
    else:
        difficulty_thresholds = None
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("\n加载 vLLM 模型...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype=torch.float16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_tokens
    )
    print("模型加载完成\n")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens
    )    

    
   
        
        
    # 准备所有 prompts
    print("准备 prompts...")
    prompts = []
    ground_truths = []
    problems = []
    
    for i in range(len(test_data)):
        item = test_data[i]
        
        formatted_prompt, ground_truth, problem = prepare_prompt_math500(
            item, tokenizer, use_cot=args.use_cot, example_pool=example_pool, 
            num_few_shot=args.num_few_shot, dataset_type=args.dataset, model_path=args.model_path, 
            rank_metric=args.rank_metric, difficulty_thresholds=difficulty_thresholds)

        prompts.append(formatted_prompt)
        
        ground_truths.append(ground_truth)
        problems.append(problem)
        
        # 显示第一个样本
        if i == 0:
            print("\n" + "="*80)
            print("样本示例（第1个）:")
            print("="*80)
            print(f"问题: {problem[:200]}...")
            print(f"答案: {ground_truth}")
            print("="*80 + "\n")
    
    print(f"准备完成，共 {len(test_data)} 个样本\n")
    
    # 批量推理
    print("开始批量推理...")
    outputs = llm.generate(prompts, sampling_params)
    print("推理完成\n")
    
    # 提取生成的文本
    generated_texts = [output.outputs[0].text.strip() for output in outputs]
    
    # 处理结果
    print("="*80)
    print("处理结果")
    print("="*80)
    
    correct = 0
    total = 0
    
    for i, generated_text in enumerate(generated_texts):
        ground_truth = ground_truths[i]
        
        # 提取答案（用于显示）
        predicted_answer = get_answer(generated_text)
        
        # 检查正确性（使用 is_equiv 比较完整文本）
        is_correct = is_equiv(generated_text, ground_truth)
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        
        # 根据verbose和print_freq打印详细信息
        should_print = args.verbose or (i + 1) % args.print_freq == 0 or i == 0
        
        if should_print:
            print(f"\n{'='*80}")
            print(f"[样本 {i+1}/{len(test_data)}]")
            print(f"{'='*80}")
            print(f"问题: {problems[i]}")
            print(f"\n模型回复:\n{generated_text}")
            print(f"\n预测答案: {predicted_answer}")
            print(f"正确答案: {ground_truth}")
            print(f"判断结果: {status} {'✓ 正确' if is_correct else '✗ 错误'}")
            print(f"当前准确率: {correct/total:.2%} ({correct}/{total})")
            print(f"{'='*80}")
    
    # 最终统计
    final_accuracy = correct / total if total > 0 else 0
    
    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"最终准确率: {final_accuracy:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()
