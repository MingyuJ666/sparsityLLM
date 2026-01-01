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
from math_equivalence import get_answer, is_equiv

def parse_args():
    parser = argparse.ArgumentParser(description="Solve MATH-500 problems using vLLM")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model path or name (local vLLM model)")
    parser.add_argument("--gpu_id", type=int, default=2,
                        help="GPU ID")
    parser.add_argument("--test_ratio", type=float, default=0.8, help="Test set ratio")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                        help="GPU memory utilization")
    parser.add_argument("--use_cot", type=str, default='curriculum', choices=['cot', 'few-shot', 'curriculum', 'auto-shot'],
                        help="Whether to use Chain-of-Thought prompting")
    parser.add_argument("--num_few_shot", type=int, default=2,
                        help="Number of few-shot examples")
    parser.add_argument("--example_pool_size", type=int, default=500,
                        help="Example pool size (number of samples selected from training set)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Whether to print detailed response for each sample")
    parser.add_argument("--print_freq", type=int, default=100,
                        help="Frequency of printing detailed information (every N samples)")
    parser.add_argument("--rank_metric", type=str, default="l0_norm",
                        choices=["l0_norm", "top10pct_ratio", "effective_rank"],
                        help="Example pool ranking metric (for curriculum learning)")
    parser.add_argument("--dataset", type=str, default="math500",
                        choices=["math500"])
    parser.add_argument("--use_local_dataset", action="store_true", default=True,
                        help="Use local Math-500 dataset (./dataset/Math-500/)")
    parser.add_argument("--local_dataset_path", type=str, default="./dataset/Math-500",
                        help="Local dataset path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fixing few-shot sample selection, ensuring experiment reproducibility")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--sample_test_size", type=int, default=None,
                        help="Number of samples to randomly sample from test set (None means use full test set)")
    parser.add_argument("--n_levels", type=int, default=5,
                        help="Number of difficulty levels (for curriculum learning)")
    return parser.parse_args()





def prepare_prompt_math500(item, tokenizer, use_cot='zero-shot', example_pool=None, num_few_shot=1, dataset_type='math500', 
                           model_path=None, rank_metric=None, return_messages=False, difficulty_thresholds=None):
    # Select correct fields based on dataset type
   
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
        
        # Add the actual problem
        messages.append({"role": "user", "content": f"{problem}\n\nPlease solve this step by step and provide your final answer."})
        
    elif use_cot == 'curriculum':
        sparsity_score = item.get('sparsity_score', 0)
        difficulty_level, difficulty_level_name = get_difficulty_level(sparsity_score, difficulty_thresholds)
        
        top_k = min(20, len(example_pool))
        similar_examples = retrieve_topk(problem, example_pool, top_k=top_k)
        
        max_level = max(difficulty_thresholds.keys())
        
        # Automatically divide into three difficulty ranges based on max_level
        easy_threshold = max_level // 3                    # Upper bound of easy range
        medium_threshold = (max_level * 2) // 3            # Upper bound of medium range
        # Easy problems: difficulty_level <= easy_threshold
        # Medium problems: easy_threshold < difficulty_level <= medium_threshold
        # Hard problems: difficulty_level > medium_threshold
        
        if difficulty_level < easy_threshold:
            # Easy problems: only need 1 similar example at the same level and 1 at a higher level
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
            # Medium problems: 1 similar example at the same level and 1 at a higher level
            examples_found = []
            for score, rank_idx, example in similar_examples:
                if example.get('difficulty_level', 0) == max(1, difficulty_level):
                    examples_found.append(example)
                    if len(examples_found) >= 2:
                        break
            # Fill up to 2 examples
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
            # Hard problems (level 4-5): one level lower + same level, progressive learning
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
    
    # If using OpenAI API, return messages directly
    if return_messages:
        return messages, ground_truth, problem
    
    # Otherwise, format using tokenizer
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
        
    return formatted_prompt, ground_truth, problem

def load_local_math500(dataset_path, split='test'):
    """Load local Math-500 dataset"""
    file_path = os.path.join(dataset_path, f"{split}.jsonl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local dataset file does not exist: {file_path}")
    
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    
    # Convert to HuggingFace Dataset format for compatibility
    return Dataset.from_list(data_list)


def main():
    args = parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print(f"🚀 CUDA_VISIBLE_DEVICES set to {args.gpu_id}")
    

    
    # Load dataset
    if args.dataset == 'math500' and args.use_local_dataset:
        print(f"Loading local MATH-500 dataset ({args.local_dataset_path})...")

        train_data = load_local_math500(args.local_dataset_path, split='train')
        test_data = load_local_math500(args.local_dataset_path, split='test')
        print(f"Test set size: {len(test_data)}")
        
        if args.sample_test_size is not None:
            sample_size = min(args.sample_test_size, len(test_data))
            sample_indices = random.sample(range(len(test_data)), sample_size)
            test_data = test_data.select(sample_indices)
            print(f"Test set size after sampling: {len(test_data)}")
        
       
        example_pool_size = min(args.example_pool_size, len(train_data))
        example_pool = train_data.select(range(example_pool_size))

    if args.use_cot == 'curriculum':
        example_pool = rank_examples_by_sparsity(example_pool, args.model_path, metric=args.rank_metric)
        example_pool, difficulty_thresholds = analyze_difficulty_levels(example_pool, n_levels=args.n_levels, verbose=True)
        test_data = rank_examples_by_sparsity(test_data, args.model_path, metric=args.rank_metric)
        
        # Force clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory cleared, currently available: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB")
    else:
        difficulty_thresholds = None
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("\nLoading vLLM model...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype=torch.float16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_tokens
    )
    print("Model loading completed\n")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens
    )    

    
   
        
        
    # Prepare all prompts
    print("Preparing prompts...")
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
        
        # Display the first sample
        if i == 0:
            print("\n" + "="*80)
            print("Sample example (1st):")
            print("="*80)
            print(f"Problem: {problem[:200]}...")
            print(f"Answer: {ground_truth}")
            print("="*80 + "\n")
    
    print(f"Preparation completed, total {len(test_data)} samples\n")
    
    # Batch inference
    print("Starting batch inference...")
    outputs = llm.generate(prompts, sampling_params)
    print("Inference completed\n")
    
    # Extract generated text
    generated_texts = [output.outputs[0].text.strip() for output in outputs]
    
    # Process results
    print("="*80)
    print("Processing results")
    print("="*80)
    
    correct = 0
    total = 0
    
    for i, generated_text in enumerate(generated_texts):
        ground_truth = ground_truths[i]
        
        # Extract answer (for display)
        predicted_answer = get_answer(generated_text)
        
        # Check correctness (using is_equiv to compare full text)
        is_correct = is_equiv(generated_text, ground_truth)
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        
        # Print detailed information based on verbose and print_freq
        should_print = args.verbose or (i + 1) % args.print_freq == 0 or i == 0
        
        if should_print:
            print(f"\n{'='*80}")
            print(f"[Sample {i+1}/{len(test_data)}]")
            print(f"{'='*80}")
            print(f"Problem: {problems[i]}")
            print(f"\nModel response:\n{generated_text}")
            print(f"\nPredicted answer: {predicted_answer}")
            print(f"Ground truth: {ground_truth}")
            print(f"Result: {status} {'✓ Correct' if is_correct else '✗ Incorrect'}")
            print(f"Current accuracy: {correct/total:.2%} ({correct}/{total})")
            print(f"{'='*80}")
    
    # Final statistics
    final_accuracy = correct / total if total > 0 else 0
    
    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Final accuracy: {final_accuracy:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()
