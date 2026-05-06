"""
Learning Dynamics Experiment:
Track top5% energy and accuracy as a function of training epoch/step.
Periodically evaluates on ID/OOD splits during training.
"""

import os
import json
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback

from pretrain import (
    LatentRuleGraph, TrainDataset, EvalDataset, CharTokenizer,
    DataCollatorForSupervisedDataset, eval_simple, set_seed,
    count_params, compute_llama_param
)


class LearningDynamicsCallback(TrainerCallback):
    """Callback that evaluates sparsity metrics at regular intervals during training."""

    def __init__(self, graph, tokenizer, eval_steps, device, num_test=200):
        self.graph = graph
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.device = device
        self.num_test = num_test
        self.results = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.eval_steps != 0 and step != 1:
            return

        print(f"\n[Step {step}] Evaluating sparsity metrics...")

        eval_dataset_easy = EvalDataset(self.graph, self.tokenizer, split="id", num_options=10)
        eval_dataset_medium = EvalDataset(self.graph, self.tokenizer, split="ood_medium", num_options=10)
        eval_dataset_hard = EvalDataset(self.graph, self.tokenizer, split="ood_hard", num_options=10)

        record = {"step": step}

        for name, ds in [("id", eval_dataset_easy),
                         ("ood_medium", eval_dataset_medium),
                         ("ood_hard", eval_dataset_hard)]:
            acc, loss, sp = eval_simple(
                ds, model, self.tokenizer,
                batch_size=32, max_length=64,
                num_test=self.num_test, device=self.device,
                verbose=False, normalize_by_len=True,
                compute_sparsity=True
            )
            record[f"{name}_acc"] = acc
            record[f"{name}_loss"] = loss
            if sp:
                record[f"{name}_top5pct_energy"] = sp["top5pct_energy"]
                record[f"{name}_top10pct_energy"] = sp["top10pct_energy"]
                record[f"{name}_l1_norm"] = sp["l1_norm"]
                record[f"{name}_effective_rank"] = sp["effective_rank"]

        self.results.append(record)
        print(f"  ID: acc={record.get('id_acc', 0):.2%}, top5%={record.get('id_top5pct_energy', 0):.4f}")
        print(f"  OOD-Medium: acc={record.get('ood_medium_acc', 0):.2%}, top5%={record.get('ood_medium_top5pct_energy', 0):.4f}")
        print(f"  OOD-Hard: acc={record.get('ood_hard_acc', 0):.2%}, top5%={record.get('ood_hard_top5pct_energy', 0):.4f}")

        # Put model back in train mode
        model.train()


def plot_learning_dynamics(results, output_path, model_name):
    """Plot top5% energy and accuracy vs training step."""

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
        'lines.linewidth': 1.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.usetex': False,
    })

    steps = [r["step"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    colors = {'id': '#2171b5', 'ood_medium': '#238b45', 'ood_hard': '#d94801'}
    labels = {'id': 'ID (Easy)', 'ood_medium': 'OOD (Medium)', 'ood_hard': 'OOD (Hard)'}

    # (a) Top5% Energy vs Step
    ax = axes[0, 0]
    for split in ['id', 'ood_medium', 'ood_hard']:
        key = f"{split}_top5pct_energy"
        vals = [r.get(key, np.nan) for r in results]
        ax.plot(steps, vals, color=colors[split], label=labels[split], marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Top-5% Energy Ratio')
    ax.set_title('(a) Top-5% Energy vs Training Step')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Top10% Energy vs Step
    ax = axes[0, 1]
    for split in ['id', 'ood_medium', 'ood_hard']:
        key = f"{split}_top10pct_energy"
        vals = [r.get(key, np.nan) for r in results]
        ax.plot(steps, vals, color=colors[split], label=labels[split], marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Top-10% Energy Ratio')
    ax.set_title('(b) Top-10% Energy vs Training Step')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Accuracy vs Step
    ax = axes[1, 0]
    for split in ['id', 'ood_medium', 'ood_hard']:
        key = f"{split}_acc"
        vals = [r.get(key, np.nan) for r in results]
        ax.plot(steps, vals, color=colors[split], label=labels[split], marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('(c) Accuracy vs Training Step')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) Effective Rank vs Step
    ax = axes[1, 1]
    for split in ['id', 'ood_medium', 'ood_hard']:
        key = f"{split}_effective_rank"
        vals = [r.get(key, np.nan) for r in results]
        ax.plot(steps, vals, color=colors[split], label=labels[split], marker='o', markersize=3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Effective Rank (normalized)')
    ax.set_title('(d) Effective Rank vs Training Step')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(f'Learning Dynamics: {model_name}', fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_size', type=str, default='llama-16-16')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--steps', type=int, default=2500, help='Total training steps')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluate every N steps')
    parser.add_argument('--num_test', type=int, default=200, help='Number of test samples per eval')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./dynamics_results')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Learning Dynamics Experiment ===")
    print(f"Model: {args.llm_size}, Steps: {args.steps}, Eval every: {args.eval_steps}")
    print(f"Device: {device}, Seed: {args.seed}")

    # Build graph
    set_seed(args.seed)
    graph = LatentRuleGraph(
        n=4000, n_r=50, n_triples=10000, n_rules=20,
        L_min=2, L_max=5, power_law=True,
        deductible_ratio=0.5, length_weighted=True,
        m=6, num_test=1000, temperature=0.25, mcmc=1.0,
        seed=args.seed
    )

    # Initialize model
    set_seed(42)
    model_name, l, h = args.llm_size.split('-')
    l, h = int(l), int(h)
    d = 64 * h

    config = transformers.LlamaConfig(
        hidden_size=d, intermediate_size=2*d,
        num_attention_heads=h, num_hidden_layers=l
    )
    tokenizer = CharTokenizer()
    config.vocab_size = len(tokenizer.vocab)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    print(f"Model params: {count_params(model):,}")

    # Dataset
    train_dataset = TrainDataset(
        graph, tokenizer=tokenizer,
        seq_length=128, num_of_sequences=1024, chars_per_token=3.6
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Callback for periodic evaluation
    dynamics_callback = LearningDynamicsCallback(
        graph=graph, tokenizer=tokenizer,
        eval_steps=args.eval_steps, device=device,
        num_test=args.num_test
    )

    # Training
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')

    train_args = TrainingArguments(
        bf16=True, max_steps=args.steps,
        per_device_train_batch_size=32, eval_strategy="no",
        save_steps=args.steps, save_total_limit=1,
        learning_rate=1e-4, weight_decay=0.0,
        warmup_ratio=0.2, lr_scheduler_type="cosine",
        logging_steps=10, output_dir=checkpoint_dir,
        report_to="none"
    )

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=train_args,
        train_dataset=train_dataset, data_collator=data_collator,
        callbacks=[dynamics_callback]
    )

    trainer.train()

    # Save results
    results = dynamics_callback.results
    results_path = os.path.join(args.output_dir, f'dynamics_{args.llm_size}_{args.steps}steps.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o))
    print(f"Saved results: {results_path}")

    # Plot
    fig_path = os.path.join(args.output_dir, f'dynamics_{args.llm_size}_{args.steps}steps.pdf')
    plot_learning_dynamics(results, fig_path, args.llm_size)

    # Cleanup checkpoint
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Cleaned up checkpoints: {checkpoint_dir}")

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    if results:
        first = results[0]
        last = results[-1]
        for split in ['id', 'ood_medium', 'ood_hard']:
            key_acc = f"{split}_acc"
            key_sp = f"{split}_top5pct_energy"
            if key_acc in first and key_acc in last:
                print(f"  {split}: acc {first[key_acc]:.2%} -> {last[key_acc]:.2%}, "
                      f"top5% {first.get(key_sp, 0):.4f} -> {last.get(key_sp, 0):.4f}")


if __name__ == "__main__":
    main()
