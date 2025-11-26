# Synthetic KG Pretraining

This module builds a synthetic knowledge-graph environment, trains a small Llama-like model from scratch, and evaluates recall/generalization across difficulty splits.

## Key Components

1. `LatentRuleGraph`: Generates acyclic logic rules and constructs a directed multigraph with deductible (rule-derived) and atomic edges.
2. `TrainDataset`: Streams triples from the generated graph to feed the language model.
3. `train(...)`: Randomly initializes a Llama-style model (config from `args.llm_size`) and runs SFT with HF `Trainer`.
4. `EvalDataset` + `eval_simple(...)`: Build MCQ-style queries for ID / OOD-medium / OOD-hard splits and measure sparsity & accuracy.

## Usage

```bash
cd pretrain
conda create -n kg-pretrain python=3.10 -y
conda activate kg-pretrain
pip install -r requirements.txt   # torch, transformers, datasets, networkx, matplotlib, numpy

python pretrain.py \
  --llm_size llama-32-32 \
  --steps 2500 \
  --gpu_id 0 \
  --seed 42
```

- `--llm_size` is parsed as `<family>-<layers>-<heads>` (e.g., `llama-32-32`). Hidden dim is `64 * heads`.
- `--steps` controls training iterations.
- `--seed` fixes the KG generation as well as model initialization.

## Outputs

- Saves the trained checkpoint, HF Trainer logs, and evaluation summaries.
- Prints accuracy/sparsity for Easy (ID), Medium (OOD-long), Hard (OOD-short) splits.
- Deletes the output dir at the end (see final section) after reporting parameter counts. Comment out the `os.system("rm -rf ...")` line if you want to keep checkpoints.

## Tips

- Adjust `n`, `n_rules`, or `deductible_ratio` inside `LatentRuleGraph` to control graph size/difficulty.
- `TrainDataset` samples triples uniformly at random; increase `num_of_sequences` or `seq_length` for longer contexts.
- `EvalDataset` mixes options by negative sampling. `is_correct`/accuracy is computed via cross-entropy ranking.

Check `pretrain.py` for additional arguments and customization hooks.

