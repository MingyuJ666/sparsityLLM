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

python pretrain.py \
  --llm_size llama-32-32 \
  --steps 2500 \
  --gpu_id 0 \
  --seed 42
```

- `--llm_size` is parsed as `<family>-<layers>-<heads>` (e.g., `llama-32-32`). Hidden dim is `64 * heads`.
- `--steps` controls training iterations.
- `--seed` fixes the KG generation as well as model initialization.
- `--num_test` (default 100) sets the number of eval samples per difficulty split. It is capped at run time to the OOD-Long ("Medium") pool size — the smallest of the three splits — so all splits stay equal-sized. If you request more than the Medium pool holds, a `[warn]` is printed and the effective value is lowered.


 ```bash
# Default: llama-16-16, 5000 steps, eval every 200 steps
python learning_dynamics.py --steps 5000 --eval_steps 200
# Larger model
python learning_dynamics.py --llm_size llama-32-32 --steps 5000 --eval_steps 200
# Custom GPU and seed
python learning_dynamics.py --gpu_id 1 --seed 42 --steps 5000
 ```

Results are saved to `--output_dir`:
 - `dynamics_{model}_{steps}steps.json` - Per-step metrics including accuracy, loss, top-5%/top-10% energy ratio, and effective rank for each split (ID, OOD-
     Medium, OOD-Hard).
- `dynamics_{model}_{steps}steps.pdf` - 4-panel figure: (a) top-5% energy vs step, (b) top-10% energy vs step, (c) accuracy vs step, (d) effective rank vs step. Each
      panel shows all three splits.
  
The key finding is a U-shaped sparsity curve: representations first become sparse during early learning, then densify as accuracy saturates.

## Tips

- Adjust `n`, `n_rules`, or `deductible_ratio` inside `LatentRuleGraph` to control graph size/difficulty.


Check `pretrain.py` for additional arguments and customization hooks.


