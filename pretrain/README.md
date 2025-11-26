# Synthetic KG Pretraining

This module builds a synthetic knowledge-graph environment, trains a small Llama-like model from scratch, and evaluates recall/generalization across difficulty splits. 
We follow the code created by **Do Larger Language Models Generalize Better? A Scaling Law for Implicit Reasoning at Pretraining Time** and **Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data?**

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

## Outputs

- Saves the trained checkpoint, HF Trainer logs, and evaluation summaries.
- Prints accuracy/sparsity for Easy (ID), Medium (OOD-long), Hard (OOD-short) splits.
- Deletes the output dir at the end (see final section) after reporting parameter counts. Comment out the `os.system("rm -rf ...")` line if you want to keep checkpoints.

## Tips

- Adjust `n`, `n_rules`, or `deductible_ratio` inside `LatentRuleGraph` to control graph size/difficulty.


Check `pretrain.py` for additional arguments and customization hooks.

## Cite
```
@inproceedings{yintransformers,
  title={Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data?},
  author={Yin, Yutong and Wang, Zhaoran},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

@article{wang2025larger,
  title={Do larger language models imply better reasoning? a pretraining scaling law for reasoning},
  author={Wang, Xinyi and Tan, Shawn and Jin, Mingyu and Wang, William Yang and Panda, Rameswar and Shen, Yikang},
  journal={arXiv preprint arXiv:2504.03635},
  year={2025}
}
```
