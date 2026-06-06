# When Data Is Scarce: Scaling Sparse Language Models with Repeated Training

This repository contains the official code for the ICML2026  *"When Data Is Scarce: Scaling Sparse Language Models with Repeated Training"*.

We investigate how **Dynamic Sparse Training (DST)** interacts with **data repetition** (multiple epochs) in data-constrained pre-training regimes, and introduce **sparsity-aware scaling laws** to characterize this interaction.


## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Evaluation](#evaluation)
- [DST Configuration Guide](#dst-configuration-guide)
- [Analysis & Scaling Laws](#analysis--scaling-laws)
- [Model Configurations](#model-configurations)
- [Citation](#citation)

---

## Project Structure

```
├── torchrun_main.py                    # Main training script (DDP)
├── utils.py                            # Distributed training & logging helpers
├── requirements.txt
│
├── sparselearning/                     # Dynamic Sparse Training engine
│   ├── core.py                         # Masking: pruning, regrowth, topology evolution
│   ├── optimizer_new.py                # Custom Adam with momentum masking
│   └── decay.py                        # Decay schedules (cosine, linear, constant, WSD)
│
├── peft_pretraining/                   # Model definition & training utilities
│   ├── modeling_llama.py               # LLaMA (HuggingFace-based)
│   ├── dataloader.py                   # Streaming iterable dataset
│   ├── training_utils.py               # LR schedulers with warmup
│   └── args_utils.py                   # Argument validation
│
├── configs_new/                        # Model configs (20M to 3.84B params)
│   ├── llama_20m.json
│   ├── llama_40m.json
│   ├── llama_60m.json
│   ├── llama_120m.json
│   ├── llama_240m.json
│   ├── llama_480m.json
│   ├── llama_960m.json
│   ├── llama_1b92.json
│   └── llama_3b84.json
│
├── evals/                              # Evaluation
│   ├── downstream_tasks_evaluation.py  # lm-eval harness for standard benchmarks
│   └── evaluation_architectures.py     # TreeNet & FullEnsemble architectures
│
├── scripts/                            # SLURM training launchers
│   ├── train_llm_dst.sh
│   ├── train_llm_dense.sh
│   ├── submit_dst.sh                   # DST hyperparameter sweep
│   └── submit_dense.sh                 # Dense hyperparameter sweep
│
├── figures/                            # Analysis notebooks
│   ├── isoFLOP.ipynb
│   ├── isoLoss.ipynb
│   └── optimal_sparsity.ipynb
│
└── fitting/                            # Scaling law fitting
    └── fitting.ipynb
```

---

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

Requires PyTorch 2.1+ (CUDA 11.8), Transformers 4.38, HuggingFace Datasets, lm-eval 0.4.2.

### Data

The project uses the **C4** dataset. You can either:
1. **Stream directly** from HuggingFace (omit `--data_dir`).
2. **Use pre-tokenized Arrow files** for faster I/O (specify `--data_dir`).

---

## Training

### Quick Start (Single Node)

```bash
# DST training: 20M model, 93.75% sparse, 10K steps
torchrun --nproc_per_node=4 ./torchrun_main.py \
    --model_config "./configs_new/llama_20m.json" \
    --density 0.0625 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --epochs 1 \
    --lr 3.12e-2 \
    --growth random \
    --prune magnitude \
    --update_frequency 100 \
    --prune_rate 0.1 \
    --sparse_init uniform \
    --dtype bfloat16 \
    --run_name "dst_20m_s0.0625"
```

### DST with / without Data Repetition

```bash
bash scripts/train_llm_dst.sh --epoch 16   # 16 epochs (repeated data)
bash scripts/train_llm_dst.sh --epoch 1    # single pass (no repetition)
```

### Dense Baseline with / without Data Repetition

```bash
bash scripts/train_llm_dense.sh --epoch 16
bash scripts/train_llm_dense.sh --epoch 1
```

> The `--epoch` argument controls how many times the model iterates over the dataset, allowing you to vary the degree of data repetition.

### SLURM Cluster

```bash
# Submit a single DST job
sbatch scripts/train_llm_dst.sh <epochs> <training_steps> <model_size> <lr> <batch_size> <density>

# Example: 20M model, 10K steps, 6.25% density
sbatch scripts/train_llm_dst.sh 1 10000 20m 3.12e-2 128 0.0625

# Dense baseline
sbatch scripts/train_llm_dense.sh 1 10000 20m 3.12e-2 128
```

For hyperparameter sweeps:
```bash
bash scripts/submit_dst.sh
bash scripts/submit_dense.sh
```

### Key Training Arguments

| Argument | Description |
|---|---|
| `--model_config` | Path to model config JSON |
| `--batch_size` | Per-GPU batch size |
| `--total_batch_size` | Effective batch size across all GPUs and gradient accumulation |
| `--num_training_steps` | Number of optimizer update steps |
| `--max_train_tokens` | Alternative to steps (e.g., `100M`, `1B`) |
| `--epochs` | Number of epochs over the dataset |
| `--lr` | Peak learning rate |
| `--scheduler` | LR schedule: `cosine`, `linear`, `cosine_restarts` |
| `--warmup_steps` | LR warmup steps |
| `--dtype` | `bfloat16` (default if supported) or `float32` |
| `--activation_checkpointing` | Gradient checkpointing to save memory |
| `--eval_every` | Evaluation interval in update steps |
| `--wandb_used` | Enable Weights & Biases logging |

---

## DST Configuration Guide

The sparsity engine in [`sparselearning/core.py`](sparselearning/core.py) supports a wide range of dynamic sparse training algorithms.

### Core DST Arguments

| Argument | Options | Default | Description |
|---|---|---|---|
| `--density` | float (0,1] | 1.0 | Fraction of weights that are non-zero (1.0 = dense) |
| `--sparse_init` | `uniform`, `fixed_ERK`, `uniform_ratio` | `uniform` | Initial mask distribution |
| `--growth` | `random`, `gradient`, `momentum`, `momentum_neuron`, `gradient_acc` | `random` | Regrowth strategy |
| `--prune` | `magnitude`, `SET`, `threshold`, `magnitude_soft`, `global_magnitude` | `magnitude` | Pruning strategy |
| `--update_frequency` | int | 100 | Steps between topology updates |
| `--prune_rate` | float | 0.5 | Fraction of active weights to prune per update |
| `--density_decay` | `constant`, `cosine`, `linear` | `constant` | Density decay schedule (ramps from dense to sparse) |
| `--fix` | bool | False | Freeze topology (no pruning/regrowth) |
| `--reinit` | `no`, `zero`, `original` | `no` | Weight reinitialization after pruning |

### Supported Sparse Training Algorithms

| Algorithm | `--growth` | `--prune` | 
|---|---|---|---|
| **SET** | `random` | `magnitude` | 
| **RigL** | `gradient` | `magnitude` |
| **RigL (momentum)** | `momentum` | `magnitude` |
| **RigL (acc. grad.)** | `gradient_acc` | `magnitude` | 
| **Soft Pruning DST** | `random` | `magnitude_soft` | 

### Attention / MLP Ratio

The `--am_ratio` argument controls the relative sparsity between attention and MLP layers:
- `--am_ratio 1.0` (default): equal density in both
- `--am_ratio 2.0`: attention layers have 2× the density of MLP layers

### DST-specific Optimizer

When using DST with `--optimizer adamdst`, a custom optimizer in [`sparselearning/optimizer_new.py`](sparselearning/optimizer_new.py) is used. It extends PyTorch's Adam with support for:
- **Momentum masking**: zeroing out momentum buffers for pruned weights
- **Regrowth momentum reinitialization**: properly initializing Adam state for newly regrown connections
- **Decayed momentum initialization**: controlling the initial step count for regrown weights via `--op_decay_steps` and `--op_decay_max`

---

## Evaluation

### Perplexity (automatic during training)

Validation loss and perplexity are computed every `--eval_every` steps on C4 validation (10M tokens).

### Downstream Tasks

```bash
python evals/downstream_tasks_evaluation.py \
    --model_config "./configs_new/llama_20m.json" \
    --continue_from "./checkpoints/model_XXXX" \
    --batch_size 8
```

This uses the **lm-eval** library to evaluate on standard NLP benchmarks.


## Analysis & Scaling Laws

Jupyter notebooks for reproducing the paper's analysis:

| Notebook | Description |
|---|---|
| [`figures/isoFLOP.ipynb`](figures/isoFLOP.ipynb) | IsoFLOP curves: compare dense vs. sparse models at fixed compute budgets |
| [`figures/isoLoss.ipynb`](figures/isoLoss.ipynb) | IsoLoss contours: find compute-optimal configurations for a given loss target |
| [`figures/optimal_sparsity.ipynb`](figures/optimal_sparsity.ipynb) | Predict optimal sparsity as a function of data budget |
| [`fitting/fitting.ipynb`](fitting/fitting.ipynb) | Fit sparsity-aware scaling laws: \( L(N, U, D, S) \) regression |

These notebooks model the relationship between:
- **Model size** \(N\) — number of non-zero parameters
- **Unique Data size** \(U\) — number of unique tokens seen
- **Data size** \(D\) — number of training tokens
- **Sparsity** \(S\) — fraction of weights that are non-zero
- **Loss** \(L\) — validation cross-entropy loss



