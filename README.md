# sklearn-autoresearch

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Give an AI agent a tabular dataset and let it experiment autonomously — trying different models, feature engineering, and hyperparameters. You come back to a log of experiments and (hopefully) a better model.

The original autoresearch lets agents optimize LLM training on a GPU. This fork adapts the same pattern for **sklearn supervised learning** on any tabular dataset — no GPU required.

## How it works

Three files:

- **`prepare.py`** — fixed data loading, train/val/test splits, and evaluation harness. Supports any CSV or sklearn built-in dataset. Auto-detects classification vs regression. **Not modified by the agent.**
- **`train.py`** — the single file the agent edits. Feature engineering, model selection, pipeline construction. Everything is fair game. **Edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. **Edited and iterated on by the human.**

The metric is auto-detected:
- **Classification**: `f1_weighted` (higher is better)
- **Regression**: `rmse` (lower is better)

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or
uv sync

# 2. Prepare a dataset (one-time)
python prepare.py --dataset california        # sklearn built-in
python prepare.py --csv mydata.csv --target y  # any CSV

# 3. Run a single training experiment
python train.py
```

## Available built-in datasets

| Name | Type | Description |
|------|------|-------------|
| `california` | Regression | California housing prices |
| `diabetes` | Regression | Diabetes progression |
| `iris` | Classification | Iris flower species |
| `wine` | Classification | Wine cultivar recognition |
| `breast_cancer` | Classification | Breast cancer diagnosis |
| `digits` | Classification | Handwritten digits 0-9 |

## Running the agent

Spin up Claude Code (or your agent of choice) in this repo, then prompt:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

## Project structure

```
prepare.py        — data prep + evaluation harness (do not modify)
train.py          — model, features, pipeline (agent modifies this)
program.md        — agent instructions
pyproject.toml    — dependencies
requirements.txt  — pip dependencies
```

## What the agent can do

- Swap between any sklearn estimator (RandomForest, GBM, SVM, LogReg, KNN, MLP, etc.)
- Engineer features (scaling, encoding, polynomial, PCA, selection, transforms)
- Tune hyperparameters
- Build ensembles (Voting, Stacking, Bagging)
- Restructure the pipeline

## What the agent cannot do

- Modify `prepare.py` or the evaluation metric
- Access the test set (held out for human evaluation)
- Install new packages

## License

MIT
