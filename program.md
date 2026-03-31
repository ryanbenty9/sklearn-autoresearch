# sklearn-autoresearch

This is an experiment to have an LLM do autonomous model selection and feature engineering on tabular data using sklearn.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar30`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, train/val/test splits, evaluation. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model selection, pipeline construction.
4. **Verify data exists**: Check that `~/.cache/sklearn-autoresearch/` contains data files and metadata.json. If not, tell the human to run `python prepare.py --dataset <name>` or `python prepare.py --csv <path> --target <col>`.
5. **Read the metadata**: `cat ~/.cache/sklearn-autoresearch/metadata.json` to understand the dataset — task type, features, number of classes, what metric you're optimizing.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs locally. You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - **Model selection**: swap between any sklearn estimator (RandomForest, GradientBoosting, SVM, LogisticRegression, KNN, AdaBoost, ExtraTrees, Ridge, Lasso, ElasticNet, MLP, etc.)
  - **Feature engineering**: scaling, encoding, polynomial features, interaction terms, binning, log transforms, feature selection (SelectKBest, mutual info, variance threshold), PCA, etc.
  - **Hyperparameter tuning**: change any model or preprocessor hyperparameters directly.
  - **Pipeline construction**: add, remove, or reorder pipeline steps.
  - **Ensembles**: VotingClassifier/Regressor, StackingClassifier/Regressor, BaggingClassifier/Regressor.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and train/val/test splits.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml` (sklearn, pandas, numpy, and their dependencies).
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.
- Access the test set. You train on train, evaluate on val. The test set is held out for final evaluation by the human.

**The goal is simple: optimize the primary metric on the validation set.**
- For classification: maximize `f1_weighted` (higher is better).
- For regression: minimize `rmse` (lower is better).

Check `metadata.json` to know which metric and direction applies to your dataset.

**Simplicity criterion**: All else being equal, simpler is better. A marginal improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so run `train.py` as-is.

## Output format

Once the script finishes it prints a summary like this (classification example):

```
---
f1_weighted:       0.953200
training_seconds:  1.2
accuracy:          0.947400
task_type:         classification
n_features:        30
n_train:           387
```

Or for regression:

```
---
rmse:              0.534200
training_seconds:  0.8
r2:                0.812300
mae:               0.412100
task_type:         regression
n_features:        8
n_train:           14448
```

You can extract the key metric from the log file:

```
grep "^f1_weighted:\|^rmse:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	score	status	description
```

1. git commit hash (short, 7 chars)
2. primary metric value achieved (e.g. 0.953200) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example (classification):

```
commit	score	status	description
a1b2c3d	0.953200	keep	baseline (GradientBoosting)
b2c3d4e	0.961400	keep	switch to RandomForest n_estimators=200
c3d4e5f	0.958100	discard	add polynomial features degree=2
d4e5f6g	0.000000	crash	SVM with rbf kernel (too slow)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar30`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Modify `train.py` with an experimental idea — change the model, tweak features, adjust hyperparameters.
3. git commit.
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^f1_weighted:\|^rmse:\|^accuracy:\|^r2:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on that idea.
7. Record the results in the TSV (NOTE: do not commit the results.tsv file, leave it untracked by git).
8. If the primary metric improved, you "advance" the branch, keeping the git commit.
9. If the primary metric is equal or worse, you `git reset` back to where you started.

**Strategy tips for sklearn experiments:**
- Start with simple models (LogisticRegression/Ridge) to establish a strong simple baseline before going complex.
- Try different model families before deep hyperparameter tuning on one model.
- Feature engineering often matters more than model choice — try scaling, encoding categoricals, interaction terms, log transforms on skewed features.
- For high-dimensional data, try feature selection (SelectKBest, mutual_info) or PCA.
- Ensemble methods (Voting, Stacking) can squeeze out the last few points once you've found good individual models.
- Watch for overfitting — if train score is much higher than val score, add regularization or simplify.

**Crashes**: If a run crashes, use your judgment. If it's a typo or missing import, fix and re-run. If the model is fundamentally broken for this data (e.g., SVM on 100K rows), skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be away and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try different model families, feature combinations, ensembles, creative feature engineering. The loop runs until the human interrupts you, period.
