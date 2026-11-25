# CE888: Offline Agentic Data Scientist

[![CI](https://github.com/tuna-b13/agentic-data-scientist/actions/workflows/ci.yml/badge.svg)](https://github.com/tuna-b13/agentic-data-scientist/actions/workflows/ci.yml)

**Developer:** Tuna Bozbay
**Module:** CE888 - Data Science and Decision Making
**Deliverable:** Final Project Code

---

## System Architecture

The system follows a modular agent-based architecture with three core agents and three tools, coordinated by a central orchestrator.

```
run_agent.py  -->  AgenticDataScientist (orchestrator)
                        |
           +------------+------------+
           |            |            |
       [Planner]   [Reflector]   [Memory]
           |            |            |
       +---+----+  +----+----+      |
       |        |  |         |      |
  [Profiler] [Modelling] [Evaluation]
```

### Orchestrator (`agentic_data_scientist.py`)
Central coordinator that runs the full pipeline: load -> profile -> plan -> train -> evaluate -> reflect -> (replan) -> report. All decisions are data-driven, not hardcoded. Includes structured logging (both file and console) and supports up to N replan iterations.

### Agents
- **Planner** (`agents/planner.py`): Returns a structured dict with `steps`, `rationale` (human-readable WHY for each decision), and `hints` (machine-consumable config for downstream tools). Selects from 5 plan templates (small, large, imbalanced, high_missing, default) based on dataset signals. Conditionally injects steps for skew transforms, outlier clipping, feature selection, and high-cardinality encoding. Uses memory from past runs to prioritise previously successful models.
- **Reflector** (`agents/reflector.py`): Analyses evaluation results to detect issues (overfitting, weak baseline improvement, class imbalance impact, low model diversity). Generates actionable suggestions and decides whether to trigger replanning with targeted strategies (regularisation, SMOTE, ensemble, feature engineering).
- **Memory** (`agents/memory.py`): JSON-based persistent storage keyed by dataset fingerprint. Remembers best models and metrics from previous runs, enabling the planner to learn from experience. Handles corrupted files gracefully with automatic backup and reset.

### Tools
- **Data Profiler** (`tools/data_profiler.py`): Generates a rich dataset profile including shape, types, missing values, skewness, IQR outliers, target correlations, duplicates, high-cardinality detection, ID column detection, and constant column detection. Produces actionable notes consumed by the planner. Supports both classification and regression target inference.
- **Modelling** (`tools/modelling.py`): Builds adaptive preprocessing pipelines (handles high-cardinality encoding via ordinal encoder, skew transforms, outlier clipping). Selects candidate models based on dataset size, imbalance, and task type (classification or regression). Trains with optional cross-validation for small datasets. Each model is wrapped in error isolation (try/except) so a single model failure does not crash the pipeline.
- **Evaluation** (`tools/evaluation.py`): Produces confusion matrix, model comparison chart, feature importance plot, per-class performance analysis, and a comprehensive markdown report with plan rationale.

---

## How to Run

### Setup
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Run on any dataset
```bash
python run_agent.py --data data/demo.csv
python run_agent.py --data data/WineQuality.csv --target quality
python run_agent.py --data data/adult.csv
```

### Parameters
| Flag | Description | Default |
|---|---|---|
| `--data` | Path to CSV dataset | (required) |
| `--target` | Target column name (auto-detected if omitted) | `auto` |
| `--output_root` | Output directory | `outputs` |
| `--seed` | Random seed | 42 |
| `--test_size` | Test split fraction | 0.2 |
| `--max_replans` | Maximum replan iterations | 2 |
| `--quiet` | Reduce log output | False |

### Outputs
Each run creates a timestamped folder in `outputs/` containing:
- `report.md` — Full markdown report with rationale
- `eda_summary.json` — Dataset profile
- `plan.json` — Execution plan with step-by-step rationale
- `metrics.json` — All model metrics
- `reflection.json` — Reflection analysis
- `run.log` — Structured log file with timestamps
- `confusion_matrix.png` — Best model confusion matrix (classification only)
- `model_comparison.png` — All models comparison chart
- `feature_importance.png` — Top feature importances (if available)

---

## How Planning Works

The planner returns a structured dict with three components:
- **steps**: ordered list of execution steps
- **rationale**: human-readable explanation for every decision (WHY, not just WHAT)
- **hints**: machine-consumable config passed to downstream tools

### Template Selection

| Condition | Template | Key Steps |
|---|---|---|
| Missing > 20% | High Missing | Advanced imputation |
| Imbalance >= 3x | Imbalanced | Class balancing, balanced metrics |
| Rows < 1000 | Small Dataset | Simple models, 10-fold CV |
| Rows > 50000 | Large Dataset | Skip expensive models (SVC, KNN) |
| Otherwise | Default | Full model suite |

### Conditional Branches

After selecting a template, the planner conditionally injects steps based on signals:
- Skewed features (|skewness| > 0.75) -> log1p transformation
- Outlier-heavy columns (>5% IQR outliers) -> clipping
- High-cardinality categoricals (>50 unique) -> ordinal encoding
- Weak predictors (correlation < 0.05) -> feature selection
- Multi-class (>10 classes) -> macro-averaged metrics
- Regression target -> regression models and R2/RMSE metrics
- Memory hints -> prioritise previously successful models or add ensemble

---

## Classification and Regression Support

The system auto-detects whether the target is classification or regression:
- **Classification**: target is categorical or has <= 50 unique values. Uses LogisticRegression, RandomForest, GradientBoosting, KNN, SVC, AdaBoost, VotingEnsemble. Metrics: accuracy, balanced accuracy, macro F1, precision, recall.
- **Regression**: target is continuous with > 50 unique values. Uses Ridge, Lasso, RandomForest, GradientBoosting, KNN, SVR. Metrics: R2, RMSE, MAE.

Both modes include a dummy baseline for comparison.

---

## How Reflection and Replanning Work

After evaluation, the reflector checks for:
1. **Baseline comparison**: Is the model meaningfully better than a dummy?
2. **Absolute thresholds**: Is F1/R2 acceptable?
3. **Overfitting**: Is there a large train-test gap?
4. **Class bias**: Does accuracy diverge from balanced accuracy?
5. **Model diversity**: Are all models performing similarly?
6. **Small dataset warning**: High score on few rows may not generalise

If issues are found and performance is below threshold, the reflector recommends replanning. Each replan applies targeted strategies:

| Issue | Strategy |
|---|---|
| Overfitting | Force regularisation (lower C, shallower trees) |
| Imbalance impact | Enable SMOTE / class weights |
| All models similar | Add feature engineering steps |
| Barely beats baseline | Try ensemble (voting classifier) |
| Weak features | Add feature selection |

The system supports up to `max_replans` iterations (default: 2).

---

## Error Isolation

Each model is trained inside its own try/except block. If a single model fails (e.g., SVC on a large dataset), the pipeline continues with the remaining models. Failed models are tracked and reported but do not crash the run.

---

## Structured Logging

Every run produces a `run.log` file with timestamped entries covering:
- Dataset loading and shape
- Profile summary (imbalance, missing data, feature counts)
- Memory lookup results
- Plan steps and rationale
- Model training progress
- Evaluation scores
- Reflection issues and suggestions
- Replan decisions

---

## How Memory Works

The agent stores a record for each unique dataset fingerprint (based on shape, columns, and target). On subsequent runs with the same dataset:
- If the previous best model had score > 0.70, it is prioritised
- If the previous run had poor results (score < 0.50), ensemble strategies are added
- Corrupted memory files are automatically backed up and reset
- Memory is stored in `agent_memory.json` and persists across runs

---

## Testing

76 unit tests across 7 test files with 89% coverage on core modules:

| Module | Tests | Coverage |
|---|---|---|
| `tools/data_profiler.py` | 13 | 79% |
| `tools/modelling.py` | 7 | 87% |
| `tools/evaluation.py` | 12 | 92% |
| `agents/planner.py` | 15 | 91% |
| `agents/reflector.py` | 22 | 96% |
| `agents/memory.py` | 6 | 100% |
| Smoke test | 1 | — |

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=tools --cov=agents --cov-report=term-missing -v
```

---

## Tested Datasets

| Dataset | Task | Best Model | Key Score |
|---|---|---|---|
| demo.csv | Classification | RandomForest | F1=0.93 |
| adult.csv | Classification | GradientBoosting | F1=0.79 |
| facebook.csv | Regression | GradientBoosting | R2=0.14 |
| WineQuality.csv | Regression | RandomForest | R2=0.33 |
| Life Expectancy.csv | Regression | GradientBoosting | R2=0.84 |
| letter-recognition.csv | Classification (16 classes) | RandomForest | F1=0.93 |
| creditcard.csv | Classification (577x imbalance) | GradientBoosting | F1=0.78 |

---

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, pytest, pytest-cov
- Fully offline — no APIs or cloud services
