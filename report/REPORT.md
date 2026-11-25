# CE888 Technical Report: Offline Agentic Data Scientist

**Author:** Tuna Bozbay
**Module:** CE888 — Data Science and Decision Making
**Date:** April 2026

---

## 1. Introduction

Machine learning workflows are often manual and repetitive: a data scientist loads a dataset, inspects its characteristics, selects appropriate preprocessing steps, chooses candidate models, evaluates results, and iterates based on observed weaknesses. This project automates that entire loop by building an offline agentic data scientist — a system that makes autonomous, data-driven decisions at every stage of the pipeline without relying on Large Language Models or external APIs.

The goal is not to maximise raw predictive accuracy but to build a system that reasons about data, plans accordingly, reflects on outcomes, and learns from experience. Every decision the agent makes is accompanied by a human-readable rationale explaining why it was made, making the system's reasoning fully transparent and traceable.

The design draws on core agentic AI principles: **autonomy** (the system operates without human intervention once started), **perception** (the data profiler extracts rich signals from the dataset), **planning** (the planner selects execution strategies based on perceived signals), **action** (tools execute the plan), and **reflection** (the reflector evaluates outcomes and triggers replanning when needed). This perceive-plan-act-reflect loop is the fundamental agentic cycle that distinguishes this system from a static pipeline. Additionally, the persistent memory component adds a learning dimension — the agent improves its strategy over repeated runs on the same dataset, moving beyond a purely reactive system toward one that accumulates experience.

The system supports both classification and regression tasks, auto-detecting the task type from the target column. It has been tested on seven diverse datasets ranging from 500 to 284,807 rows, covering binary classification, multi-class classification (16 classes), extreme class imbalance (577:1 ratio), and continuous regression targets.

---

## 2. System Design

### 2.1 Architecture Overview

The system follows a modular agent-based architecture with three agents (decision-makers) and three tools (executors), coordinated by a central orchestrator:

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

**Orchestrator** (`agentic_data_scientist.py`): Coordinates the full pipeline loop: load → profile → plan → train → evaluate → reflect → (replan) → report. Manages the run context (seed, test size, output directory), handles the replan loop, and persists all artefacts. Uses Python's logging module to write timestamped entries to both console and a `run.log` file.

**Planner Agent** (`agents/planner.py`): Analyses dataset profile signals and generates a structured execution plan. Returns a dictionary with three components: `steps` (ordered execution steps), `rationale` (human-readable WHY for each decision), and `hints` (machine-consumable configuration for downstream tools).

**Reflector Agent** (`agents/reflector.py`): Analyses evaluation results in context of the dataset profile. Checks for overfitting, weak baseline improvement, class imbalance impact, low model diversity, and small dataset reliability. Generates actionable suggestions and decides whether to trigger replanning.

**Memory Agent** (`agents/memory.py`): JSON-based persistent storage keyed by dataset fingerprint. Enables cross-run learning: the planner can prioritise previously successful models or add ensemble strategies when past performance was poor. Handles corrupted files gracefully with automatic backup and reset.

**Data Profiler** (`tools/data_profiler.py`): Generates a comprehensive dataset profile with 20+ signals consumed by the planner.

**Modelling Tool** (`tools/modelling.py`): Builds adaptive preprocessing pipelines and selects candidate models based on dataset characteristics. Supports both classification (7 model types) and regression (6 model types). Each model is trained inside its own error isolation block.

**Evaluation Tool** (`tools/evaluation.py`): Produces confusion matrices, model comparison charts, feature importance plots, per-class performance analysis, and comprehensive markdown reports.

### 2.2 Design Rationale

The agent-tool separation mirrors how a human data scientist works: you first decide what to do (planning), then execute it (tools), then evaluate whether it worked (reflection). Keeping these as separate modules makes the system testable — each component can be unit-tested independently — and extensible — new models or preprocessing steps can be added without changing the planning or reflection logic.

The structured plan dictionary (steps + rationale + hints) was designed to support traceability. A flat list of steps would tell you what the system did but not why. The rationale dictionary provides a tagged explanation for every conditional decision, and the hints dictionary passes machine-readable configuration to downstream tools without coupling the planner to the modelling implementation.

---

## 3. Data Signals

The data profiler extracts a rich set of signals that drive all downstream decisions. These signals fall into several categories:

### 3.1 Structural Signals

- **Shape**: Row and column counts determine template selection (small/large dataset) and model eligibility (SVC excluded above 10k rows).
- **Feature types**: Numeric vs categorical classification determines which preprocessing pipelines to build.
- **ID columns**: Detected by name pattern (`id`, `_id`, `index`) or uniqueness (all values unique, non-float). These carry no predictive signal and are dropped.
- **Constant columns**: Zero or near-zero variance columns are detected and dropped.

### 3.2 Distribution Signals

- **Skewness**: Computed for all numeric features. Columns with |skewness| > 0.75 are flagged for log1p transformation to normalise distributions and improve model convergence.
- **Outliers (IQR)**: For each numeric column, outliers are detected using the interquartile range method. Columns with >5% outliers are flagged for clipping. The exact lower and upper bounds are stored and used during preprocessing.
- **Missing data**: Both per-column and overall missing percentages are computed. Total missing above 20% triggers the high-missing template with advanced imputation.

### 3.3 Target Signals

- **Task type**: Classification vs regression is inferred from the target column. Categorical targets or targets with ≤50 unique values are treated as classification; otherwise regression.
- **Class balance**: For classification targets, class counts and imbalance ratio (majority/minority) are computed. Ratio ≥ 3.0 triggers the imbalanced template.
- **Target correlations**: Absolute Pearson correlation between each numeric feature and the target is computed. Features with correlation < 0.05 are flagged as weak predictors.

### 3.4 Cardinality Signals

- **High-cardinality categoricals**: Categorical columns with >50 unique values are flagged. These cannot be one-hot encoded efficiently (would create thousands of sparse features) and instead use ordinal encoding.
- **Duplicate detection**: Duplicate row percentage is computed. Rates above 5% trigger duplicate removal to prevent data leakage.

### 3.5 Actionable Notes

All detected signals are compiled into a list of human-readable notes that appear in the profile, the execution log, and the final report. These notes serve as a transparent audit trail of what the profiler found and how it will influence planning.

---

## 4. Decision Making

### 4.1 Plan Template Selection

The planner selects one of five base templates based on the primary challenge detected:

| Priority | Condition | Template | Key Difference |
|---|---|---|---|
| 1 | Missing > 20% | High Missing | Advanced imputation step |
| 2 | Imbalance ≥ 3x | Imbalanced | Class balancing, balanced metrics |
| 3 | Rows < 1000 | Small Dataset | Simpler models, 10-fold CV |
| 4 | Rows > 50000 | Large Dataset | Skip expensive models (SVC, KNN) |
| 5 | Otherwise | Default | Full model suite |

The priority order ensures the most impactful challenge is addressed first: missing data affects all downstream processing, imbalance affects model selection and evaluation metrics, and size affects computational feasibility.

### 4.2 Conditional Branches

After template selection, additional steps are injected based on secondary signals. Each injection produces a rationale entry. For example:

```
[log_transform] 4 features have |skewness| > 0.75.
Applying log1p to normalise distributions and improve model convergence.
```

### 4.3 Model Selection

The modelling tool selects candidates based on dataset characteristics:

- **SVC**: Only for datasets with <10k rows and <100 columns (O(n²) memory complexity).
- **KNN**: Only for datasets with <20k rows (requires distance computation to all training points).
- **Class weights**: When imbalance ≥ 3x, models supporting `class_weight="balanced"` are configured accordingly.
- **Regularisation**: When the reflector triggers `force_regularisation`, models use stronger regularisation (lower C, shallower trees, fewer estimators).
- **Ensembles**: When `try_ensemble` is set, AdaBoost and VotingClassifier are added.

For regression tasks, the system selects Ridge, Lasso, RandomForest, GradientBoosting, KNN, and SVR with analogous size-based filtering.

### 4.4 Memory-Guided Planning

When the planner detects a memory record for the current dataset fingerprint:

- If the previous best model scored above 0.70, that model is added to the priority list — it worked before on the same data structure.
- If the previous best score was below 0.50, an ensemble strategy is injected from the start, saving one replan cycle.

---

## 5. Tool Evaluation

### 5.1 Preprocessing Pipeline

The preprocessing pipeline adapts to feature types using sklearn's ColumnTransformer:

- **Numeric**: Median imputation (robust to outliers) + StandardScaler.
- **Low-cardinality categoricals**: Most-frequent imputation + OneHotEncoder with `handle_unknown="ignore"`.
- **High-cardinality categoricals**: Most-frequent imputation + OrdinalEncoder with `unknown_value=-1`.
- **ID/constant columns**: Dropped via ColumnTransformer's `remainder="drop"`.

Before modelling, the system also applies data-level preprocessing: dropping duplicates (if >5%), clipping outliers to IQR bounds, and log1p-transforming skewed features.

### 5.2 Training and Error Isolation

Each model is trained inside its own try/except block. If a model fails (e.g., convergence issues, memory errors), the pipeline continues with remaining models. Failed models are tracked with their error messages and reported. The system only raises a RuntimeError if all models fail. This ensures robustness: a single problematic model does not crash the entire run.

### 5.3 Cross-Validation

For datasets with fewer than 1,000 rows, cross-validation scores are automatically computed. This is critical because small datasets produce high-variance train/test splits. The system uses StratifiedKFold for classification and KFold for regression, with up to 5 folds.

### 5.4 Evaluation Artefacts

The evaluation tool produces:
- **Confusion matrix** (classification only): Visualises prediction distribution across classes.
- **Model comparison chart**: Bar chart comparing all candidates on key metrics (balanced accuracy + F1 for classification; R2 + MAE for regression).
- **Feature importance plot**: Top 15 feature importances extracted from tree-based or linear models.
- **Per-class performance**: Precision, recall, and F1 for each class individually.
- **Markdown report**: Comprehensive report including dataset profile, plan rationale, results, and reflection.

---

## 6. Reflection and Replanning

### 6.1 Reflection Checks

After evaluation, the reflector analyses results through multiple lenses:

1. **Baseline comparison**: If the best model improves less than 0.02 over the dummy baseline, the model is essentially not learning from features.
2. **Absolute thresholds**: F1 < 0.40 (very low) or < 0.60 (modest) for classification; R2 < 0.0 (worse than mean), < 0.30 (weak), or < 0.50 (modest) for regression.
3. **Overfitting**: Train-test accuracy gap > 0.15 (severe) or > 0.08 (mild) for classification; R2 gap > 0.20 or > 0.10 for regression. Overfitting always triggers replanning.
4. **Class bias**: Accuracy exceeding balanced accuracy by > 0.10 indicates majority class bias.
5. **Model diversity**: All non-dummy models within 0.02 of each other suggests the bottleneck is preprocessing, not model choice.
6. **Small dataset reliability**: Score > 0.90 on < 500 rows may not generalise.

### 6.2 Targeted Replan Strategies

Each identified issue triggers a specific strategy:

| Issue | Strategy | Effect |
|---|---|---|
| Overfitting | Force regularisation | Lower C, shallower trees, fewer estimators |
| Imbalance impact | Enable SMOTE / class weights | Better minority class performance |
| All models similar | Add feature engineering | Skew transform + outlier clipping |
| Barely beats baseline | Try ensemble | VotingClassifier + AdaBoost |
| Weak features | Feature selection | Remove low-correlation features |

If no specific strategy matches, the fallback increases model diversity.

### 6.3 Replan Limits

The system supports a configurable maximum number of replan attempts (default: 2). This prevents infinite loops when the data simply does not contain enough signal. After reaching the maximum, the system finalises with the best results achieved. This was validated on WineQuality where R2 plateaued at 0.33 — the system correctly stopped rather than replanning indefinitely.

---

## 7. Memory and Learning

### 7.1 Dataset Fingerprinting

The memory system uses a fingerprint function combining the dataset's shape, target column, and column names into a deterministic hash. The same CSV always produces the same fingerprint. Adding/removing columns, changing the target, or changing row count all change the fingerprint. This ensures memory records are only applied when the dataset structure genuinely matches a previous run.

### 7.2 What Is Stored

Each memory record contains: last run timestamp, target column, dataset shape, best model name, best metrics, and replan count. The JSONMemory class handles corrupted files gracefully — if JSON parsing fails, it creates a `.bak` backup and resets to empty with a note about the reset.

### 7.3 How Memory Is Used

Memory enables two forms of cross-run learning:

1. **Model prioritisation**: If a previous run found GradientBoosting as best with score > 0.70, subsequent runs prioritise that model. This does not skip other models — it ensures the proven model is always included.
2. **Strategy escalation**: If a previous run scored < 0.50, the planner adds an ensemble strategy from the start rather than waiting for the reflector to recommend it, saving one replan cycle.

---

## 8. Ethics and Limitations

### 8.1 Ethical Considerations

**Data Privacy**: The system operates fully offline with no external API calls, meaning no data leaves the user's machine. This is critical for sensitive datasets (e.g., the credit card fraud dataset contains real financial transaction data) where data privacy regulations such as GDPR may apply. The offline design was a deliberate architectural choice, not a limitation.

**Fairness and Bias**: The system does not perform fairness analysis — it does not check whether model predictions are biased against protected groups (gender, race, age). For example, when running on the adult income dataset, the system optimises for overall F1 but does not audit whether predictions are equitable across demographic groups. A responsible deployment would require adding fairness metrics (e.g., equalised odds, demographic parity) and auditing model decisions across protected attributes before using predictions for any real-world decision-making.

**Transparency vs Autonomy**: The system's autonomous nature creates a tension between convenience and oversight. While every planning decision includes a human-readable rationale, the model training itself remains a black box for complex models like GradientBoosting and ensemble methods. The feature importance plots provide some insight into what features drive predictions, but they do not explain individual predictions. Adding interpretability tools such as SHAP (Lundberg and Lee, 2017) or LIME would improve per-prediction explainability.

**Automation Risk**: An autonomous system that selects models and preprocessing steps without human oversight could make inappropriate choices for domain-specific applications. For instance, in medical diagnosis, a model that optimises for overall accuracy at the expense of recall on rare conditions could have serious consequences. The reflection mechanism partially addresses this by flagging class-specific failures, but domain expertise should always supplement the system's automated decisions.

**AI Disclosure**: This project used AI assistance, disclosed in accordance with the module's academic integrity policy.

### 8.2 Technical Limitations

**Target column auto-detection**: The system infers the target column using a heuristic: first checking common names (`target`, `label`, `class`, `y`, `outcome`), then preferring the sole non-numeric column, and finally falling back to the last column. This works reliably for datasets with clearly named targets or a single categorical column (e.g., `letter` in letter-recognition, `quality` in WineQuality, `Class` in creditcard, `label` in demo). However, it fails on datasets with no header row (adult.csv), multiple categorical columns where the target is ambiguous (Life Expectancy), or where the target is not the obvious categorical column (facebook). For these cases, the `--target` flag must be provided explicitly. The table below summarises auto-detection accuracy across all tested datasets:

| Dataset | Auto-detected | Correct | Notes |
|---|---|---|---|
| demo.csv | label | Yes | Matched by name heuristic |
| WineQuality.csv | quality | Yes | Last column, low cardinality |
| creditcard.csv | Class | Yes | Matched by name heuristic |
| letter-recognition.csv | letter | Yes | Sole non-numeric column |
| adult.csv | Male | No | CSV has no header row; requires `--target` |
| Life Expectancy.csv | Status | No | Multiple categoricals; actual target is numeric (`Life expectancy`); requires `--target` |
| facebook.csv | Type | No | Multiple plausible targets; requires `--target` |

**Feature engineering depth**: The system applies log1p transforms and outlier clipping but does not generate interaction terms, polynomial features, or domain-specific transformations. This limits performance on datasets like WineQuality where raw features have limited predictive power.

**Hyperparameter tuning**: The system uses sensible defaults and adjusts key parameters based on profile signals but does not perform systematic hyperparameter search (grid search, Bayesian optimisation). This was a deliberate trade-off: tuning adds significant compute time and the assignment prioritises reasoning and architecture over raw accuracy.

**Advanced imputation**: While the system detects high missing data rates and selects an appropriate template, the actual imputation is median/mode-based. More sophisticated methods (iterative imputation, KNN imputation) could be added.

**SMOTE integration**: The replan strategy can flag SMOTE as needed, but the current implementation relies on class weights rather than actual oversampling. Integrating imblearn's SMOTE would provide an additional option for handling imbalanced datasets.

---

## 9. Conclusion and Future Work

### 9.1 Summary

This project demonstrates that an offline agentic data scientist can make autonomous, justified decisions across the full machine learning pipeline. The key contribution is not raw predictive accuracy but the system's ability to reason about data characteristics, adapt its approach, reflect on outcomes, and learn from experience.

The plan-execute-reflect-replan loop, combined with structured rationale at every decision point, creates a transparent and traceable system. Every run produces not just predictions and metrics, but a complete record of why each decision was made and what the system learned.

The system was validated on seven diverse datasets:

| Dataset | Rows | Task | Template | Best Model | Key Score | Replanned |
|---|---|---|---|---|---|---|
| demo.csv | 150 | Classification (3 classes) | Small | RandomForest | F1=0.93 | No |
| adult.csv | 32,561 | Classification (2 classes) | Default | GradientBoosting | F1=0.79 | No |
| facebook.csv | 500 | Regression | Small | GradientBoosting | R2=0.14 | No |
| WineQuality.csv | 1,599 | Regression | Small | RandomForest | R2=0.33 | Yes (2x) |
| Life Expectancy.csv | 2,938 | Regression | Default | GradientBoosting | R2=0.84 | No |
| letter-recognition.csv | 20,000 | Classification (16 classes) | Default | RandomForest | F1=0.93 | Yes |
| creditcard.csv | 284,807 | Classification (577:1) | Imbalanced | GradientBoosting | F1=0.78 | Yes |

**Key observations from testing:**

- **Replanning is selective**: Only 3 out of 7 datasets triggered replanning. The system does not blindly replan — it only does so when specific issues are detected and performance is below threshold.
- **Template selection matters**: The imbalanced template for creditcard and the small dataset template for demo/WineQuality produced meaningfully different pipelines. A one-size-fits-all approach would have either wasted compute (running SVC on 284k rows) or missed critical steps (not using balanced metrics for imbalanced data).
- **Honest about limitations**: On WineQuality (R2=0.33) and facebook (R2=0.14), the system achieved modest scores but correctly identified that the data itself has limited predictive signal. The reflector's suggestions were appropriate rather than falsely promising improvement.
- **Overfitting recovery**: On letter-recognition, the initial iteration showed severe overfitting (train-test gap > 0.15). The replan triggered forced regularisation, which reduced the gap while maintaining F1=0.93 — demonstrating that the targeted strategy was effective.
- **Scale handling**: The creditcard dataset (284k rows) ran successfully with the large-scale model selection (excluding SVC/KNN) and imbalanced template (class weights), achieving F1=0.78 despite 577:1 class imbalance.

### 9.2 Testing and Reproducibility

The system includes 76 unit tests across 7 test files with 89% coverage on core modules (agents and tools). The test strategy follows a layered approach:

- **Unit tests**: Each function is tested in isolation with synthetic data. For example, the planner tests verify that a profile with 300 rows selects the small dataset template, while a profile with 100,000 rows selects the large dataset template. The reflector tests verify that a train-test accuracy gap of 0.20 triggers an overfitting issue, while a gap of 0.05 does not.
- **Integration tests**: The smoke test runs the full pipeline end-to-end on a synthetic dataset and verifies that all expected output files (report.md, plan.json, metrics.json, reflection.json, confusion_matrix.png, model_comparison.png) are created.
- **Edge case tests**: Error isolation is tested by injecting a deliberately broken model into the candidate list and verifying that the pipeline completes successfully with the remaining models. Memory corruption recovery is tested by writing invalid JSON to the memory file and verifying that the system resets gracefully.

All random operations use a configurable seed (default: 42) passed through the entire pipeline — train/test splits, model initialisation, cross-validation folds — ensuring that the same dataset with the same parameters always produces identical results.

### 9.3 Future Work

Several directions could extend this system's capabilities:

- **Automated feature engineering**: Generating interaction terms and polynomial features to improve performance on feature-limited datasets. This would address cases like WineQuality where raw features have limited predictive power but combinations of features might capture non-linear relationships.
- **Hyperparameter optimisation**: Adding Bayesian optimisation (e.g., Optuna) for key parameters without excessive compute cost. The current system adjusts a few parameters based on profile signals, but systematic tuning could improve performance especially on medium-sized datasets.
- **Fairness metrics**: Auditing model predictions across protected attributes to ensure equitable outcomes. This is particularly important for datasets like adult income where demographic features are present.
- **SHAP/LIME integration**: Providing per-prediction explanations beyond aggregate feature importance, enabling users to understand why the model made a specific prediction.
- **Similarity-based memory retrieval**: Matching against structurally similar (not just identical) datasets to transfer learning across related problems. Currently, memory only matches exact fingerprints — a similarity metric based on dataset characteristics could enable knowledge transfer between related datasets.

---

## References

- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.
- He, H. and Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE TKDE*, 21(9), 1263-1284.
- Lundberg, S. M. and Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30.
