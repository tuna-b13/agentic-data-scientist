"""
Evaluation Tool
Evaluates the best model with detailed metrics: confusion matrix,
per-class performance, classification report, and feature importance.
Generates markdown reports and saves all artefacts to the output directory.
"""

import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_model_comparison(all_metrics: List[Dict[str, Any]], out_path: str) -> None:
    """Bar chart comparing all candidate models on key metrics."""
    models = [m["model"] for m in all_metrics]
    is_clf = "accuracy" in all_metrics[0]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))

    if is_clf:
        bar1 = [m.get("balanced_accuracy", 0) for m in all_metrics]
        bar2 = [m.get("f1_macro", 0) for m in all_metrics]
        ax.bar(x - width / 2, bar1, width, label="Balanced Accuracy", color="#3498db")
        ax.bar(x + width / 2, bar2, width, label="F1 Macro", color="#e74c3c")
        ax.set_ylim(0, 1.05)
    else:
        bar1 = [m.get("r2", 0) for m in all_metrics]
        bar2 = [m.get("mae", 0) for m in all_metrics]
        ax.bar(x - width / 2, bar1, width, label="R2", color="#3498db")
        ax.bar(x + width / 2, bar2, width, label="MAE", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance: Dict[str, float], out_path: str, top_n: int = 15) -> None:
    """Plot top feature importances if available."""
    sorted_feats = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    if not sorted_feats:
        return
    names, values = zip(*sorted_feats)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#2ecc71")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _extract_feature_importance(best_result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract feature importances from the best model pipeline if possible."""
    try:
        pipe = best_result["pipeline"]
        model = pipe.named_steps["model"]
        preprocessor = pipe.named_steps["preprocess"]

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return None

        if len(feature_names) != len(importances):
            return None

        clean_names = [str(n).replace("num__", "").replace("cat__", "").replace("high_cat__", "") for n in feature_names]
        return {n: round(float(v), 4) for n, v in zip(clean_names, importances)}
    except Exception:
        return None


def _per_class_analysis(y_test, y_pred) -> List[Dict[str, Any]]:
    """Generate per-class precision, recall, F1 for detailed analysis."""
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    per_class = []
    for label, metrics in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        per_class.append({
            "class": str(label),
            "precision": round(float(metrics.get("precision", 0)), 4),
            "recall": round(float(metrics.get("recall", 0)), 4),
            "f1": round(float(metrics.get("f1-score", 0)), 4),
            "support": int(metrics.get("support", 0)),
        })
    return per_class


def evaluate_best(training_payload: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Full evaluation of the best model with all artefacts."""
    best = training_payload["best"]
    all_metrics = training_payload["all_metrics"]
    is_clf = training_payload.get("is_classification", True)

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    # Model comparison chart
    comparison_path = os.path.join(output_dir, "model_comparison.png")
    plot_model_comparison(all_metrics, comparison_path)

    # Feature importance
    importance = _extract_feature_importance(best)
    importance_path = None
    if importance:
        importance_path = os.path.join(output_dir, "feature_importance.png")
        plot_feature_importance(importance, importance_path)

    result = {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "model_comparison_path": comparison_path,
        "feature_importance_path": importance_path,
        "feature_importance": importance,
        "is_classification": is_clf,
    }

    if is_clf:
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted([str(x) for x in y_test.dropna().unique().tolist()])
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {best['name']}")
        cls_report = classification_report(y_test, y_pred, zero_division=0)
        per_class = _per_class_analysis(y_test, y_pred)
        result["confusion_matrix_path"] = cm_path
        result["classification_report"] = cls_report
        result["per_class_performance"] = per_class
    else:
        result["confusion_matrix_path"] = None
        result["classification_report"] = None
        result["per_class_performance"] = []

    return result


def _format_best_metrics(best: Dict[str, Any], is_clf: bool) -> str:
    if is_clf:
        return f"""| Metric | Score |
|---|---|
| Accuracy | {best.get("accuracy", 0):.4f} |
| Balanced Accuracy | {best.get("balanced_accuracy", 0):.4f} |
| Macro F1 | {best.get("f1_macro", 0):.4f} |
| Macro Precision | {best.get("precision_macro", 0):.4f} |
| Macro Recall | {best.get("recall_macro", 0):.4f} |
| Train Accuracy | {best.get("train_accuracy", 0):.4f} |"""
    else:
        return f"""| Metric | Score |
|---|---|
| R2 | {best.get("r2", 0):.4f} |
| RMSE | {best.get("rmse", 0):.4f} |
| MAE | {best.get("mae", 0):.4f} |
| Train R2 | {best.get("train_r2", 0):.4f} |"""


def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: Dict[str, Any],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
) -> None:
    """Generate a comprehensive markdown report for the run."""
    best = eval_payload["best_metrics"]

    def short_list(xs: List[str], n: int = 12) -> str:
        return ", ".join(xs[:n]) + (" ..." if len(xs) > n else "")

    numeric = dataset_profile.get("feature_types", {}).get("numeric", [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes = dataset_profile.get("notes", [])
    per_class = eval_payload.get("per_class_performance", [])

    steps = plan.get("steps", [])
    rationale = plan.get("rationale", {})

    # Plan explanation
    plan_text = "\n".join([f"  {i}. {step}" for i, step in enumerate(steps, 1)])

    # Per-class table
    pc_header = "| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n"
    pc_rows = "\n".join([
        f"| {p['class']} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |"
        for p in per_class
    ])
    pc_table = pc_header + pc_rows if per_class else "(not available)"

    # Reflection
    issues_text = "\n".join([f"- {i}" for i in reflection.get("issues", [])]) or "- No issues detected."
    suggestions_text = "\n".join([f"- {s}" for s in reflection.get("suggestions", [])]) or "- None."

    md = f"""# Agentic Data Scientist Report

**Run ID:** `{ctx.run_id}`
**Started (UTC):** {ctx.started_at}
**Dataset:** `{ctx.data_path}`
**Target:** `{ctx.target}`
**Fingerprint:** `{fingerprint}`

---

## 1. Dataset Profile

| Property | Value |
|---|---|
| Rows | {dataset_profile["shape"]["rows"]} |
| Columns | {dataset_profile["shape"]["cols"]} |
| Classification | {dataset_profile.get("is_classification")} |
| Imbalance Ratio | {dataset_profile.get("imbalance_ratio")} |
| Missing Data | {dataset_profile.get("total_missing_pct", 0):.1f}% |
| Duplicates | {dataset_profile.get("duplicate_pct", 0):.1f}% |
| Skewed Features | {len(dataset_profile.get("highly_skewed_cols", []))} |

**Feature Types**
- Numeric ({len(numeric)}): {short_list(numeric)}
- Categorical ({len(categorical)}): {short_list(categorical)}

**Signals & Notes**
{chr(10).join([f"- {n}" for n in notes]) if notes else "- (none)"}

---

## 2. Execution Plan

The planner selected the following steps based on dataset signals:

{plan_text}

### Rationale (why each decision was made)

{"".join([f"- **[{tag}]** {reason}{chr(10)}" for tag, reason in rationale.items()]) if rationale else "- (no rationale recorded)"}

---

## 3. Results (Best Model)

**Model:** `{best.get("model")}`

{_format_best_metrics(best, eval_payload.get("is_classification", True))}

### All Model Results

```json
{json.dumps(eval_payload.get("all_metrics", []), indent=2)}
```

### Per-Class Performance

{pc_table}

---

## 4. Reflection

**Status:** {reflection.get("status", "unknown")}
**Replan Recommended:** {reflection.get("replan_recommended", False)}

### Issues Identified
{issues_text}

### Suggestions
{suggestions_text}

---

## 5. Artefacts

- Confusion Matrix: `{eval_payload.get("confusion_matrix_path")}`
- Model Comparison: `{eval_payload.get("model_comparison_path")}`
- Feature Importance: `{eval_payload.get("feature_importance_path", "N/A")}`

"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
