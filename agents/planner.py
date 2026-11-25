"""
Planner Agent
Analyses dataset profile signals and generates a structured execution plan
with rationale (WHY each decision was made) and hints (machine-consumable
recommendations for downstream tools).

Every planning decision is justified and traceable.
"""

from typing import Any, Dict, List, Optional


# Plan templates

BASE_STEPS = [
    "profile_dataset",
    "build_preprocessor",
    "select_models",
    "train_models",
    "evaluate",
    "reflect",
    "write_report",
]


def _small_dataset_steps() -> List[str]:
    return [
        "profile_dataset",
        "drop_id_and_constant_cols",
        "handle_missing_data",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


def _large_dataset_steps() -> List[str]:
    return [
        "profile_dataset",
        "drop_id_and_constant_cols",
        "drop_duplicates",
        "handle_missing_data",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


def _imbalanced_steps() -> List[str]:
    return [
        "profile_dataset",
        "drop_id_and_constant_cols",
        "handle_missing_data",
        "apply_class_balancing",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


def _high_missing_steps() -> List[str]:
    return [
        "profile_dataset",
        "drop_id_and_constant_cols",
        "advanced_imputation",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


# Helpers

def _insert_before(plan: List[str], anchor: str, step: str) -> None:
    if anchor in plan and step not in plan:
        plan.insert(plan.index(anchor), step)


def _insert_after(plan: List[str], anchor: str, step: str) -> None:
    if anchor in plan and step not in plan:
        plan.insert(plan.index(anchor) + 1, step)


def _deduplicate(plan: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in plan:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# Signal extraction from profile

def _extract_signals(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Pull all planning-relevant signals into a flat dict."""
    return {
        "rows": profile.get("shape", {}).get("rows", 0),
        "cols": profile.get("shape", {}).get("cols", 0),
        "is_classification": profile.get("is_classification", True),
        "imbalance_ratio": float(profile.get("imbalance_ratio") or 1.0),
        "total_missing_pct": float(profile.get("total_missing_pct", 0)),
        "has_id_cols": bool(profile.get("id_columns")),
        "has_constant_cols": bool(profile.get("constant_columns")),
        "has_high_cardinality": bool(profile.get("high_cardinality_cols")),
        "has_skewed_cols": bool(profile.get("highly_skewed_cols")),
        "has_outlier_cols": bool(profile.get("high_outlier_cols")),
        "n_skewed": len(profile.get("highly_skewed_cols", [])),
        "n_outlier": len(profile.get("high_outlier_cols", [])),
        "n_weak_predictors": len(profile.get("weak_predictors", [])),
        "duplicate_pct": float(profile.get("duplicate_pct", 0)),
        "n_classes": profile.get("n_classes"),
        "id_columns": profile.get("id_columns", []),
        "high_cardinality_cols": profile.get("high_cardinality_cols", []),
        "highly_skewed_cols": profile.get("highly_skewed_cols", []),
        "high_outlier_cols": profile.get("high_outlier_cols", []),
    }


# Main entry point

def create_plan(
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a structured execution plan with rationale and hints.

    Returns:
        {
            "steps": List[str],
            "rationale": Dict[str, str],   # tag -> human-readable reason
            "hints": Dict[str, Any],       # machine-consumable downstream config
        }
    """
    sig = _extract_signals(dataset_profile)
    rationale: Dict[str, str] = {}
    hints: Dict[str, Any] = {
        "drop_columns": [],
        "priority_models": [],
        "log_transform_cols": [],
        "clip_outlier_cols": [],
        "resample_strategy": None,
        "cv_folds": 5,
        "plan_template": "default",
        "replan_attempt": 0,
    }

    # Step 1: Select base template
    if sig["total_missing_pct"] > 20:
        steps = _high_missing_steps()
        hints["plan_template"] = "high_missing"
        rationale["template_high_missing"] = (
            f"Overall missing data is {sig['total_missing_pct']:.1f}% (>20%). "
            "Selected high-missing template with advanced imputation."
        )
    elif sig["is_classification"] and sig["imbalance_ratio"] >= 3.0:
        steps = _imbalanced_steps()
        hints["plan_template"] = "imbalanced"
        hints["resample_strategy"] = "class_weight"
        rationale["template_imbalanced"] = (
            f"Class imbalance ratio is {sig['imbalance_ratio']:.1f}x (>=3.0). "
            "Selected imbalanced template with class balancing and balanced metrics."
        )
    elif sig["rows"] < 1000:
        steps = _small_dataset_steps()
        hints["plan_template"] = "small"
        hints["cv_folds"] = 10
        hints["priority_models"] = ["LogisticRegression", "RandomForest"]
        rationale["template_small"] = (
            f"Dataset has only {sig['rows']} rows (<1000). "
            "Selected small-dataset template: simpler models, 10-fold CV to guard against overfitting."
        )
    elif sig["rows"] > 50000:
        steps = _large_dataset_steps()
        hints["plan_template"] = "large"
        hints["priority_models"] = ["RandomForest", "GradientBoosting"]
        rationale["template_large"] = (
            f"Dataset has {sig['rows']} rows (>50k). "
            "Selected large-dataset template: skipping expensive models (SVC, KNN)."
        )
    else:
        steps = list(BASE_STEPS)
        hints["plan_template"] = "default"
        rationale["template_default"] = (
            f"Standard dataset ({sig['rows']} rows, {sig['cols']} cols). "
            "No extreme conditions detected; using default template."
        )

    # Step 2: Conditional branches

    # ID / constant columns
    if sig["has_id_cols"] or sig["has_constant_cols"]:
        _insert_after(steps, "profile_dataset", "drop_id_and_constant_cols")
        hints["drop_columns"].extend(sig["id_columns"])
        rationale["drop_ids"] = (
            f"Detected ID-like or constant columns: {sig['id_columns']}. "
            "These carry no predictive signal and would add noise."
        )

    # Duplicates
    if sig["duplicate_pct"] > 5:
        _insert_after(steps, "profile_dataset", "drop_duplicates")
        rationale["drop_duplicates"] = (
            f"Duplicate rows: {sig['duplicate_pct']:.1f}% of data. "
            "Dropping to prevent data leakage and biased evaluation."
        )

    # Skewed features
    if sig["has_skewed_cols"]:
        _insert_before(steps, "build_preprocessor", "apply_skew_transform")
        hints["log_transform_cols"] = sig["highly_skewed_cols"]
        rationale["log_transform"] = (
            f"{sig['n_skewed']} features have |skewness| > 0.75. "
            "Applying log1p to normalise distributions and improve model convergence."
        )

    # Outlier-heavy features
    if sig["has_outlier_cols"]:
        _insert_before(steps, "build_preprocessor", "clip_outliers")
        hints["clip_outlier_cols"] = sig["high_outlier_cols"]
        rationale["clip_outliers"] = (
            f"{sig['n_outlier']} features have >5% outliers (IQR method). "
            "Clipping at 5th/95th percentile to reduce extreme value impact."
        )

    # High-cardinality categoricals
    if sig["has_high_cardinality"]:
        _insert_before(steps, "build_preprocessor", "encode_high_cardinality")
        rationale["target_encode"] = (
            f"High-cardinality categoricals detected: {sig['high_cardinality_cols']}. "
            "Using ordinal encoding instead of one-hot to prevent feature explosion."
        )

    # Weak predictors
    if sig["n_weak_predictors"] > 3:
        _insert_before(steps, "select_models", "feature_selection")
        rationale["feature_selection"] = (
            f"{sig['n_weak_predictors']} features have correlation < 0.05 with target. "
            "Adding feature selection to remove noise."
        )

    # Many classes
    if sig["n_classes"] and sig["n_classes"] > 10:
        rationale["multiclass"] = (
            f"Multi-class problem with {sig['n_classes']} classes. "
            "Using macro-averaged metrics to give equal weight to all classes."
        )

    # Non-classification (regression)
    if not sig["is_classification"]:
        rationale["regression"] = (
            "Target is continuous (not classification). "
            "Using regression models (Ridge, Lasso, RF, GB) and R2/RMSE metrics."
        )

    steps = _deduplicate(steps)

    # Step 3: Memory-guided adjustments
    if memory_hint:
        _apply_memory_hints(steps, hints, rationale, memory_hint)

    return {
        "steps": steps,
        "rationale": rationale,
        "hints": hints,
    }


def _apply_memory_hints(
    steps: List[str],
    hints: Dict[str, Any],
    rationale: Dict[str, str],
    memory_hint: Dict[str, Any],
) -> None:
    """Adjust plan based on memory from previous runs."""
    best_prev = memory_hint.get("best_model", "")
    prev_metrics = memory_hint.get("best_metrics", {})
    prev_score = prev_metrics.get("f1_macro") or prev_metrics.get("r2") or 0.0
    prev_score = float(prev_score)
    hint_source = memory_hint.get("hint_source", "exact")

    if best_prev and prev_score > 0.70:
        if best_prev not in hints["priority_models"]:
            hints["priority_models"].insert(0, best_prev)
        tag = f"memory_priority_{hint_source}"
        rationale[tag] = (
            f"Memory ({hint_source} match): previous best was {best_prev} "
            f"(score={prev_score:.3f}). Prioritising this model."
        )

    if 0 < prev_score < 0.50:
        _insert_before(steps, "train_models", "try_ensemble_strategy")
        rationale["memory_low_score"] = (
            f"Memory: previous score was low ({prev_score:.3f}). "
            "Adding ensemble strategy for this iteration."
        )


def explain_plan(plan: Dict[str, Any]) -> str:
    """Generate a human-readable explanation of the plan with rationale."""
    lines = []
    steps = plan.get("steps", [])
    rationale = plan.get("rationale", {})

    lines.append("EXECUTION PLAN:")
    for i, step in enumerate(steps, 1):
        lines.append(f"  {i}. {step}")

    if rationale:
        lines.append("")
        lines.append("RATIONALE (why each decision was made):")
        for tag, reason in rationale.items():
            lines.append(f"  [{tag}] {reason}")

    return "\n".join(lines)
