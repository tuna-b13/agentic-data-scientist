"""
Reflector Agent
Analyses model evaluation results in the context of dataset characteristics.
Identifies issues (overfitting, weak performance, class-specific failures),
generates actionable suggestions, and applies concrete replan strategies
when performance is insufficient.
"""

from typing import Any, Dict, List, Tuple


# Reflection: analyse results and identify issues

def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyse evaluation results and generate a structured reflection.

    Checks for:
    - Weak improvement over dummy baseline
    - Low absolute F1 / balanced accuracy
    - Overfitting signals (train >> test)
    - Class imbalance impact
    - Model diversity (are all models similar?)
    - Per-class performance gaps
    """
    best_model = evaluation.get("model", "unknown")
    is_clf = dataset_profile.get("is_classification", True)
    rows = dataset_profile.get("shape", {}).get("rows", 0)

    issues: List[str] = []
    suggestions: List[str] = []

    if is_clf:
        primary_score, replan_thresh = _reflect_classification(evaluation, all_metrics, dataset_profile, issues, suggestions)
    else:
        primary_score, replan_thresh = _reflect_regression(evaluation, all_metrics, dataset_profile, issues, suggestions)

    # Model diversity check
    if len(all_metrics) >= 3:
        non_dummy = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
        if len(non_dummy) >= 2:
            score_key = "f1_macro" if is_clf else "r2"
            scores = [float(m.get(score_key, 0)) for m in non_dummy]
            score_range = max(scores) - min(scores)
            if score_range < 0.02 and primary_score < 0.70:
                issues.append(
                    f"All models perform similarly ({score_key} range={score_range:.3f}). "
                    "Problem may be in preprocessing, not model choice."
                )
                suggestions.append(
                    "Focus on feature engineering and preprocessing "
                    "rather than trying more model types."
                )

    # Small dataset warning
    if rows < 500 and primary_score > 0.90:
        issues.append(
            "Very high performance on a small dataset —"
            "results may not generalise. Cross-validation recommended."
        )
        suggestions.append(
            "Use stratified k-fold CV instead of a single train/test split "
            "to get more reliable performance estimates."
        )

    has_overfitting = any("overfitting" in i.lower() for i in issues)
    status = "needs_attention" if issues else "ok"
    replan_recommended = (primary_score < replan_thresh and len(issues) > 0) or has_overfitting

    return {
        "status": status,
        "best_model": best_model,
        "primary_score": round(primary_score, 4),
        "issues": issues,
        "suggestions": suggestions,
        "replan_recommended": replan_recommended,
        "n_issues": len(issues),
    }


def _reflect_classification(evaluation, all_metrics, profile, issues, suggestions):
    """Classification-specific reflection checks."""
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    accuracy = float(evaluation.get("accuracy", 0.0))
    imb = float(profile.get("imbalance_ratio") or 1.0)

    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba
        if improvement < 0.02:
            issues.append(f"Best model barely beats baseline (improvement={improvement:.3f}).")
            suggestions.append("Investigate feature quality, check for target leakage or noisy features.")
        elif improvement < 0.05:
            issues.append(f"Marginal improvement over baseline ({improvement:.3f}).")
            suggestions.append("Try polynomial features, interaction terms, or different encoding.")

    if f1_macro < 0.40:
        issues.append(f"Very low macro F1 ({f1_macro:.3f}).")
        suggestions.append("Consider different preprocessing, feature engineering, or ensemble methods.")
    elif f1_macro < 0.60:
        issues.append(f"Modest macro F1 ({f1_macro:.3f}). Room for improvement.")
        suggestions.append("Try hyperparameter tuning or adding more model candidates.")

    train_acc = float(evaluation.get("train_accuracy", 0.0))
    if train_acc > 0 and accuracy > 0:
        gap = train_acc - accuracy
        if gap > 0.15:
            issues.append(f"Overfitting detected: train-test gap={gap:.3f}.")
            suggestions.append("Reduce model complexity, add regularisation, or use cross-validation.")
        elif gap > 0.08:
            issues.append(f"Mild overfitting: train-test gap={gap:.3f}.")
            suggestions.append("Consider lighter regularisation or pruning.")

    if accuracy > 0 and bal_acc > 0 and (accuracy - bal_acc) > 0.10:
        issues.append(f"Accuracy ({accuracy:.3f}) >> balanced accuracy ({bal_acc:.3f}). Majority class bias.")
        suggestions.append("Apply class weights, SMOTE, or threshold tuning.")

    if imb >= 3.0 and f1_macro < 0.50:
        issues.append(f"Imbalanced dataset (ratio={imb:.1f}) with low macro F1.")
        suggestions.append("Try SMOTE, class_weight='balanced', or threshold tuning.")

    return f1_macro, 0.55


def _reflect_regression(evaluation, all_metrics, profile, issues, suggestions):  # noqa: ARG001
    """Regression-specific reflection checks."""
    r2 = float(evaluation.get("r2", 0.0))
    train_r2 = float(evaluation.get("train_r2", 0.0))

    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    if dummy is not None:
        dummy_r2 = float(dummy.get("r2", 0.0))
        improvement = r2 - dummy_r2
        if improvement < 0.05:
            issues.append(f"Best model barely beats baseline (R2 improvement={improvement:.3f}).")
            suggestions.append("Investigate feature quality or try feature engineering.")

    if r2 < 0.0:
        issues.append(f"Negative R2 ({r2:.3f}). Model is worse than predicting the mean.")
        suggestions.append("Data may have no learnable signal, or preprocessing is harmful.")
    elif r2 < 0.30:
        issues.append(f"Low R2 ({r2:.3f}). Weak predictive power.")
        suggestions.append("Try different preprocessing, feature engineering, or non-linear models.")
    elif r2 < 0.50:
        issues.append(f"Modest R2 ({r2:.3f}). Room for improvement.")
        suggestions.append("Try hyperparameter tuning or ensemble methods.")

    if train_r2 > 0 and r2 > 0:
        gap = train_r2 - r2
        if gap > 0.20:
            issues.append(f"Overfitting detected: train R2={train_r2:.3f} >> test R2={r2:.3f}.")
            suggestions.append("Reduce model complexity or add regularisation.")
        elif gap > 0.10:
            issues.append(f"Mild overfitting: train-test R2 gap={gap:.3f}.")
            suggestions.append("Consider lighter regularisation.")

    return r2, 0.30


# Replan decision

def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Decide whether to trigger replanning.
    Considers: replan flag, number of issues, and severity.
    """
    if reflection.get("replan_recommended", False):
        return True
    if reflection.get("n_issues", 0) >= 3 and reflection.get("best_f1_macro", 1.0) < 0.60:
        return True
    return False


# Replan strategies: concrete changes to plan and profile

def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Modify the plan and profile based on reflection findings.
    Each issue type triggers a specific, targeted strategy change.
    """
    new_plan = list(plan)
    new_profile = dict(dataset_profile)
    notes = list(new_profile.get("notes", []))

    issues_text = " ".join(reflection.get("issues", []))
    suggestions_text = " ".join(reflection.get("suggestions", []))
    imb = float(new_profile.get("imbalance_ratio") or 1.0)

    strategies_applied = []

    # Strategy 1: If overfitting detected, add regularisation
    if "overfitting" in issues_text.lower():
        new_profile["force_regularisation"] = True
        _ensure_step(new_plan, "apply_regularisation", before="train_models")
        strategies_applied.append("Added regularisation to combat overfitting.")

    # Strategy 2: If imbalance is hurting performance, enable SMOTE
    if imb >= 3.0 and "minority" in issues_text.lower():
        new_profile["use_smote"] = True
        _ensure_step(new_plan, "apply_class_balancing", before="build_preprocessor")
        strategies_applied.append("Enabled SMOTE for minority class oversampling.")

    # Strategy 3: If all models similar, focus on feature engineering
    if "similarly" in issues_text.lower() or "preprocessing" in suggestions_text.lower():
        _ensure_step(new_plan, "apply_skew_transform", before="build_preprocessor")
        _ensure_step(new_plan, "clip_outliers", before="build_preprocessor")
        new_profile["force_feature_engineering"] = True
        strategies_applied.append("Added feature engineering steps (skew transform + outlier clipping).")

    # Strategy 4: If barely beats baseline, try ensemble
    if "baseline" in issues_text.lower() or "barely" in issues_text.lower():
        new_profile["try_ensemble"] = True
        _ensure_step(new_plan, "try_ensemble_strategy", before="train_models")
        strategies_applied.append("Adding ensemble/voting strategy to improve over baseline.")

    # Strategy 5: If weak features detected, add feature selection
    if "feature" in suggestions_text.lower() and "engineering" in suggestions_text.lower():
        _ensure_step(new_plan, "feature_selection", before="select_models")
        strategies_applied.append("Added feature selection step.")

    # Fallback: if no specific strategy matched, apply general improvements
    if not strategies_applied:
        new_profile["increase_model_diversity"] = True
        strategies_applied.append("General replan: increasing model diversity and preprocessing.")

    # Log what changed
    notes.append(f"Replan applied: {'; '.join(strategies_applied)}")
    new_profile["notes"] = notes

    _ensure_step(new_plan, "replan_attempt", after="reflect")

    return new_plan, new_profile


def _ensure_step(plan: List[str], step: str, before: str = None, after: str = None) -> None:
    """Insert a step if not already present."""
    if step in plan:
        return
    if before and before in plan:
        plan.insert(plan.index(before), step)
    elif after and after in plan:
        plan.insert(plan.index(after) + 1, step)
    else:
        plan.append(step)
