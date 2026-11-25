"""Tests for the reflector agent."""
import pytest
from agents.reflector import reflect, should_replan, apply_replan_strategy


def _make_profile(**overrides):
    base = {
        "shape": {"rows": 500, "cols": 10},
        "is_classification": True,
        "imbalance_ratio": 1.5,
        "notes": [],
    }
    base.update(overrides)
    return base


def _make_eval(**overrides):
    base = {
        "model": "RandomForest",
        "accuracy": 0.85,
        "balanced_accuracy": 0.83,
        "f1_macro": 0.82,
        "precision_macro": 0.81,
        "recall_macro": 0.80,
        "train_accuracy": 0.88,
    }
    base.update(overrides)
    return base


def _make_all_metrics(best_eval):
    dummy = {
        "model": "DummyMostFrequent",
        "accuracy": 0.50,
        "balanced_accuracy": 0.50,
        "f1_macro": 0.33,
    }
    return [best_eval, dummy]


def test_reflect_good_results_no_issues():
    ev = _make_eval()
    ref = reflect(_make_profile(), ev, _make_all_metrics(ev))
    assert ref["status"] == "ok"
    assert ref["n_issues"] == 0
    assert ref["replan_recommended"] is False


def test_reflect_low_f1_flags_issue():
    ev = _make_eval(f1_macro=0.35, balanced_accuracy=0.40, accuracy=0.45, train_accuracy=0.48)
    ref = reflect(_make_profile(), ev, _make_all_metrics(ev))
    assert ref["n_issues"] > 0
    assert any("F1" in i for i in ref["issues"])


def test_reflect_overfitting_detected():
    ev = _make_eval(train_accuracy=0.98, accuracy=0.60)
    ref = reflect(_make_profile(), ev, _make_all_metrics(ev))
    assert any("overfitting" in i.lower() for i in ref["issues"])


def test_reflect_majority_class_bias():
    ev = _make_eval(accuracy=0.90, balanced_accuracy=0.55)
    ref = reflect(_make_profile(), ev, _make_all_metrics(ev))
    assert any("bias" in i.lower() or "majority" in i.lower() for i in ref["issues"])


def test_reflect_regression_mode():
    profile = _make_profile(is_classification=False)
    ev = {"model": "Ridge", "r2": 0.75, "rmse": 10.0, "mae": 8.0, "train_r2": 0.78}
    all_m = [ev, {"model": "DummyMean", "r2": 0.0, "rmse": 30.0, "mae": 25.0}]
    ref = reflect(profile, ev, all_m)
    assert ref["status"] == "ok"


def test_reflect_regression_low_r2():
    profile = _make_profile(is_classification=False)
    ev = {"model": "Ridge", "r2": 0.10, "rmse": 50.0, "mae": 40.0, "train_r2": 0.12}
    all_m = [ev, {"model": "DummyMean", "r2": 0.0}]
    ref = reflect(profile, ev, all_m)
    assert ref["n_issues"] > 0


def test_should_replan_when_recommended():
    ref = {"replan_recommended": True, "n_issues": 2}
    assert should_replan(ref) is True


def test_should_not_replan_when_ok():
    ref = {"replan_recommended": False, "n_issues": 0}
    assert should_replan(ref) is False


def test_apply_replan_adds_steps():
    steps = ["profile_dataset", "build_preprocessor", "select_models", "train_models", "evaluate", "reflect"]
    profile = _make_profile()
    reflection = {
        "issues": ["Overfitting detected: train-test gap=0.200."],
        "suggestions": ["Reduce model complexity, add regularisation."],
    }
    new_steps, new_profile = apply_replan_strategy(steps, profile, reflection)
    assert "apply_regularisation" in new_steps
    assert new_profile.get("force_regularisation") is True


def test_apply_replan_does_not_duplicate_steps():
    steps = ["profile_dataset", "apply_regularisation", "train_models", "reflect"]
    profile = _make_profile()
    reflection = {
        "issues": ["Overfitting detected."],
        "suggestions": ["Add regularisation."],
    }
    new_steps, _ = apply_replan_strategy(steps, profile, reflection)
    assert new_steps.count("apply_regularisation") == 1


def test_reflect_barely_beats_baseline():
    ev = _make_eval(balanced_accuracy=0.52, f1_macro=0.60, accuracy=0.55, train_accuracy=0.56)
    dummy = {"model": "DummyMostFrequent", "balanced_accuracy": 0.51, "f1_macro": 0.33, "accuracy": 0.50}
    ref = reflect(_make_profile(), ev, [ev, dummy])
    assert any("baseline" in i.lower() or "barely" in i.lower() for i in ref["issues"])


def test_reflect_marginal_improvement():
    ev = _make_eval(balanced_accuracy=0.54, f1_macro=0.65, accuracy=0.58, train_accuracy=0.60)
    dummy = {"model": "DummyMostFrequent", "balanced_accuracy": 0.50, "f1_macro": 0.33, "accuracy": 0.50}
    ref = reflect(_make_profile(), ev, [ev, dummy])
    assert any("marginal" in i.lower() for i in ref["issues"])


def test_reflect_mild_overfitting():
    ev = _make_eval(train_accuracy=0.93, accuracy=0.83)
    ref = reflect(_make_profile(), ev, _make_all_metrics(ev))
    assert any("mild" in i.lower() or "overfitting" in i.lower() for i in ref["issues"])


def test_reflect_imbalanced_low_f1():
    ev = _make_eval(f1_macro=0.45, balanced_accuracy=0.50, accuracy=0.70, train_accuracy=0.72)
    profile = _make_profile(imbalance_ratio=5.0)
    ref = reflect(profile, ev, _make_all_metrics(ev))
    assert any("imbalanced" in i.lower() for i in ref["issues"])


def test_reflect_model_diversity_warning():
    ev1 = _make_eval(f1_macro=0.55, balanced_accuracy=0.55, accuracy=0.55, train_accuracy=0.57)
    ev2 = {"model": "LR", "f1_macro": 0.54, "balanced_accuracy": 0.54, "accuracy": 0.54}
    ev3 = {"model": "KNN", "f1_macro": 0.55, "balanced_accuracy": 0.55, "accuracy": 0.55}
    ref = reflect(_make_profile(), ev1, [ev1, ev2, ev3])
    assert any("similarly" in i.lower() for i in ref["issues"])


def test_reflect_small_dataset_high_score():
    ev = _make_eval(f1_macro=0.95, balanced_accuracy=0.95, accuracy=0.95, train_accuracy=0.96)
    profile = _make_profile(shape={"rows": 100, "cols": 5})
    ref = reflect(profile, ev, _make_all_metrics(ev))
    assert any("small" in i.lower() or "generalise" in i.lower() for i in ref["issues"])


def test_reflect_regression_negative_r2():
    profile = _make_profile(is_classification=False)
    ev = {"model": "Ridge", "r2": -0.5, "rmse": 50.0, "mae": 40.0, "train_r2": 0.1}
    all_m = [ev, {"model": "DummyMean", "r2": 0.0}]
    ref = reflect(profile, ev, all_m)
    assert any("negative" in i.lower() for i in ref["issues"])


def test_reflect_regression_overfitting():
    profile = _make_profile(is_classification=False)
    ev = {"model": "RF", "r2": 0.40, "rmse": 20.0, "mae": 15.0, "train_r2": 0.95}
    all_m = [ev, {"model": "DummyMean", "r2": 0.0}]
    ref = reflect(profile, ev, all_m)
    assert any("overfitting" in i.lower() for i in ref["issues"])


def test_reflect_regression_barely_beats_baseline():
    profile = _make_profile(is_classification=False)
    ev = {"model": "Ridge", "r2": 0.03, "rmse": 30.0, "mae": 25.0, "train_r2": 0.05}
    dummy = {"model": "DummyMean", "r2": 0.0, "rmse": 31.0, "mae": 26.0}
    ref = reflect(profile, ev, [ev, dummy])
    assert any("baseline" in i.lower() or "barely" in i.lower() for i in ref["issues"])


def test_apply_replan_baseline_adds_ensemble():
    steps = ["profile_dataset", "train_models", "evaluate", "reflect"]
    profile = _make_profile()
    reflection = {
        "issues": ["Best model barely beats baseline."],
        "suggestions": ["Investigate feature quality."],
    }
    new_steps, new_profile = apply_replan_strategy(steps, profile, reflection)
    assert new_profile.get("try_ensemble") is True


def test_apply_replan_similar_models_adds_feature_eng():
    steps = ["profile_dataset", "build_preprocessor", "train_models", "reflect"]
    profile = _make_profile()
    reflection = {
        "issues": ["All models perform similarly."],
        "suggestions": ["Focus on feature engineering and preprocessing."],
    }
    new_steps, new_profile = apply_replan_strategy(steps, profile, reflection)
    assert new_profile.get("force_feature_engineering") is True


def test_apply_replan_fallback_increases_diversity():
    steps = ["profile_dataset", "train_models", "reflect"]
    profile = _make_profile()
    reflection = {
        "issues": ["Some unknown issue."],
        "suggestions": ["Some unknown suggestion."],
    }
    new_steps, new_profile = apply_replan_strategy(steps, profile, reflection)
    assert new_profile.get("increase_model_diversity") is True
