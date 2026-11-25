"""Tests for the planner agent."""
import pytest
from agents.planner import create_plan, explain_plan


def _make_profile(**overrides):
    base = {
        "shape": {"rows": 500, "cols": 10},
        "is_classification": True,
        "imbalance_ratio": 1.5,
        "total_missing_pct": 2.0,
        "id_columns": [],
        "constant_columns": [],
        "high_cardinality_cols": [],
        "highly_skewed_cols": [],
        "high_outlier_cols": [],
        "weak_predictors": [],
        "duplicate_pct": 0,
        "n_classes": 3,
    }
    base.update(overrides)
    return base


def test_plan_returns_dict_with_required_keys():
    plan = create_plan(_make_profile())
    assert isinstance(plan, dict)
    assert "steps" in plan
    assert "rationale" in plan
    assert "hints" in plan


def test_small_dataset_template():
    plan = create_plan(_make_profile(shape={"rows": 200, "cols": 5}))
    assert plan["hints"]["plan_template"] == "small"
    assert plan["hints"]["cv_folds"] == 10
    assert "template_small" in plan["rationale"]


def test_large_dataset_template():
    plan = create_plan(_make_profile(shape={"rows": 60000, "cols": 20}))
    assert plan["hints"]["plan_template"] == "large"
    assert "template_large" in plan["rationale"]


def test_imbalanced_template():
    plan = create_plan(_make_profile(imbalance_ratio=5.0))
    assert plan["hints"]["plan_template"] == "imbalanced"
    assert "apply_class_balancing" in plan["steps"]


def test_high_missing_template():
    plan = create_plan(_make_profile(total_missing_pct=30.0))
    assert plan["hints"]["plan_template"] == "high_missing"
    assert "advanced_imputation" in plan["steps"]


def test_default_template():
    plan = create_plan(_make_profile(shape={"rows": 5000, "cols": 10}))
    assert plan["hints"]["plan_template"] == "default"


def test_skewed_cols_add_transform():
    plan = create_plan(_make_profile(highly_skewed_cols=["col1", "col2"]))
    assert "apply_skew_transform" in plan["steps"]
    assert "log_transform" in plan["rationale"]
    assert plan["hints"]["log_transform_cols"] == ["col1", "col2"]


def test_outlier_cols_add_clipping():
    plan = create_plan(_make_profile(high_outlier_cols=["col1"]))
    assert "clip_outliers" in plan["steps"]
    assert "clip_outliers" in plan["rationale"]


def test_id_columns_dropped():
    plan = create_plan(_make_profile(id_columns=["id_col"], has_id_cols=True))
    assert "id_col" in plan["hints"]["drop_columns"]


def test_weak_predictors_trigger_feature_selection():
    weak = [f"w{i}" for i in range(5)]
    plan = create_plan(_make_profile(weak_predictors=weak))
    assert "feature_selection" in plan["steps"]


def test_explain_plan_returns_string():
    plan = create_plan(_make_profile())
    text = explain_plan(plan)
    assert isinstance(text, str)
    assert "EXECUTION PLAN" in text


def test_memory_hint_prioritises_model():
    memory = {"best_model": "RandomForest", "best_metrics": {"f1_macro": 0.85}}
    plan = create_plan(_make_profile(), memory_hint=memory)
    assert "RandomForest" in plan["hints"]["priority_models"]


def test_steps_always_start_with_profile():
    plan = create_plan(_make_profile())
    assert plan["steps"][0] == "profile_dataset"


def test_steps_always_end_with_report():
    plan = create_plan(_make_profile())
    assert plan["steps"][-1] == "write_report"
