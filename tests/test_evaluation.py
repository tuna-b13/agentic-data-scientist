"""Tests for the evaluation tool."""
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from tools.evaluation import (
    evaluate_best, plot_model_comparison, plot_feature_importance,
    _format_best_metrics, _per_class_analysis, write_markdown_report, save_json,
)


@pytest.fixture
def clf_training_payload():
    np.random.seed(42)
    y_test = pd.Series(np.random.choice(["A", "B"], 50))
    y_pred = pd.Series(np.random.choice(["A", "B"], 50))
    y_train = pd.Series(np.random.choice(["A", "B"], 150))

    pipe = MagicMock()
    pipe.named_steps = {"preprocess": MagicMock(), "model": MagicMock()}
    pipe.named_steps["preprocess"].get_feature_names_out.side_effect = Exception("mock")

    return {
        "best": {
            "name": "RandomForest",
            "pipeline": pipe,
            "metrics": {
                "model": "RandomForest",
                "accuracy": 0.80,
                "balanced_accuracy": 0.78,
                "f1_macro": 0.77,
                "precision_macro": 0.76,
                "recall_macro": 0.75,
                "train_accuracy": 0.85,
            },
            "y_test": y_test,
            "y_pred": y_pred,
        },
        "all_metrics": [
            {"model": "RandomForest", "accuracy": 0.80, "balanced_accuracy": 0.78, "f1_macro": 0.77},
            {"model": "DummyMostFrequent", "accuracy": 0.55, "balanced_accuracy": 0.50, "f1_macro": 0.35},
        ],
        "is_classification": True,
    }


def test_evaluate_best_classification(clf_training_payload, tmp_path):
    result = evaluate_best(clf_training_payload, output_dir=str(tmp_path))
    assert "best_metrics" in result
    assert "all_metrics" in result
    assert "confusion_matrix_path" in result
    assert "classification_report" in result
    assert os.path.exists(result["confusion_matrix_path"])
    assert os.path.exists(result["model_comparison_path"])


def test_evaluate_best_has_per_class(clf_training_payload, tmp_path):
    result = evaluate_best(clf_training_payload, output_dir=str(tmp_path))
    assert "per_class_performance" in result
    assert len(result["per_class_performance"]) > 0


def test_save_json(tmp_path):
    path = str(tmp_path / "test.json")
    save_json(path, {"key": "value"})
    assert os.path.exists(path)


def test_plot_model_comparison(tmp_path):
    metrics = [
        {"model": "RF", "balanced_accuracy": 0.8, "f1_macro": 0.75, "accuracy": 0.82},
        {"model": "LR", "balanced_accuracy": 0.7, "f1_macro": 0.68, "accuracy": 0.72},
    ]
    out = str(tmp_path / "comp.png")
    plot_model_comparison(metrics, out)
    assert os.path.exists(out)


def test_plot_model_comparison_regression(tmp_path):
    metrics = [
        {"model": "RF", "r2": 0.8, "mae": 5.0},
        {"model": "Ridge", "r2": 0.6, "mae": 8.0},
    ]
    out = str(tmp_path / "comp_reg.png")
    plot_model_comparison(metrics, out)
    assert os.path.exists(out)


def test_plot_feature_importance(tmp_path):
    importance = {"feat_a": 0.35, "feat_b": 0.25, "feat_c": 0.15, "feat_d": 0.10}
    out = str(tmp_path / "imp.png")
    plot_feature_importance(importance, out)
    assert os.path.exists(out)


def test_plot_feature_importance_empty(tmp_path):
    out = str(tmp_path / "imp_empty.png")
    plot_feature_importance({}, out)
    assert not os.path.exists(out)


def test_per_class_analysis():
    y_test = pd.Series(["A", "B", "A", "B", "A", "B"])
    y_pred = pd.Series(["A", "B", "A", "A", "B", "B"])
    result = _per_class_analysis(y_test, y_pred)
    assert len(result) == 2
    assert all(k in result[0] for k in ["class", "precision", "recall", "f1", "support"])


def test_format_best_metrics_classification():
    best = {"accuracy": 0.85, "balanced_accuracy": 0.82, "f1_macro": 0.80,
            "precision_macro": 0.79, "recall_macro": 0.78, "train_accuracy": 0.90}
    text = _format_best_metrics(best, is_clf=True)
    assert "Accuracy" in text
    assert "F1" in text


def test_format_best_metrics_regression():
    best = {"r2": 0.75, "rmse": 10.0, "mae": 8.0, "train_r2": 0.80}
    text = _format_best_metrics(best, is_clf=False)
    assert "R2" in text
    assert "RMSE" in text


def test_evaluate_best_regression(tmp_path):
    np.random.seed(42)
    y_test = pd.Series(np.random.normal(100, 10, 50))
    y_pred = y_test + np.random.normal(0, 5, 50)

    pipe = MagicMock()
    pipe.named_steps = {"preprocess": MagicMock(), "model": MagicMock()}
    pipe.named_steps["preprocess"].get_feature_names_out.side_effect = Exception("mock")

    payload = {
        "best": {
            "name": "Ridge",
            "pipeline": pipe,
            "metrics": {"model": "Ridge", "r2": 0.75, "rmse": 5.0, "mae": 4.0, "train_r2": 0.80},
            "y_test": y_test,
            "y_pred": y_pred,
        },
        "all_metrics": [
            {"model": "Ridge", "r2": 0.75, "mae": 4.0},
            {"model": "DummyMean", "r2": 0.0, "mae": 10.0},
        ],
        "is_classification": False,
    }
    result = evaluate_best(payload, output_dir=str(tmp_path))
    assert result["confusion_matrix_path"] is None
    assert result["classification_report"] is None
    assert result["is_classification"] is False


def test_write_markdown_report(tmp_path):
    from dataclasses import dataclass

    @dataclass
    class FakeCtx:
        run_id: str = "test_run"
        started_at: str = "2026-01-01T00:00:00Z"
        data_path: str = "data/test.csv"
        target: str = "target"

    ctx = FakeCtx()
    profile = {
        "shape": {"rows": 100, "cols": 5},
        "is_classification": True,
        "imbalance_ratio": 1.5,
        "total_missing_pct": 2.0,
        "duplicate_pct": 0,
        "highly_skewed_cols": [],
        "feature_types": {"numeric": ["a", "b"], "categorical": ["c"]},
        "notes": ["Test note"],
    }
    plan = {
        "steps": ["profile_dataset", "train_models", "evaluate"],
        "rationale": {"template_default": "Standard dataset."},
        "hints": {},
    }
    eval_payload = {
        "best_metrics": {"model": "RF", "accuracy": 0.85, "balanced_accuracy": 0.82,
                         "f1_macro": 0.80, "precision_macro": 0.79, "recall_macro": 0.78,
                         "train_accuracy": 0.88},
        "all_metrics": [{"model": "RF", "accuracy": 0.85}],
        "is_classification": True,
        "per_class_performance": [
            {"class": "A", "precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
        ],
        "confusion_matrix_path": "cm.png",
        "model_comparison_path": "comp.png",
        "feature_importance_path": None,
    }
    reflection = {"status": "ok", "issues": [], "suggestions": [], "replan_recommended": False}

    out = str(tmp_path / "report.md")
    write_markdown_report(out, ctx, "fp_test", profile, plan, eval_payload, reflection)
    assert os.path.exists(out)
    content = open(out).read()
    assert "RF" in content
    assert "Rationale" in content
