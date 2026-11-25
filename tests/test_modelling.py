"""Tests for the modelling tool."""
import pandas as pd
import numpy as np
import pytest
from tools.modelling import build_preprocessor, select_models, train_models


@pytest.fixture
def clf_profile():
    return {
        "shape": {"rows": 200, "cols": 5},
        "is_classification": True,
        "imbalance_ratio": 1.2,
        "feature_types": {"numeric": ["age", "income"], "categorical": ["city"]},
        "id_columns": [],
        "constant_columns": [],
        "high_cardinality_cols": [],
        "highly_skewed_cols": [],
        "high_outlier_cols": [],
        "duplicate_pct": 0,
    }


@pytest.fixture
def reg_profile():
    return {
        "shape": {"rows": 200, "cols": 4},
        "is_classification": False,
        "imbalance_ratio": None,
        "feature_types": {"numeric": ["x1", "x2", "x3"], "categorical": []},
        "id_columns": [],
        "constant_columns": [],
        "high_cardinality_cols": [],
        "highly_skewed_cols": [],
        "high_outlier_cols": [],
        "duplicate_pct": 0,
    }


@pytest.fixture
def clf_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(18, 80, 200),
        "income": np.random.normal(50000, 15000, 200),
        "city": np.random.choice(["A", "B", "C"], 200),
        "target": np.random.choice(["yes", "no"], 200),
    })


@pytest.fixture
def reg_df():
    np.random.seed(42)
    return pd.DataFrame({
        "x1": np.random.normal(0, 1, 200),
        "x2": np.random.normal(5, 2, 200),
        "x3": np.random.normal(10, 3, 200),
        "price": np.random.normal(100, 30, 200),
    })


def test_build_preprocessor_returns_transformer(clf_profile):
    ct = build_preprocessor(clf_profile)
    assert hasattr(ct, "fit_transform")


def test_select_models_classification(clf_profile):
    candidates = select_models(clf_profile, seed=42)
    names = [n for n, _ in candidates]
    assert "DummyMostFrequent" in names
    assert "LogisticRegression" in names
    assert "RandomForest" in names
    assert len(candidates) >= 3


def test_select_models_regression(reg_profile):
    candidates = select_models(reg_profile, seed=42)
    names = [n for n, _ in candidates]
    assert "DummyMean" in names
    assert "Ridge" in names
    assert "RandomForest" in names


def test_train_models_classification(clf_df, clf_profile, tmp_path):
    preprocessor = build_preprocessor(clf_profile)
    candidates = select_models(clf_profile, seed=42)
    result = train_models(
        df=clf_df, target="target", preprocessor=preprocessor,
        candidates=candidates, seed=42, test_size=0.2,
        output_dir=str(tmp_path), profile=clf_profile,
    )
    assert "best" in result
    assert "all_metrics" in result
    assert result["is_classification"] is True
    best_metrics = result["best"]["metrics"]
    assert "accuracy" in best_metrics
    assert "f1_macro" in best_metrics
    assert "balanced_accuracy" in best_metrics


def test_train_models_regression(reg_df, reg_profile, tmp_path):
    preprocessor = build_preprocessor(reg_profile)
    candidates = select_models(reg_profile, seed=42)
    result = train_models(
        df=reg_df, target="price", preprocessor=preprocessor,
        candidates=candidates, seed=42, test_size=0.2,
        output_dir=str(tmp_path), profile=reg_profile,
    )
    assert result["is_classification"] is False
    best_metrics = result["best"]["metrics"]
    assert "r2" in best_metrics
    assert "rmse" in best_metrics


def test_error_isolation_does_not_crash(clf_df, clf_profile, tmp_path):
    """A broken model should not crash the entire training."""
    preprocessor = build_preprocessor(clf_profile)

    class BrokenModel:
        def fit(self, X, y):
            raise ValueError("Intentional failure")
        def predict(self, X):
            raise ValueError("Intentional failure")

    candidates = [
        ("DummyMostFrequent", __import__("sklearn.dummy", fromlist=["DummyClassifier"]).DummyClassifier(strategy="most_frequent")),
        ("BrokenModel", BrokenModel()),
    ]
    result = train_models(
        df=clf_df, target="target", preprocessor=preprocessor,
        candidates=candidates, seed=42, test_size=0.2,
        output_dir=str(tmp_path), profile=clf_profile,
    )
    assert result["n_failed"] == 1
    assert result["best"]["name"] == "DummyMostFrequent"


def test_train_models_best_is_not_failed(clf_df, clf_profile, tmp_path):
    preprocessor = build_preprocessor(clf_profile)
    candidates = select_models(clf_profile, seed=42)
    result = train_models(
        df=clf_df, target="target", preprocessor=preprocessor,
        candidates=candidates, seed=42, test_size=0.2,
        output_dir=str(tmp_path), profile=clf_profile,
    )
    assert result["best"]["pipeline"] is not None
