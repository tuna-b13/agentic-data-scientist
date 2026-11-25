"""Tests for the data profiler tool."""
import pandas as pd
import numpy as np
import pytest
from tools.data_profiler import profile_dataset, infer_target_column, is_classification_target, dataset_fingerprint


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
        "x1": np.random.normal(0, 1, 300),
        "x2": np.random.normal(5, 2, 300),
        "price": np.random.normal(100, 30, 300),
    })


def test_infer_target_finds_target_column(clf_df):
    result = infer_target_column(clf_df)
    assert result == "target"


def test_infer_target_falls_back_to_last_col():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "label": ["x", "y", "x"]})
    assert infer_target_column(df) == "label"


def test_is_classification_target_categorical():
    s = pd.Series(["a", "b", "a", "c"])
    assert is_classification_target(s) is True


def test_is_classification_target_numeric_few_unique():
    s = pd.Series([0, 1, 0, 1, 0, 1])
    assert is_classification_target(s) is True


def test_is_classification_target_continuous():
    s = pd.Series(np.random.normal(0, 1, 200))
    assert is_classification_target(s) is False


def test_profile_returns_required_keys(clf_df):
    profile = profile_dataset(clf_df, "target")
    required = ["shape", "feature_types", "is_classification", "total_missing_pct",
                 "imbalance_ratio", "duplicate_pct", "id_columns", "constant_columns"]
    for key in required:
        assert key in profile, f"Missing key: {key}"


def test_profile_shape_correct(clf_df):
    profile = profile_dataset(clf_df, "target")
    assert profile["shape"]["rows"] == 200
    assert profile["shape"]["cols"] == 4


def test_profile_detects_classification(clf_df):
    profile = profile_dataset(clf_df, "target")
    assert profile["is_classification"] is True


def test_profile_detects_regression(reg_df):
    profile = profile_dataset(reg_df, "price")
    assert profile["is_classification"] is False


def test_profile_numeric_features(clf_df):
    profile = profile_dataset(clf_df, "target")
    assert "age" in profile["feature_types"]["numeric"]
    assert "income" in profile["feature_types"]["numeric"]


def test_profile_categorical_features(clf_df):
    profile = profile_dataset(clf_df, "target")
    assert "city" in profile["feature_types"]["categorical"]


def test_fingerprint_deterministic(clf_df):
    fp1 = dataset_fingerprint(clf_df, "target")
    fp2 = dataset_fingerprint(clf_df, "target")
    assert fp1 == fp2


def test_fingerprint_changes_with_target(clf_df):
    fp1 = dataset_fingerprint(clf_df, "target")
    fp2 = dataset_fingerprint(clf_df, "age")
    assert fp1 != fp2


def test_profile_missing_data():
    df = pd.DataFrame({
        "a": [1, 2, None, 4, None],
        "b": [None, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
    })
    profile = profile_dataset(df, "target")
    assert profile["total_missing_pct"] > 0
