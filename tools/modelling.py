"""
Modelling Tool
Builds preprocessing pipelines, selects candidate models, and trains them.
Supports both classification and regression tasks.
Adapts to dataset profile signals: size, imbalance, skewness, outliers,
high-cardinality features, and replan flags from the reflector.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

# Classification models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def build_preprocessor(profile: Dict[str, Any]) -> ColumnTransformer:
    """Build a preprocessing pipeline adapted to the dataset profile."""
    num_cols = list(profile["feature_types"]["numeric"])
    cat_cols = list(profile["feature_types"]["categorical"])
    high_card = profile.get("high_cardinality_cols", [])
    id_cols = profile.get("id_columns", [])

    drop_cols = set(id_cols + profile.get("constant_columns", []))
    num_cols = [c for c in num_cols if c not in drop_cols]
    cat_cols = [c for c in cat_cols if c not in drop_cols]

    low_card_cats = [c for c in cat_cols if c not in high_card]
    high_card_cats = [c for c in cat_cols if c in high_card]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    high_card_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = [("num", numeric_transformer, num_cols)]
    if low_card_cats:
        transformers.append(("cat", categorical_transformer, low_card_cats))
    if high_card_cats:
        transformers.append(("high_cat", high_card_transformer, high_card_cats))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# Model selection

def select_models(profile: Dict[str, Any], seed: int = 42) -> List[Tuple[str, Any]]:
    """Select candidate models based on dataset characteristics and task type."""
    is_clf = profile.get("is_classification", True)
    if is_clf:
        return _select_classification_models(profile, seed)
    else:
        return _select_regression_models(profile, seed)


def _select_classification_models(profile: Dict[str, Any], seed: int) -> List[Tuple[str, Any]]:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    imb = float(profile.get("imbalance_ratio") or 1.0)
    class_weight = "balanced" if imb >= 3.0 else None
    try_ensemble = profile.get("try_ensemble", False)
    force_reg = profile.get("force_regularisation", False)
    increase_diversity = profile.get("increase_model_diversity", False)

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
    ]

    lr_c = 0.1 if force_reg else 1.0
    candidates.append((
        "LogisticRegression",
        LogisticRegression(max_iter=2000, C=lr_c, class_weight=class_weight, random_state=seed),
    ))

    n_trees = 100 if force_reg else (200 if rows < 5000 else 300)
    max_depth = 5 if force_reg else (10 if rows < 1000 else None)
    candidates.append((
        "RandomForest",
        RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth,
            random_state=seed, n_jobs=-1, class_weight=class_weight,
        ),
    ))

    if rows <= 50000:
        lr_gb = 0.05 if force_reg else 0.1
        gb_depth = 2 if force_reg else 3
        gb_trees = 100 if force_reg else 200
        candidates.append((
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=gb_trees, learning_rate=lr_gb, max_depth=gb_depth, random_state=seed,
            ),
        ))

    if rows <= 20000:
        k = 3 if rows < 500 else 5
        candidates.append(("KNN", KNeighborsClassifier(n_neighbors=k, n_jobs=-1)))

    if rows <= 10000 and cols <= 100:
        candidates.append((
            "SVC_RBF",
            SVC(kernel="rbf", class_weight=class_weight, random_state=seed),
        ))

    if increase_diversity or try_ensemble:
        candidates.append((
            "AdaBoost",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2),
                n_estimators=100, random_state=seed,
            ),
        ))

    if try_ensemble and rows <= 50000:
        voting_estimators = [
            ("lr", LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=seed, class_weight=class_weight)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=seed)),
        ]
        candidates.append((
            "VotingEnsemble",
            VotingClassifier(estimators=voting_estimators, voting="soft"),
        ))

    return candidates


def _select_regression_models(profile: Dict[str, Any], seed: int) -> List[Tuple[str, Any]]:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    force_reg = profile.get("force_regularisation", False)

    candidates: List[Tuple[str, Any]] = [
        ("DummyMean", DummyRegressor(strategy="mean")),
    ]

    alpha = 10.0 if force_reg else 1.0
    candidates.append(("Ridge", Ridge(alpha=alpha, random_state=seed)))
    candidates.append(("Lasso", Lasso(alpha=0.1 if force_reg else 0.01, random_state=seed, max_iter=5000)))

    n_trees = 100 if force_reg else (200 if rows < 5000 else 300)
    max_depth = 5 if force_reg else (10 if rows < 1000 else None)
    candidates.append((
        "RandomForest",
        RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=seed, n_jobs=-1),
    ))

    if rows <= 50000:
        lr_gb = 0.05 if force_reg else 0.1
        gb_depth = 2 if force_reg else 3
        gb_trees = 100 if force_reg else 200
        candidates.append((
            "GradientBoosting",
            GradientBoostingRegressor(n_estimators=gb_trees, learning_rate=lr_gb, max_depth=gb_depth, random_state=seed),
        ))

    if rows <= 20000:
        k = 3 if rows < 500 else 5
        candidates.append(("KNN", KNeighborsRegressor(n_neighbors=k, n_jobs=-1)))

    if rows <= 10000 and cols <= 100:
        candidates.append(("SVR_RBF", SVR(kernel="rbf")))

    return candidates


# Preprocessing

def _preprocess_dataframe(df: pd.DataFrame, profile: Dict[str, Any], target: str) -> pd.DataFrame:
    """Apply data-level preprocessing before modelling."""
    df = df.copy()

    drop_cols = set(profile.get("id_columns", []) + profile.get("constant_columns", []))
    drop_cols.discard(target)
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    if profile.get("duplicate_pct", 0) > 5:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if after < before:
            print(f"[Modelling] Dropped {before - after} duplicate rows.")

    for col in profile.get("high_outlier_cols", []):
        if col in df.columns and col != target:
            info = profile.get("outliers", {}).get(col, {})
            lower, upper = info.get("lower"), info.get("upper")
            if lower is not None and upper is not None:
                df[col] = df[col].clip(lower=lower, upper=upper)

    for col in profile.get("highly_skewed_cols", []):
        if col in df.columns and col != target:
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])

    return df


# Training

def _compute_classification_metrics(y_true, y_pred, y_train, y_pred_train) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
    }


def _compute_regression_metrics(y_true, y_pred, y_train, y_pred_train) -> Dict[str, Any]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "train_r2": float(r2_score(y_train, y_pred_train)),
    }


def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
    profile: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Train all candidate models, evaluate on test set, return ranked results."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    is_clf = profile.get("is_classification", True) if profile else True

    if profile:
        df = _preprocess_dataframe(df, profile, target)

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # For classification: ensure target is string type for consistency
    if is_clf and y.nunique() <= 50:
        y = y.astype(str)

    # Stratify only for classification
    if is_clf:
        stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify,
    )

    results: List[Dict[str, Any]] = []
    use_cv = profile is not None and profile.get("shape", {}).get("rows", 0) < 1000

    for name, model in candidates:
        if verbose:
            print(f"[Modelling] Training: {name}")

        try:
            pipe = Pipeline(steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_pred_train = pipe.predict(X_train)

            if is_clf:
                metrics = _compute_classification_metrics(y_test, y_pred, y_train, y_pred_train)
            else:
                metrics = _compute_regression_metrics(y_test, y_pred, y_train, y_pred_train)

            metrics["model"] = name

            # Cross-validation for small datasets
            if use_cv and "Dummy" not in name:
                try:
                    if is_clf:
                        n_splits = min(5, y_train.value_counts().min())
                        if n_splits >= 2:
                            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                            scoring = "f1_macro"
                    else:
                        n_splits = min(5, len(y_train) // 5)
                        if n_splits >= 2:
                            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                            scoring = "r2"

                    if n_splits >= 2:
                        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
                        metrics["cv_score_mean"] = float(np.mean(cv_scores))
                        metrics["cv_score_std"] = float(np.std(cv_scores))
                except Exception:
                    pass

            results.append({
                "name": name,
                "pipeline": pipe,
                "metrics": metrics,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
            })

        except Exception as e:
            if verbose:
                print(f"[Modelling] FAILED: {name} — {e}")
            results.append({
                "name": name,
                "pipeline": None,
                "metrics": {"model": name, "error": str(e)},
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": None,
            })

    successful = [r for r in results if r["pipeline"] is not None]
    failed = [r for r in results if r["pipeline"] is None]

    if not successful:
        raise RuntimeError(f"All {len(candidates)} models failed to train.")

    if is_clf:
        successful.sort(key=lambda r: (r["metrics"].get("balanced_accuracy", 0), r["metrics"].get("f1_macro", 0)), reverse=True)
    else:
        successful.sort(key=lambda r: r["metrics"].get("r2", -999), reverse=True)

    ordered = successful + failed

    return {
        "results": ordered,
        "best": successful[0],
        "all_metrics": [r["metrics"] for r in ordered],
        "is_classification": is_clf,
        "n_failed": len(failed),
    }
