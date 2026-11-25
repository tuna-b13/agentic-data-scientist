"""
Data Profiler Tool
Generates a rich dataset profile used by the Planner to make autonomous decisions.
Extracts: shape, types, missing values, skewness, outliers, correlations,
duplicates, cardinality, class balance, and actionable signals.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic target inference based on common naming patterns and cardinality."""
    candidates = ["target", "label", "class", "y", "outcome"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(cat_cols) == 1:
        return cat_cols[0]
    if len(cat_cols) > 1:
        best = min(cat_cols, key=lambda c: df[c].nunique(dropna=True))
        if df[best].nunique(dropna=True) <= 50:
            return best

    last = df.columns[-1]
    uniq = df[last].nunique(dropna=True)
    n = len(df)
    if n > 0 and (uniq <= 50 or (uniq / max(n, 1) < 0.05)):
        return last

    if pd.api.types.is_numeric_dtype(df[last]):
        return last

    return None


def is_classification_target(series: pd.Series) -> bool:
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True
    return series.nunique(dropna=True) <= 50


def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = abs(hash(base)) % (10**12)
    return f"fp_{h}"


def _compute_skewness(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """Compute skewness for each numeric column."""
    result = {}
    for col in numeric_cols:
        try:
            result[col] = round(float(df[col].skew()), 3)
        except Exception:
            result[col] = 0.0
    return result


def _detect_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Detect outliers using IQR method for each numeric column."""
    result = {}
    for col in numeric_cols:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            count = int(outlier_mask.sum())
            pct = round(count / max(len(df), 1) * 100, 2)
            result[col] = {"count": count, "pct": pct, "lower": round(float(lower), 3), "upper": round(float(upper), 3)}
        except Exception:
            result[col] = {"count": 0, "pct": 0.0, "lower": 0.0, "upper": 0.0}
    return result


def _compute_target_correlations(df: pd.DataFrame, target: str, numeric_cols: List[str]) -> Dict[str, float]:
    """Compute absolute correlation of each numeric feature with the target."""
    result = {}
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return result
    for col in numeric_cols:
        if col == target:
            continue
        try:
            corr = df[col].corr(df[target])
            if pd.notna(corr):
                result[col] = round(float(abs(corr)), 3)
        except Exception:
            pass
    return result


def _detect_constant_columns(df: pd.DataFrame) -> List[str]:
    """Find columns with zero or near-zero variance."""
    return [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]


def _detect_high_cardinality(df: pd.DataFrame, cat_cols: List[str], threshold: int = 50) -> List[str]:
    """Identify categorical columns with too many unique values for one-hot encoding."""
    return [c for c in cat_cols if df[c].nunique(dropna=True) > threshold]


def _detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Heuristic: columns that look like IDs (all unique, or name contains 'id')."""
    result = []
    for c in df.columns:
        name_lower = str(c).lower()
        if name_lower in ("id", "index") or name_lower.endswith("_id"):
            result.append(str(c))
        elif df[c].nunique(dropna=True) == len(df) and df[c].dtype != float:
            result.append(str(c))
    return result


def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Generate a comprehensive dataset profile for the Planner agent.
    Returns a dictionary with shape, types, missing values, skewness,
    outliers, correlations, duplicates, cardinality, and signals.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    profile: Dict[str, Any] = {}

    # Basic shape and columns
    profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    # Missing values
    missing = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}
    profile["total_missing_pct"] = round(float(df.isna().mean().mean() * 100), 2)

    # Target info
    profile["target"] = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))

    # Feature types
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
    cat_cols = [c for c in X.columns.astype(str).tolist() if c not in numeric_cols]

    profile["feature_types"] = {"numeric": numeric_cols, "categorical": cat_cols}
    profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}

    # ID-like columns
    profile["id_columns"] = _detect_id_columns(df)

    # Constant columns
    profile["constant_columns"] = _detect_constant_columns(df)

    # High cardinality categoricals
    profile["high_cardinality_cols"] = _detect_high_cardinality(df, cat_cols)

    # Duplicates
    profile["duplicate_rows"] = int(df.duplicated().sum())
    profile["duplicate_pct"] = round(profile["duplicate_rows"] / max(len(df), 1) * 100, 2)

    # Skewness (numeric features)
    profile["skewness"] = _compute_skewness(df, numeric_cols)
    profile["highly_skewed_cols"] = [c for c, v in profile["skewness"].items() if abs(v) > 0.75]

    # Outliers (IQR)
    profile["outliers"] = _detect_outliers_iqr(df, numeric_cols)
    profile["high_outlier_cols"] = [c for c, v in profile["outliers"].items() if v["pct"] > 5.0]

    # Correlation with target
    profile["target_correlations"] = _compute_target_correlations(df, target, numeric_cols)
    sorted_corr = sorted(profile["target_correlations"].items(), key=lambda x: x[1], reverse=True)
    profile["top_predictors"] = [c for c, _ in sorted_corr[:5]]
    profile["weak_predictors"] = [c for c, v in sorted_corr if v < 0.05]

    # Class balance (classification only)
    if profile["is_classification"]:
        vc = y.value_counts(dropna=False)
        profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        profile["n_classes"] = int(len(vc))
        if len(vc) >= 2:
            ratio = float(vc.max() / max(vc.min(), 1))
        else:
            ratio = 1.0
        profile["imbalance_ratio"] = round(ratio, 3)
    else:
        profile["class_counts"] = None
        profile["n_classes"] = None
        profile["imbalance_ratio"] = None

    # Actionable notes / signals for the planner
    notes: List[str] = []

    if profile["shape"]["rows"] < 500:
        notes.append("Very small dataset (<500 rows): high overfitting risk, use simple models and strong CV.")
    elif profile["shape"]["rows"] < 1000:
        notes.append("Small dataset (<1000 rows): prefer simpler models, guard against overfitting.")
    elif profile["shape"]["rows"] > 50000:
        notes.append("Large dataset (>50k rows): consider skipping expensive models like SVC.")

    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 columns): watch one-hot expansion and overfitting.")

    if profile["total_missing_pct"] > 20:
        notes.append(f"Severe missing data ({profile['total_missing_pct']:.1f}%): consider advanced imputation.")
    elif profile["total_missing_pct"] > 0:
        notes.append(f"Missing data present ({profile['total_missing_pct']:.1f}%): imputation needed.")

    if profile["duplicate_pct"] > 5:
        notes.append(f"High duplicate rate ({profile['duplicate_pct']:.1f}%): consider dropping duplicates.")

    if profile["highly_skewed_cols"]:
        notes.append(f"Skewed features detected ({len(profile['highly_skewed_cols'])}): consider log/power transform.")

    if profile["high_outlier_cols"]:
        notes.append(f"Outlier-heavy features ({len(profile['high_outlier_cols'])}): consider clipping or robust scaling.")

    if profile["high_cardinality_cols"]:
        notes.append(f"High-cardinality categoricals: {profile['high_cardinality_cols']}. Use target/ordinal encoding instead of one-hot.")

    if profile["id_columns"]:
        notes.append(f"Potential ID columns detected: {profile['id_columns']}. Should be dropped before modelling.")

    if profile["constant_columns"]:
        notes.append(f"Constant columns (zero variance): {profile['constant_columns']}. Will be dropped.")

    if profile["is_classification"] and profile["imbalance_ratio"] and profile["imbalance_ratio"] >= 3.0:
        notes.append(f"Class imbalance detected (ratio={profile['imbalance_ratio']:.1f}): use balanced class weights or SMOTE.")

    if profile["is_classification"] and profile.get("n_classes", 0) and profile["n_classes"] > 10:
        notes.append(f"Many classes ({profile['n_classes']}): multi-class problem, prefer macro-averaged metrics.")

    if not profile["is_classification"]:
        notes.append("Non-classification target detected: will use regression models and metrics.")

    if profile["weak_predictors"]:
        notes.append(f"Weak predictors (corr<0.05): {profile['weak_predictors'][:5]}. May add noise.")

    profile["notes"] = notes

    return profile
