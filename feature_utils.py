import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def detect_id_columns(df: pd.DataFrame, threshold: float = 0.95):
    """Detect ID-like columns based on high uniqueness. Excludes floats."""
    id_cols = []
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) >= threshold * len(df):
            id_cols.append(col)
    return id_cols


def drop_low_variance_features(X: pd.DataFrame, threshold: float = 0.0):
    """Remove near-zero variance features."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    kept_columns = X.columns[selector.get_support()]
    return X[kept_columns]


def drop_highly_correlated_features(X: pd.DataFrame, threshold: float = 0.9):
    """Remove highly correlated numeric features."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop), to_drop


def detect_leakage_features(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95):
    """
    Detect leakage-prone features based on:
    - Identical values to target
    - Extremely high correlation with target
    """
    leakage_cols = []

    for col in X.columns:
        try:
            # identical column
            if X[col].equals(y):
                leakage_cols.append(col)
                continue

            # correlation-based leakage (numeric only)
            if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(y):
                corr = X[col].corr(y)
                if corr is not None and abs(corr) >= threshold:
                    leakage_cols.append(col)
        except Exception:
            continue

    return list(set(leakage_cols))
