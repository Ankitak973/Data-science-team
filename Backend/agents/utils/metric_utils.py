# Metric Utils


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)


def classification_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    # ROC-AUC only if binary and y_true has at least two classes
    try:
        if len(set(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except Exception:
        pass

    return metrics


def regression_metrics(y_true, y_pred):
    import numpy as np
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
