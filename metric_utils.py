# Metric Utils


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)


def classification_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    # ROC-AUC only if binary
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
    except Exception:
        pass

    return metrics


def regression_metrics(y_true, y_pred):
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred)
    }
