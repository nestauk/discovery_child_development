"""
Utils functions for training a classifier
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)


def add_binarise_labels(
    df: pd.DataFrame, label_column: str, not_valid_label: str = None
) -> pd.DataFrame:
    """Add label dummy columns to dataframe.

    Args:
        df: Dataframe to add dummy columns to.
        label_column: Column with labels to turn into dummy column.
            The label column must have values in a list.
        not_valid_label: Label that indicates that the record is
            not relevant or valid. If a record has this label,
            all of its other labels will be set to 0. The dummy
            column relating to this label will be removed.

    Returns:
        Dataframe with additional dummy label columns
    """
    mlb = MultiLabelBinarizer()
    dummy_cols = pd.DataFrame(
        mlb.fit_transform(df[label_column]), columns=mlb.classes_, index=df.index
    )

    if not_valid_label is not None:
        valid_cols = [col for col in dummy_cols.columns if col != not_valid_label]
        # Set all other labels to 0 if row has not valid label
        dummy_cols = dummy_cols[valid_cols].mask(dummy_cols[not_valid_label] == 1, 0)
    else:
        valid_cols = dummy_cols.columns

    return dummy_cols[valid_cols], mlb


def create_average_metrics(Y_test, Y_pred, average="samples"):
    # Accuracy
    accuracy = accuracy_score(Y_test, Y_pred)

    # Precision
    precision = precision_score(
        Y_test, Y_pred, average=average
    )  # Use 'samples' for multi-label

    # Recall
    recall = recall_score(
        Y_test, Y_pred, average=average
    )  # Use 'samples' for multi-label

    # F1-Score
    f1 = f1_score(Y_test, Y_pred, average=average)  # Use 'samples' for multi-label

    # Hamming Loss
    hamming = hamming_loss(Y_test, Y_pred)

    # Jaccard Score
    jaccard = jaccard_score(
        Y_test, Y_pred, average=average
    )  # Use 'samples' for multi-label

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hamming": hamming,
        "jaccard": jaccard,
    }

    return results


def flatten_classification_report(report):
    """
    Flatten a scikit learn classification report into a format that can be stored on wandb
    """
    flat_report = {}
    for class_label, metrics in report.items():
        if class_label not in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
            for metric, value in metrics.items():
                flat_report[f"class_{class_label}_{metric}"] = value
        else:
            flat_report[class_label] = report[class_label]
    return flat_report
