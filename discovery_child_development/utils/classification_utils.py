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


def create_average_metrics(Y_test, Y_pred, average="macro"):
    """Calculate and return a dictionary of average performance metrics for
    multilabel classification problems.

    This function computes a series of evaluation metrics for assessing the
    performance of a classification model on a multilabel task. It includes
    accuracy, precision, recall, F1-score, Hamming loss, and Jaccard score.
    The function allows specifying the averaging method to be used for
    multi-class calculations.

    Parameters:
    - Y_test (array-like): True labels of the data. Must be of the same length as Y_pred.
    - Y_pred (array-like): Predicted labels by the classification model. Must be of the
      same length as Y_test.
    - average (str, optional): A string specifying the averaging method to be applied for
      precision, recall, and F1-score. Default is 'macro'. Other valid options are
      'micro', 'weighted', and 'samples'.

    Returns:
    - dict: A dictionary with keys as metric names and values as the corresponding
      metric scores. The dictionary contains the following keys:
        - 'accuracy': The accuracy score of the model.
        - 'precision': The precision score of the model using the specified averaging method.
        - 'recall': The recall score of the model using the specified averaging method.
        - 'f1': The F1 score of the model using the specified averaging method.
        - 'hamming': The Hamming loss of the model.
        - 'jaccard': The Jaccard score of the model using the specified averaging method.

    Notes:
    - The 'macro' average is preferred for this function as it does not favor larger
      classes, which is particularly useful when the model's performance on
      underrepresented classes is important.
    """
    # Accuracy
    accuracy = accuracy_score(Y_test, Y_pred)

    # Precision
    precision = precision_score(Y_test, Y_pred, average=average)

    # Recall
    recall = recall_score(Y_test, Y_pred, average=average)

    # F1-Score
    f1 = f1_score(Y_test, Y_pred, average=average)

    # Hamming Loss
    hamming = hamming_loss(Y_test, Y_pred)

    # Jaccard Score
    jaccard = jaccard_score(Y_test, Y_pred, average=average)

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
