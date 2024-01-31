"""
Utils functions for training a classifier
"""

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    multilabel_confusion_matrix,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)
import seaborn as sns
from typing import Tuple


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


def plot_confusion_matrix(y_true, y_pred, label_index, label_name):
    """
    Plot the confusion matrix for a single label.

    Parameters:
    y_true (array-like): True binary labels in binary indicator format for a single label.
    y_pred (array-like): Binary labels predicted by the classifier for a single label.
    label_index (int): Index of the label for which to plot the confusion matrix.
    label_name (str): Name of the label.
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true[:, label_index], y_pred[:, label_index])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {label_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show(block=False)

    # Create a DataFrame with the data
    confusion_matrix_values = [
        "True Negatives",
        "False Negatives",
        "False Positives",
        "True Positives",
    ]
    df = pd.DataFrame(
        {
            "confusion_matrix": confusion_matrix_values,
            "values": list(itertools.chain.from_iterable(cm)),
        },
    )
    return df


def plot_roc_curve(y_true, y_pred_proba, label_index, label_name):
    """
    Plot the ROC curve for a single label.

    Parameters:
    y_true (array-like): True binary labels in binary indicator format for a single label.
    y_pred_proba (array-like): Probabilities of the positive class predicted by the classifier for a single label.
    label_index (int): Index of the label for which to plot the ROC curve.
    label_name (str): Name of the label.
    """
    fpr, tpr, _ = roc_curve(y_true[:, label_index], y_pred_proba[:, label_index])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f}) for {label_name}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic for {label_name}")
    plt.legend(loc="lower right")
    plt.show()


def evaluate_model_performance(y_true, y_pred, y_pred_proba, label_names):
    """
    Evaluate the model's performance for multi-label classification.

    Parameters:
    y_true (array-like): True binary labels in binary indicator format.
    y_pred (array-like): Binary labels predicted by the classifier.
    y_pred_proba (array-like): Probabilities of the positive class predicted by the classifier.
    label_names (list): List of label names corresponding to the columns of y_true and y_pred.
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for i, label_name in enumerate(label_names):
        plot_confusion_matrix(y_true, y_pred, i, label_name)
        plot_roc_curve(y_true, y_pred_proba, i, label_name)


# Example usage:
# evaluate_model_performance(y_true, y_pred, y_pred_proba, label_names)


def create_confusion_matrix(y_true, y_pred, label_names, proportions=False):
    # Calculate confusion matrices for each label
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    # Initialize the dataframe that will hold true negatives, false positives, false negatives, true positives
    data = []

    # Populate the dataframe with values for each label
    for i, label in enumerate(label_names):
        tn, fp, fn, tp = mcm[i].ravel()
        if proportions:
            total = np.sum(mcm[i])
            values = [tn / total, fp / total, fn / total, tp / total]
        else:
            values = [tn, fp, fn, tp]
        data.append(values)

    # Create a DataFrame with the data
    df = pd.DataFrame(
        data,
        index=label_names,
        columns=[
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
        ],
    ).reset_index()
    return df


def create_heatmap_table(y_true, y_pred, label_names, proportions=False):
    df = create_confusion_matrix(y_true, y_pred, label_names, proportions=False)

    # Create a mask for True Negatives
    mask = np.zeros_like(df, dtype=bool)
    mask[:, 0] = True  # Mask the True Negatives column

    # Plotting
    plt.figure(figsize=(10, len(label_names) * 0.5))  # Adjust as necessary
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f" if proportions else "d",
        cmap="coolwarm",
        cbar=True,
        linewidths=0.5,
    )

    plt.title(
        "Confusion Matrix Stats per Label (Proportions)"
        if proportions
        else "Confusion Matrix Stats per Label"
    )
    plt.show()


# Example usage:
# label_name = 'ar / vr'
# false_negatives_identifiers = get_false_negatives_identifiers(Y_val, predictions_val, label_name)
# print(false_negatives_identifiers)


def categorise_predictions(
    label: str, predictions: np.array, actual: pd.DataFrame
) -> Tuple[pd.Index, pd.Index, pd.Index, pd.Index]:
    """
    If you have the predictions from a multilabel classifier and want to inspect the model's performance for
    an individual label, you can supply the label (eg "Robotics"), the model's predictions, and the actual
    y values and get back the indexes of the true positives, true negatives, false positives and false negatives.

    The index of `actual` should be meaningful eg a unique identifier.

    Parameters:
    label (str): The label for which predictions are to be categorised.
    predictions (np.array): A DataFrame containing the prediction values.
    actual (pd.DataFrame): A DataFrame containing the actual values.

    Returns:
    Tuple[pd.Index, pd.Index, pd.Index, pd.Index]: A tuple containing indices for true positives,
    true negatives, false positives, and false negatives.
    """
    predictions_val_df = pd.DataFrame(
        predictions, index=actual.index, columns=actual.columns
    )
    merged_predictions = actual.merge(
        predictions_val_df,
        left_index=True,
        right_index=True,
        suffixes=("_actual", "_predicted"),
    )

    tp = merged_predictions[
        (merged_predictions[f"{label}_actual"] == 1)
        & (merged_predictions[f"{label}_predicted"] == 1)
    ].index
    tn = merged_predictions[
        (merged_predictions[f"{label}_actual"] == 0)
        & (merged_predictions[f"{label}_predicted"] == 0)
    ].index
    fp = merged_predictions[
        (merged_predictions[f"{label}_actual"] == 0)
        & (merged_predictions[f"{label}_predicted"] == 1)
    ].index
    fn = merged_predictions[
        (merged_predictions[f"{label}_actual"] == 1)
        & (merged_predictions[f"{label}_predicted"] == 0)
    ].index

    return tp, tn, fp, fn


def prediction_simple(classifier, text_data: list) -> pd.DataFrame:
    """Predicts the labels for a list of text data using a simple classifier

    Args:
        classifier (sklearn classifier): Trained classifier
        text_data (list): List of strings

    Returns:
        predictions (pd.DataFrame): A DataFrame containing the predictions
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_vectors_384 = model.encode(text_data, show_progress_bar=True)
    vectors_as_list = [list(vec) for vec in sentence_vectors_384]

    test_df = pd.DataFrame(
        {
            "openalex_id": [str(i) for i in range(len(vectors_as_list))],
            "miniLM_384_vector": vectors_as_list,
        }
    )

    # Setting up the test set
    X_test = test_df["miniLM_384_vector"].apply(pd.Series).values

    predictions = classifier.predict(X_test)

    # Make a dataframe with the predictions
    predictions = pd.DataFrame(
        {
            "text": text_data,
            "prediction": predictions,
        }
    )

    return predictions
