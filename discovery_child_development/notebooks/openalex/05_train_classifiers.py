# %% [markdown]
# Resources:
# * Scikit-learn documentation of the MultiOutputClassifier [here](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-multioutput-classification)
# * Kaggle example of classification of arXiv papers [here](https://www.kaggle.com/code/kobakhit/eda-and-multi-label-classification-for-arxiv) (one vs rest)
# * Example using MultiOutputClassifier and XGBoost [here](https://dongr0510.medium.com/multi-label-classification-example-with-multioutputclassifier-and-xgboost-in-python-98c84c7d379f)

# %%
import altair as alt
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from typing import Any, Iterable, List, Tuple, Union
import wandb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.getters import openalex as oa
from discovery_child_development.pipeline.models import baseline_model as bm
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import cluster_analysis_utils as cau
from discovery_child_development.utils import wandb as wb
from discovery_child_development.utils.io import import_config

load_dotenv()

S3_BUCKET = os.environ.get("S3_BUCKET")
PARAMS = import_config("config.yaml")
CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])
INPUT_PATH = f"data/openAlex/processed/openalex_data_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023_train.csv"
VECTORS_FILEPATH = "data/openAlex/vectors/sentence_vectors_384.parquet"
DATA_PATH_LOCAL = PROJECT_DIR / "inputs/data/"
FIG_PATH = PROJECT_DIR / "outputs/figures/"
MODEL_PATH = PROJECT_DIR / "outputs/models/"
SEED = 42
# Set the seed
np.random.seed(SEED)


# %%
# Functions


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
    plt.show()


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


def create_heatmap_table(y_true, y_pred, label_names, proportions=False):
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
    )

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
# create_heatmap_table(y_true, y_pred, label_names, proportions=True)

# def get_false_negatives_indexes(y_true, y_pred, label_index):
#     """
#     Get the indexes of false negatives.

#     Parameters:
#     y_true (array-like or DataFrame): True binary labels in binary indicator format for a single label.
#     y_pred (array-like or DataFrame): Binary labels predicted by the classifier for a single label.
#     label_index (int): Index of the label for which to find false negatives.

#     Returns:
#     np.array: Indexes of the false negatives.
#     """
#     # If the inputs are DataFrames, convert them to numpy arrays
#     if isinstance(y_true, pd.DataFrame):
#         y_true = y_true.values
#     if isinstance(y_pred, pd.DataFrame):
#         y_pred = y_pred.values

#     # Get the indexes where y_true is 1 (positive class) and y_pred is 0 (negative class)
#     false_negatives = np.where((y_true[:, label_index] == 1) & (y_pred[:, label_index] == 0))
#     return false_negatives


def get_false_negatives_identifiers(y_true, y_pred, label_name):
    """
    Get the identifiers of false negatives from a DataFrame index.

    Parameters:
    y_true (DataFrame): True binary labels with unique identifiers as index.
    y_pred (DataFrame or array-like): Binary labels predicted by the classifier.
    label_name (str): Name of the label for which to find false negatives.

    Returns:
    Index: Identifiers of the false negatives.
    """
    # If y_pred is a numpy array, convert it to DataFrame with the same index as y_true
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)

    # Select the series for the label of interest
    y_true_label = y_true[label_name]
    y_pred_label = y_pred[label_name]

    # Get the identifiers where y_true is 1 (positive class) and y_pred is 0 (negative class)
    false_negatives_identifiers = y_true_label[
        (y_true_label == 1) & (y_pred_label == 0)
    ].index
    return false_negatives_identifiers


# Example usage:
# label_name = 'ar / vr'
# false_negatives_identifiers = get_false_negatives_identifiers(Y_val, predictions_val, label_name)
# print(false_negatives_identifiers)


def categorise_predictions(label="mobile", predictions=predictions_val, actual=Y_val):
    predictions_val_df = pd.DataFrame(
        predictions_val, index=Y_val.index, columns=Y_val.columns
    )
    merged_predictions = Y_val.merge(
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


# %%
# Load the data. Just the training set by default
openalex_data = oa.get_labelled_data()

# Check distribution of the labels
openalex_data["sub_category"].value_counts()

# %%
# 81% of publications get tagged with "personal, social, emotional" and 75% get tagged with "development (general)".
# That means that for most labels, 0 is the majority class, but for just these two labels, 1 is the majority class.
# It's important to inspect the confusion matrices for individual labels.
bm.get_label_probabilities(
    openalex_data, "sub_category", len(openalex_data["openalex_id"].unique())
)

# %%
# Filter the data using a score threshold (0.3 is the threshold used by OpenAlex)
openalex_data_wide = (
    # Squash sub-categories into one tuple per work (rather than one row per sub-category per work).
    # This is the required input to the sklearn MultiLabelBinarizer.
    openalex_data.groupby(["openalex_id", "text"])["sub_category"]
    .agg(lambda x: tuple(set(x)))
    .reset_index()
)
# Set the index - useful later for creating training/validation split
openalex_data_wide = openalex_data_wide.set_index("openalex_id")

# %%
# Load embeddings
# These are not actually used for prediction, but this is the form our input data will take
# in future when we're using an actual classifier instead of a dummy one.
embeddings = S3.download_obj(
    S3_BUCKET,
    path_from=VECTORS_FILEPATH,
    download_as="dataframe",
)

embeddings = embeddings.set_index("openalex_id")

openalex_data_wide = openalex_data_wide.join(embeddings, on="openalex_id", how="left")

# The multilabel binarizer splits the sub-category tuple into binary labels.
# Y has a column for each unique sub-category in the data, and one row per OpenAlex ID.
Y, mlb = classification_utils.add_binarise_labels(
    openalex_data_wide, label_column="sub_category", not_valid_label=None
)

# Split IDs into random train and validation subsets
unique_ids = openalex_data_wide.index.unique()

# We will only get metrics on the training set for now, because the baseline should be
# the best possible score we can get from a probability/majority-based dummy classifier,
# and we assume the metrics will be slightly better on the training set.
train_ids, val_ids = train_test_split(unique_ids, test_size=0.1, random_state=SEED)

X_train = (
    openalex_data_wide[openalex_data_wide.index.isin(train_ids)]["miniLM_384_vector"]
    .apply(pd.Series)
    .values
)
X_val = (
    openalex_data_wide[openalex_data_wide.index.isin(val_ids)]["miniLM_384_vector"]
    .apply(pd.Series)
    .values
)

Y_train = Y[Y.index.isin(train_ids)]
Y_val = Y[Y.index.isin(val_ids)]

# %%
# ~45 classes
len(mlb.classes_)

# %% [markdown]
# # OneVsRest with Logistic Regression
# One logistic regression model is fit per label. All labels are treated as independent of one another.

# %%
lg = LogisticRegression(penalty="l2", random_state=SEED)
onevsrest_classifier = OneVsRestClassifier(lg, n_jobs=2)

# %%
onevsrest_classifier.fit(X_train, Y_train)

# %%
predictions_train = onevsrest_classifier.predict(X_train)
predictions_val = onevsrest_classifier.predict(X_val)

# %%
# Get accuracy. We expect this to be low, because it's difficult to get exactly the right combination of labels for every datapoint
accuracy_score(Y_val, predictions_val)

# %%
# Micro = global. If classes are imbalanced, the classes with higher numbers of datapoints skew the score.
classification_utils.create_average_metrics(Y_val, predictions_val, average="micro")

# %%
# Macro gives equal weight to all classes - this is better for us because we have some quite small classes.
# Notice that precision and recall are much lower
classification_utils.create_average_metrics(Y_val, predictions_val, average="macro")

# %%
# "samples" calculates metrics for each datapoint (which you can do with a multilabel dataset)
# and averages the scores across all datapoints. Again, this will end up favouring larger
# classes, because they have more datapoints.
classification_utils.create_average_metrics(Y_val, predictions_val, average="samples")

# %%
create_heatmap_table(Y_val, predictions_val, mlb.classes_, proportions=False)

# %% [markdown]
# We can see that there are some classes where the model doesn't manage to get a single True Positive correct!

# %%
overall_classification_report = classification_report(
    Y_val, predictions_val, target_names=mlb.classes_, output_dict=True
)

overall_classification_report["mobile"]

# %%
# This one has a lot of false negatives
plot_confusion_matrix(
    Y_val,
    predictions_val,
    Y_val.columns.get_loc("statistical methods"),
    "statistical methods",
)

# %%
tp, tn, fp, fn = categorise_predictions(
    label="technology (general)", predictions=predictions_val, actual=Y_val
)

# %%
df = pd.DataFrame()

mapping = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

for key, values in mapping.items():
    temp_df = openalex_data_wide.loc[values]
    temp_df["outcome"] = key
    df = pd.concat([df, temp_df])

# %%
df.head()

# %%
embeddings = np.stack(df["miniLM_384_vector"].values)
embeddings_2d = cau.reduce_to_2D(embeddings, random_state=SEED)

# %%
df["x"] = embeddings_2d[:, 0]
df["y"] = embeddings_2d[:, 1]

# %%
fig_hdbscan = (
    alt.Chart(
        df[df["outcome"].isin(["tp", "fp", "fn"])]
    )  # filtering out some rows because altair has a limit on how many data points you can plot
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("outcome:N", legend=alt.Legend(title="outcome")),
        tooltip=["text"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig_hdbscan
