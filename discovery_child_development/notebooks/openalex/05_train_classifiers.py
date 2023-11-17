# %% [markdown]
# This notebook "trains" the two baseline classifiers ("majority combination" and "most probable") plus actually trains a KNN model,  one-vs-rest logistic regression and a Random Forest classifier. It has been refactored into the script `pipeline/models/classifiers.py`. This notebook shows somewhat more of the thinking behind the process, compared to the script.
#
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
# Load the data. Just the training set by default
openalex_data = oa.get_labelled_data()[0]

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
# # Baseline classifiers

# %%
# majority combination
top_combinations, _ = bm.find_most_frequent_labels(
    openalex_data_wide, label_col="sub_category", head=20
)

most_common_combination_one_hot = mlb.transform([top_combinations.index[0]])
majority_classifier = bm.MostCommonClassifier(labels=most_common_combination_one_hot)

baseline_majority_predictions = majority_classifier.predict(X_val)

majority_metrics = classification_utils.create_average_metrics(
    Y_val, baseline_majority_predictions, average="samples"
)
logging.info(majority_metrics)

baseline_majority_confusion_matrix = classification_utils.create_heatmap_table(
    Y_val, baseline_majority_predictions, mlb.classes_, proportions=False
)

# %%
# probability-based
# Assign probabilities using the training set.
# This bit looks convoluted because we need a long-form dataframe to calculate
# the label probabilities, so we can't use openalex_data_wide.
train_df = openalex_data[openalex_data["openalex_id"].isin(train_ids)]
label_probabilities = bm.get_label_probabilities(
    train_df,
    "sub_category",
    len(train_df["openalex_id"].unique()),
    targets=Y_train.columns,
)
# sort the index of label_probabilities so that it matches the order of columns in Y_train and Y_val
label_probabilities.sort_index(inplace=True)
baseline_probability_classifier = bm.MostProbableClassifier(
    label_probabilities=label_probabilities
)

baseline_probability_predictions = baseline_probability_classifier.predict(X_val)

baseline_probability_metrics = classification_utils.create_average_metrics(
    Y_val, baseline_probability_predictions, average="macro"
)
logging.info(baseline_probability_metrics)

baseline_probability_confusion_matrix = classification_utils.create_heatmap_table(
    Y_val, baseline_probability_predictions, mlb.classes_, proportions=False
)

# %% [markdown]
# # OneVsRest with Logistic Regression
# One logistic regression model is fit per label. All labels are treated as independent of one another.

# %%
lg = LogisticRegression(penalty="l2", random_state=SEED)
onevsrest_classifier = OneVsRestClassifier(lg, n_jobs=2)

onevsrest_classifier.fit(X_train, Y_train)

one_vs_rest_predictions = onevsrest_classifier.predict(X_val)

# Get accuracy. We expect this to be low, because it's difficult to get exactly the right combination of labels for every datapoint
accuracy_score(Y_val, one_vs_rest_predictions)

# Micro = global. If classes are imbalanced, the classes with higher numbers of datapoints skew the score.

# "samples" calculates metrics for each datapoint (which you can do with a multilabel dataset)
# and averages the scores across all datapoints. Again, this will end up favouring larger
# classes, because they have more datapoints.

# Macro gives equal weight to all classes - this is better for us because we have some quite small classes.
# Notice that precision and recall are much lower
one_vs_rest_metrics = classification_utils.create_average_metrics(
    Y_val, one_vs_rest_predictions, average="macro"
)
logging.info(one_vs_rest_metrics)

classification_utils.create_heatmap_table(
    Y_val, one_vs_rest_predictions, mlb.classes_, proportions=False
)

# %% [markdown]
# We can see that there are some classes where the model doesn't manage to get a single True Positive correct!

# %%
one_vs_rest_classification_report = classification_report(
    Y_val, one_vs_rest_predictions, target_names=mlb.classes_, output_dict=True
)

one_vs_rest_classification_report["mobile"]

# %%
# This one has a lot of false negatives
classification_utils.plot_confusion_matrix(
    Y_val,
    one_vs_rest_predictions,
    Y_val.columns.get_loc("statistical methods"),
    "statistical methods",
)

# %%
# We can investigate whether there is any trend in the errors being made by plotting the text embeddings for the true positives,
# false positives and false negatives in 2D space.
tp, tn, fp, fn = classification_utils.categorise_predictions(
    label="technology (general)", predictions=one_vs_rest_predictions, actual=Y_val
)

df = pd.DataFrame()

mapping = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

for key, values in mapping.items():
    temp_df = openalex_data_wide.loc[values]
    temp_df["outcome"] = key
    df = pd.concat([df, temp_df])

embeddings = np.stack(df["miniLM_384_vector"].values)
embeddings_2d = cau.reduce_to_2D(embeddings, random_state=SEED)

df["x"] = embeddings_2d[:, 0]
df["y"] = embeddings_2d[:, 1]

fig_hdbscan = (
    alt.Chart(df[df["outcome"].isin(["tp", "fp", "fn"])])
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

# %% [markdown]
# ## KNN classifier

# %%
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier()

# Train the classifier
knn.fit(X_train, Y_train)

# Predict on the test data
knn_predictions = knn.predict(X_val)

# better precision, better F1 compared to onevsrest logistic regression
knn_metrics = classification_utils.create_average_metrics(
    Y_val, knn_predictions, average="macro"
)
logging.info(knn_metrics)

# Notice that this classifier does way better on some of the minority classes eg 'ar / vr'
classification_utils.create_heatmap_table(
    Y_val, knn_predictions, mlb.classes_, proportions=False
)

# %% [markdown]
# ## Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

# Initialize the RandomForestClassifier with a random state for reproducibility
rf = RandomForestClassifier(random_state=SEED)

# Train the classifier on the training data
rf.fit(X_train, Y_train)

# Predict on the test set
rf_predictions = rf.predict(X_val)

# high precision but terrible recall and F1
rf_metrics = classification_utils.create_average_metrics(
    Y_val, rf_predictions, average="macro"
)
logging.info(rf_metrics)

classification_utils.create_heatmap_table(
    Y_val, rf_predictions, mlb.classes_, proportions=False
)

# %%
