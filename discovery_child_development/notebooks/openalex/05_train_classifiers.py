# %% [markdown]
# Resources:
# * Scikit-learn documentation of the MultiOutputClassifier [here](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-multioutput-classification)
# * Kaggle example of classification of arXiv papers [here](https://www.kaggle.com/code/kobakhit/eda-and-multi-label-classification-for-arxiv) (one vs rest)
# * Example using MultiOutputClassifier and XGBoost [here](https://dongr0510.medium.com/multi-label-classification-example-with-multioutputclassifier-and-xgboost-in-python-98c84c7d379f)

# %%
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from typing import Any, Iterable, List, Tuple, Union
import wandb

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.utils import classification_utils
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

SCORE_THRESHOLD = 0.3  # we will remove any concepts (and corresponding subcategories) assigned with less than 0.3 confidence by the OpenAlex algorithm

# %%
# Load the data
openalex_data = S3.download_obj(
    S3_BUCKET,
    path_from=INPUT_PATH,
    download_as="dataframe",
    kwargs_reading={"index_col": 0},
)

# Filter the data using a score threshold (0.3 is the threshold used by OpenAlex)
openalex_data_wide = (
    openalex_data[openalex_data["score"] >= SCORE_THRESHOLD]
    # Squash sub-categories into one tuple per work (rather than one row per sub-category per work).
    # This is the required input to the sklearn MultiLabelBinarizer.
    .groupby(["openalex_id", "text"])["sub_category"]
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

# %% [markdown]
# # OneVsRest

# %%
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# %%
lg = LogisticRegression(penalty="l2", random_state=SEED)
onevsrest_classifier = OneVsRestClassifier(lg, n_jobs=2)

# %%
onevsrest_classifier.fit(X_train, Y_train)

# %%
predictions_train = onevsrest_classifier.predict(X_train)
predictions_val = onevsrest_classifier.predict(X_val)

# %%
predictions_train[0]

# %%
Y_train.head(1)

# %%

# %% [markdown]
# # Boosted trees

# %% [markdown]
# # Fine tune the embeddings

# %% [markdown]
#
