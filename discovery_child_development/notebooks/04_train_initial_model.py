# %% [markdown]
# # Train a baseline model and log on wandb
#
# In this notebook we:
# * Perform some EDA on the (training) data - check distribution of scores and then use this to filter the data; check distributions of OpenAlex concepts and the sub-categories that we assigned
# * Turn the data into the correct format for training classification models (ie a one-hot-encoding representation of the target)
# * Define a baseline classifier than guesses that all new publications will be tagged with the most common combination of labels that we have in the data
# * Log the model and key metrics on wandb

# %%
# Load libraries
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    classification_report,
)
from sklearn.base import BaseEstimator, ClassifierMixin

## plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

import wandb

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.utils import bigquery, labelling_utils
from discovery_child_development.utils import wandb as wb
from discovery_child_development.utils.io import import_config

bigquery.find_credentials("GOOGLE_SHEETS_CREDENTIALS")

## constants

S3_BUCKET = os.environ["S3_BUCKET"]
PARAMS = import_config("config.yaml")
CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])
INPUT_PATH = "data/openAlex/processed/"
DATA_PATH_LOCAL = os.path.join(PROJECT_DIR, "inputs", "data")
FIG_PATH = os.path.join(PROJECT_DIR, "outputs", "figures")
MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "models")
SEED = 42

## variables
SCORE_THRESHOLD = 0.3  # we will remove any concepts (and corresponding subcategories) assigned with less than 0.3 confidence by the OpenAlex algorithm


# %%
# helpful functions
class MostCommonClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, labels):
        self.labels = labels

    def fit(self, X, y=None):
        # Nothing happens here because it doesn't learn from the data
        return self

    def predict(self, X):
        # Returns the most common label combination for all input
        return [self.labels for _ in range(len(X))]


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


def find_most_frequent_labels(df: pd.DataFrame, label_col: str="sub_category", head: int=20) -> Tuple[pd.DataFrame, str]:
    """Find the most frequent combination of labels - for a dataframe where one column contains a list of labels"""
    sub_category_combinations = df[label_col].value_counts()

    top_combinations = sub_category_combinations.head(head)

    labels = top_combinations.index[0]

    return top_combinations, labels


def find_most_common_row(df: pd.DataFrame):
    """Find the row pattern that occurs most frequently (for a one-hot-encoded dataframe)"""
    # make a copy, otherwise this will modify the original dataframe even though we are not returning it
    df_copy = df.copy()

    # Convert each row into a string representation
    df_copy["combined"] = df_copy.apply(
        lambda row: "".join(row.astype(str).values), axis=1, result_type="reduce"
    )

    # Find the most common set of binary labels
    most_common_set = df_copy["combined"].value_counts().idxmax()
    count = df_copy["combined"].value_counts().max()

    return most_common_set, count


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


# %%
# Initialize a run and log score threshold with wandb
run = wandb.init(
    project="ISS supervised ML",
    job_type="Baseline modeling",
    save_code=True,
    config={"score_threshold": SCORE_THRESHOLD},
)

# %%
CONCEPT_IDS.replace("|", "_")

# %%
# Load data from s3
training_data_filename = (
    f"openalex_data_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023_train.csv"
)
# We will use set the wandb description to be the dataset name (which includes details of concepts and years)
# - we set the description, and not the name, as there is a limit on the length of artifact names.
# wandb artifact names also cannot include "|" so we replace this.
wandb_name = training_data_filename.replace("|", "_")

openalex_data = S3.download_obj(
    S3_BUCKET,
    path_from=f"{INPUT_PATH}{training_data_filename}",
    download_as="dataframe",
    kwargs_reading={"index_col": 0},
)

# add reference to this data in wandb
wb.add_ref_to_data(
    run, "openalex_train_data_raw", wandb_name, S3_BUCKET, training_data_filename
)

openalex_data.head()

# %% [markdown]
# ## EDA

# %% [markdown]
# Check how many OpenAlex IDs there are in the data (~20,000)

# %%
len(openalex_data["openalex_id"].unique())

# %% [markdown]
# Check distributions of `level` and `sub_category`. We can notice:
# * It turns out most of the concepts/categories that we determined to be relevant were level 1 or level 2 concepts
# * There are around 30,000 instances of something being tagged with "general development", which is more than the number of OpenAlex works!! That means that some of the concepts from within the "general development" category cover basically the whole dataset. In terms of creating a baseline model, we could try just predicting that every single paper is 1 for "general development" and 0 for all the other sub-categories, and see what evaluation metrics that gets.

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 18))

# Histogram of the "level" column
sns.histplot(openalex_data["level"], bins=6, kde=False, ax=axes[0])
axes[0].set_title("Distribution of Level")
axes[0].set_xlabel("Level")
axes[0].set_ylabel("Count")

# Count plot for the "sub_category" column
sub_category_order = openalex_data["sub_category"].value_counts().index
sns.countplot(
    data=openalex_data, y="sub_category", order=sub_category_order, ax=axes[1]
)
axes[1].set_title("Count of Sub-categories")
axes[1].set_xlabel("Count")
axes[1].set_ylabel("Sub-category")

plt.tight_layout()
plt.show()

# %% [markdown]
# Check how many sub-categories each OpenAlex ID gets tagged with (via the concepts). Turns out to be 4 on average, and maximum 13 in this dataset.

# %%
# How many categories does each work get tagged with?
# Count the number of unique sub-categories for each openalex_id
sub_categories_per_id = openalex_data.groupby("openalex_id")["sub_category"].nunique()

# Summary statistics of the number of sub-categories per openalex_id
sub_categories_stats = sub_categories_per_id.describe()

# Visualize the distribution of the number of sub-categories per openalex_id
plt.figure(figsize=(12, 6))
sns.histplot(sub_categories_per_id, bins=30, kde=False)
plt.title("Distribution of Number of Sub-categories per openalex_id")
plt.xlabel("Number of Sub-categories")
plt.ylabel("Count of openalex_id")
plt.grid(True)

sub_categories_stats, plt.show()

# %%
# Check distribution of scores
# Shows bimodal/multimodal distribution: there are a lot of very low scores close to 0, and another peak at around 0.5
openalex_data[["score"]].hist()


# %%
# Check the score distribution for each of the top-level concepts in the data to see if most observations in each bucket are above or below the 0.3 threshold


def hist_with_line(data, color):
    n, bins, patches = plt.hist(
        data, color=color, edgecolor="k"
    )  # Plotting the histogram
    plt.axvline(0.3, color="red", linestyle="--")  # Adding a vertical line at x=0.3


fig = plt.figure()

# Create a grid of histograms
g = sns.FacetGrid(
    openalex_data[openalex_data["level"] == 1], col="display_name", col_wrap=4, height=5
)
g.map(hist_with_line, "score")

# Setting the same x-axis limits for all facets
x_min = 0  # define your own min value
x_max = 1  # define your own max value
g.set(xlim=(x_min, x_max))

plt.savefig(os.path.join(FIG_PATH, "level1_concept_score_distributions.png"))
plt.show()

# %%
openalex_data.head()

# %%
# Filter the data using a score threshold (0.3 is the threshold used by OpenAlex)
openalex_data_wide = (
    openalex_data[openalex_data["score"] >= SCORE_THRESHOLD]
    # Squash sub-categories into one tuple per work (rather than one row per sub-category per work)
    .groupby(["openalex_id", "text"])["sub_category"]
    .agg(lambda x: tuple(set(x)))
    .reset_index()
)

prop_less_than_threshold = (
    len(openalex_data)
    - openalex_data[openalex_data["score"] >= SCORE_THRESHOLD].shape[0]
) / len(openalex_data)
logging.info(
    f"Prop of OpenAlex data/concepts tagged with a score less than {SCORE_THRESHOLD}: {prop_less_than_threshold}"
)
logging.info(f"N rows: {openalex_data_wide.shape[0]}")

# %%
openalex_data_wide = openalex_data_wide.set_index("openalex_id")

# %% [markdown]
# Checking the distribution of combinations of sub category also shows us that there is a clear majority class: (general development / personal, social, emotional)

# %%
top_combinations, most_common_combination = find_most_frequent_labels(
    openalex_data_wide, label_col="sub_category", head=20
)

# Create a bar plot for the most common sub_category combinations
plt.figure(figsize=(10, 8))
top_combinations.plot(kind="barh", color="skyblue")
plt.title("Top 20 Most Common Sub-category Combinations")
plt.xlabel("Count")
plt.ylabel("Sub-category Combinations")
plt.grid(axis="x")

# Show the plot
plt.tight_layout()
plt.show()

top_combinations

# %%
data = pd.DataFrame(top_combinations)
# Create a wandb.Table object
table = wandb.Table(columns=["sub_category", "Quantity"], data=data)

# Log the table to W&B
wandb.log({"top_combinations": table})

# %% [markdown]
# ## Prepare data for training a model

# %%
# Load embeddings
embeddings = S3.download_obj(
    "discovery-iss",
    path_from=f"data/openAlex/vectors/sentence_vectors_384.parquet",
    download_as="dataframe",
)

# %%
embeddings = embeddings.set_index("openalex_id")

# %%
openalex_data_wide = openalex_data_wide.join(embeddings, on="openalex_id", how="left")
openalex_data_wide.head()

# %%
Y, mlb = labelling_utils.add_binarise_labels(
    openalex_data_wide, label_column="sub_category", not_valid_label=None
)


# %%
# Split IDs into random train and test subsets
unique_ids = openalex_data_wide.index.unique()

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
# Check that both train and validation sets have the same most frequent label combination
Y_train_labels, Y_train_count = find_most_common_row(Y_train)
Y_val_labels, Y_val_count = find_most_common_row(Y_val)

Y_train_labels == Y_val_labels

# %% [markdown]
# ## Predict majority class (most commonly occurring label combination)

# %%
most_common_combination_one_hot = mlb.transform([top_combinations.index[0]])

# %% [markdown]
# We'll create predictions using `MostCommonClassifier` for both the training and validation sets and use the metrics from the training set as the baseline. The reason for this is the "stupid" classifier should make equally good (bad) predictions for both the training and validation sets, but there are more observations in the training set so the statistics should be most robust (?)

# %%
classifier = MostCommonClassifier(labels=most_common_combination_one_hot)

baseline_predictions_val = classifier.predict(X_val)
baseline_predictions_train = classifier.predict(X_train)

# %%
# The formats of Y_train and the predictions need to be tweaked a bit so that we can compare them
Y_train_correct = Y_train.values.tolist()
predictions_correct = [pred[0].tolist() for pred in baseline_predictions_train]

# %%
metrics = create_average_metrics(
    Y_train_correct, predictions_correct, average="samples"
)
metrics

# %%
for key, value in metrics.items():
    wandb.log({f"{key}": value})

# %%
report = classification_report(
    Y_train_correct,
    predictions_correct,
    target_names=mlb.classes_,
    zero_division=0,
    output_dict=True,
)

report

# %%
flat_report = flatten_classification_report(report)
flat_report

# %%
# Save and log the model and metrics
model_path = f"{MODEL_PATH}/baseline_majority_classifier.pkl"
pickle.dump(classifier, open(model_path, "wb"))
baseline_model = wandb.Artifact("baseline_classifier", type="model")
baseline_model.add_file(model_path)
run.log_artifact(baseline_model)

wandb.log({"classification_report": flat_report})

# %%
# End the weights and biases run
wandb.finish()

# %%
