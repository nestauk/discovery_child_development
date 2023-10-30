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

# import pickle
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

# from sklearn.base import BaseEstimator, ClassifierMixin

## plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.analysis import baseline_model as bm
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils.io import import_config

## constants

S3_BUCKET = os.environ["S3_BUCKET"]
PARAMS = import_config("config.yaml")
CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])
INPUT_PATH = "data/openAlex/processed/"
DATA_PATH_LOCAL = os.path.join(PROJECT_DIR, "inputs", "data")
FIG_PATH = os.path.join(PROJECT_DIR, "outputs", "figures")
MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "models")
SEED = 42

# List of paths to ensure they exist
paths_to_create = [DATA_PATH_LOCAL, FIG_PATH, MODEL_PATH]

for path in paths_to_create:
    os.makedirs(path, exist_ok=True)

## variables
SCORE_THRESHOLD = 0.3  # we will remove any concepts (and corresponding subcategories) assigned with less than 0.3 confidence by the OpenAlex algorithm

# %%
# Load data from s3
training_data_filename = (
    f"openalex_data_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023_train.csv"
)

openalex_data = S3.download_obj(
    S3_BUCKET,
    path_from=f"{INPUT_PATH}{training_data_filename}",
    download_as="dataframe",
    kwargs_reading={"index_col": 0},
)

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
# Set the index to be the OpenAlex ID - useful when we do train/test split
openalex_data_wide = openalex_data_wide.set_index("openalex_id")

# %% [markdown]
# Checking the distribution of combinations of sub category also shows us that there is a clear majority class: (general development / personal, social, emotional)

# %%
top_combinations, most_common_combination = bm.find_most_frequent_labels(
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

# %%
Y, mlb = classification_utils.add_binarise_labels(
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
# Check that both train and validation sets have the same most frequent label combination.
# It's not a problem if not, but just helpful to know. If they are different, it means that the train and validation sets have different distributions of labels.
Y_train_labels, Y_train_count = bm.find_most_common_row(Y_train)
Y_val_labels, Y_val_count = bm.find_most_common_row(Y_val)

Y_train_labels == Y_val_labels

# %% [markdown]
# ## Predict majority class (most commonly occurring label combination)

# %%
most_common_combination_one_hot = mlb.transform([top_combinations.index[0]])

# %% [markdown]
# We'll create predictions using `MostCommonClassifier` for both the training and validation sets and use the metrics from the training set as the baseline. The reason for this is the "stupid" classifier should make equally good (bad) predictions for both the training and validation sets, but there are more observations in the training set so the statistics should be most robust (?)

# %%
classifier = bm.MostCommonClassifier(labels=most_common_combination_one_hot)

baseline_predictions_val = classifier.predict(X_val)
baseline_predictions_train = classifier.predict(X_train)

# %%
metrics = classification_utils.create_average_metrics(
    Y_train, baseline_predictions_train, average="samples"
)
metrics

# %%
report = classification_report(
    Y_train,
    baseline_predictions_train,
    target_names=mlb.classes_,
    zero_division=0,
    output_dict=True,
)

report

# %%
# Check if these stats are the same on the validation set - does this baseline model generalise?
metrics_val = classification_utils.create_average_metrics(
    Y_val, baseline_predictions_val, average="samples"
)
metrics_val

# %% [markdown]
# ## Predict using probability

# %%
# Assign probabilities using the training set. It is data leakage if we get the probabilities from the whole dataset, so we get the probabilities just from the training set.

# subset the original dataframe so that we get just the training set
train_df = openalex_data[
    (openalex_data["score"] >= SCORE_THRESHOLD)
    & (openalex_data["openalex_id"].isin(train_ids))
]
label_probabilities = bm.get_label_probabilities(
    train_df,
    "sub_category",
    len(train_df["openalex_id"].unique()),
    targets=Y_train.columns,
)
# sort the index of label_probabilities so that it matches the order of columns in Y_train and Y_val
label_probabilities.sort_index(inplace=True)
# instantiate the classifier
probability_classifier = bm.MostProbableClassifier(
    label_probabilities=label_probabilities
)

# create predictions
random_choice_predictions_val = probability_classifier.predict(X_val)
random_choice_predictions_train = probability_classifier.predict(X_train)

# %%
rc_metrics_train = classification_utils.create_average_metrics(
    Y_train, random_choice_predictions_train, average="samples"
)
rc_metrics_train

# %%
rc_metrics_val = classification_utils.create_average_metrics(
    Y_val, random_choice_predictions_val, average="samples"
)
rc_metrics_val

# %% [markdown]
# Some observations:
# * The accuracy is similar whether you are using the majority-combination-classifier, or the probability-based random choice classifier.
# * The average precision is somewhat worse for the random choice classifier compared to the majority combination classifier. Same for recall and F1.
# * The Hamming loss is higher for the random choice classifier compared to majority combination
# * The jaccard score is lower for the random choice classifier compared to majority combination
#
# Additionally:
# * Although the probabilities for the random choice classifier came from the training set, there is similar performance on the training and validation sets (it does not perform worse on the validation set, as we might have expected). Presumably the distribution of labels is similar in the validation set compared to the training set.

# %%
