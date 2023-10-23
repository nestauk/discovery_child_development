# %% [markdown]
# # Train a baseline model and log on wandb

# %%
# Load libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

## plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

import wandb

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.utils import bigquery, labelling_utils
from discovery_child_development.utils.io import import_config

bigquery.find_credentials("GOOGLE_SHEETS_CREDENTIALS")

## constants

S3_BUCKET = os.environ["S3_BUCKET"]
PARAMS = import_config("config.yaml")
CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])
INPUT_PATH = "data/openAlex/processed/"
DATA_PATH_LOCAL = os.path.join(PROJECT_DIR, "inputs", "data")
FIG_PATH = os.path.join(PROJECT_DIR, "outputs", "figures")
SEED = 42

## variables
SCORE_THRESHOLD = 0.3  # we will remove any concepts (and corresponding subcategories) assigned with less than 0.3 confidence by the OpenAlex algorithm

# %%
# Initialize a run and log score threshold with wandb
run = wandb.init(
    project="ISS supervised ML",
    job_type="Baseline modeling",
    save_code=True,
    config={"score_threshold": SCORE_THRESHOLD},
)

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

openalex_data.head()

# %%
# Log the data as a wandb artifact
original_data = os.path.join(DATA_PATH_LOCAL, "openalex_training_data.csv")

openalex_data.to_csv(original_data)
# Create an Artifact
data_artifact = wandb.Artifact(
    name="openalex_training_data", type="data", description="OpenAlex training data"
)
# Add the file to the Artifact
data_artifact.add_file(original_data)
# Log the Artifact as part of the run
run.log_artifact(data_artifact)

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

# Log the plot as a wandb artifact
wandb.log({"level1_concept_score_distributions": wandb.Image(fig)})

# %%
openalex_data.head()

# %%
# Filter the data using a score threshold (0.3 is the threshold used by OpenAlex)
openalex_data_wide = (
    openalex_data[openalex_data["score"] >= SCORE_THRESHOLD]
    # Squash sub-categories into one tuple per work (rather than one row per sub-category per work)
    .groupby(["openalex_id", "text"])["sub_category"]
    .agg(tuple)
    .reset_index()
)

# logging information
prop_less_than_threshold = (
    len(openalex_data)
    - openalex_data[openalex_data["score"] >= SCORE_THRESHOLD].shape[0]
) / len(openalex_data)
logging.info(
    f"Prop of OpenAlex data/concepts tagged with a score less than {SCORE_THRESHOLD}: {prop_less_than_threshold}"
)
logging.info(f"N rows: {openalex_data_wide.shape[0]}")

# %%
Y = labelling_utils.add_binarise_labels(
    openalex_data_wide, label_column="sub_category", not_valid_label=None
)
Y.head()

# %%
# Load embeddings
embeddings = S3.download_obj(
    "discovery-iss",
    path_from=f"data/openAlex/vectors/sentence_vectors_384.parquet",
    download_as="dataframe",
)

# %%
openalex_data_wide = pd.merge(
    openalex_data_wide, embeddings, on="openalex_id", how="left"
)
openalex_data_wide.head()

# %%
# Save and log the new version of the data
preprocessed_data = os.path.join(DATA_PATH_LOCAL, "openalex_training_data_wide.csv")
openalex_data_wide.to_csv(preprocessed_data, index=False)

# Create an Artifact
preprocessed_data_artifact = wandb.Artifact(
    name="openalex_training_data_wide",
    type="data",
    description="OpenAlex data with squashed sub-categories and embeddings",
)
# Add the file to the Artifact
preprocessed_data_artifact.add_file(preprocessed_data)
# Log the Artifact as part of the run
run.log_artifact(preprocessed_data_artifact)


# %%
X = openalex_data_wide["miniLM_384_vector"].apply(pd.Series).values

# %%
Y = Y.to_numpy()

# %%
# Initialize the RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)

# Define the cross-validation iterator
kf = KFold(n_splits=5, random_state=SEED, shuffle=True)  # 5-fold cross-validation

# Evaluate the model using cross-validation
f1_scores = cross_val_score(classifier, X, Y, scoring="f1_micro", cv=kf)

print("F1 Scores for each fold:", f1_scores)
print("Average F1 Score: ", np.mean(f1_scores))

# %%
# log the scores with wandb
wandb.log({"average_f1_score": np.mean(f1_scores)})

# %%
wandb.finish()

# %%
