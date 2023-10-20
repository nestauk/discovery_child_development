# %% [markdown]
# # Train a baseline model and log on wandb

# %%
# Load libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split

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
SEED = 42

## variables
SCORE_THRESHOLD = 0.3  # we will remove any concepts (and corresponding subcategories) assigned with less than 0.3 confidence by the OpenAlex algorithm

# %%
# Initialize a run
run = wandb.init(
    project="ISS supervised ML", job_type="Baseline modeling", save_code=True
)

# %%
# Load data
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
original_data = os.path.join(
    PROJECT_DIR, "inputs", "data", "openalex_training_data.csv"
)

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
openalex_data.head()

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


# Create a grid of histograms
g = sns.FacetGrid(
    openalex_data[openalex_data["level"] == 1], col="display_name", col_wrap=4, height=5
)
g.map(hist_with_line, "score")

# Setting the same x-axis limits for all facets
x_min = 0  # define your own min value
x_max = 1  # define your own max value
g.set(xlim=(x_min, x_max))

plt.show()

# %%

# %%

# %%

# %%
# Filter the data using a score threshold

# # Subset the data using the 0.3 threshold
# openalex_concepts_subset = openalex_concepts[openalex_concepts['score']>=0.3].copy()
# logging.info(f"Prop of concepts tagged with a score less than 0.3: {(len(openalex_concepts) - len(openalex_concepts_subset))/len(openalex_concepts)}")
# logging.info(f"N rows: {len(openalex_concepts_subset)}")

# %%
# Squash sub-categories into one tuple per work (rather than one row per sub-category per work)
# openalex_data = openalex_data.groupby(['openalex_id', 'text'])['sub_category'].agg(tuple).reset_index()


# %%
Y = labelling_utils.add_binarise_labels(
    openalex_data, label_column="sub_category", not_valid_label="?"
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
openalex_data = pd.merge(openalex_data, embeddings, on="openalex_id", how="left")
openalex_data.head()

# %%
X = openalex_data["miniLM_384_vector"].apply(pd.Series).values

# %%
Y = Y.to_numpy()

# %%
# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# Initialize the RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, Y_train)

# Make predictions
Y_pred = classifier.predict(X_test)

# Evaluate the predictions
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print(
    "F1 Score: ", f1_score(Y_test, Y_pred, average="micro")
)  # using 'micro' average for multilabel classification
print("Hamming Loss: ", hamming_loss(Y_test, Y_pred))
