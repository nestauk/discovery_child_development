"""
Get predictions from a baseline model
"""

from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
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
INPUT_PATH = "data/openAlex/processed/"
VECTORS_FILEPATH = "data/openAlex/vectors/sentence_vectors_384.parquet"
DATA_PATH_LOCAL = os.path.join(PROJECT_DIR, "inputs", "data")
FIG_PATH = os.path.join(PROJECT_DIR, "outputs", "figures")
MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "models")
SEED = 42
# Set the seed
np.random.seed(SEED)

WANDB = False
MODEL_TYPE = "most_probable"
SCORE_THRESHOLD = 0.3  # we will remove any concepts (and corresponding subcategories) assigned with less than 0.3 confidence by the OpenAlex algorithm


class MostCommonClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, labels):
        self.labels = labels

    def fit(self, X, y=None):
        # Nothing happens here because it doesn't learn from the data
        return self

    def predict(self, X):
        # Returns the most common label combination for all input
        return np.tile(self.labels, (len(X), 1))


def generate_predictions(labels, label_probabilities):
    """When you pass in a set of labels and the probabilities of those labels, this function will use the probabilities
    to randomly generate a prediction for a single datapoint.

    Args:
        labels (_type_): _description_
        label_probabilities (_type_): _description_

    Returns:
        _type_: _description_
    """
    sample_predictions = []
    for label in labels:
        sample_predictions.append(
            np.random.choice(
                [0, 1],
                1,
                p=[
                    1 - label_probabilities[label],
                    label_probabilities[label],
                ],
            )[0]
        )
    return sample_predictions


class MostProbableClassifier:
    def __init__(self, label_probabilities):
        self.labels = label_probabilities.index
        self.label_probabilities = label_probabilities

    def predict(self, X):
        num_samples = len(X)
        predictions = []

        for _ in range(num_samples):
            sample_predictions = generate_predictions(
                self.labels, self.label_probabilities
            )
            predictions.append(sample_predictions)
        return pd.DataFrame(predictions, columns=self.labels)


def find_most_frequent_labels(
    df: pd.DataFrame, label_col: str = "sub_category", head: int = 20
):
    """Find the most frequent combination of labels - for a dataframe where one column contains a list of labels"""
    sub_category_combinations = df[label_col].value_counts()

    top_combinations = sub_category_combinations.head(head)

    labels = top_combinations.index[0]

    return top_combinations, labels


def find_most_common_row(df: pd.DataFrame) -> tuple[str, int]:
    """Find the row pattern that occurs most frequently (for a one-hot-encoded dataframe)"""

    # make a copy, otherwise this will modify the original dataframe even though we are not returning it
    df_copy = df.copy()

    # Convert each row into a string representation
    df_copy["combined"] = df_copy.apply(
        lambda row: "".join(row.astype(str).values), axis=1, result_type="reduce"
    )

    # Find the most common set of binary labels
    most_common_set = df_copy["combined"].value_counts().idxmax()
    # Find out how many times this combination occurs
    count = df_copy["combined"].value_counts().max()

    return most_common_set, count


def get_label_probabilities(
    df: pd.DataFrame, label_col: str = "sub_category", n: int = 1000, targets=None
) -> pd.Series:
    # Some OpenAlex IDs get tagged with the same sub-category multiple times (because one sub-category maps to multiple concepts)
    # So drop_duplicates ensures that each sub-category is only counted once per OpenAlex ID
    df_cleaned = df.drop_duplicates(subset=["openalex_id", label_col])
    label_probabilities = pd.DataFrame(df_cleaned[label_col].value_counts())
    # Normalise using the number of unique documents as the denominator (each label can appear at most once per document because of
    # how we cleaned the data above, so we have made sure that the max "prob" possible is 1)
    label_probabilities["prob"] = label_probabilities[label_col] / n
    label_probabilities = label_probabilities[
        ["prob"]
    ]  # drop the count column, so it is just index and probability
    label_probabilities = label_probabilities.squeeze()  # convert to a Series

    if targets is not None:
        missing_labels = pd.Series([], index=[])
        for x in targets:
            if x not in label_probabilities.index:
                missing_labels = missing_labels.append(pd.Series([float(0)], index=[x]))
        label_probabilities = label_probabilities.append(missing_labels)

    return label_probabilities


def run_baseline_model(
    model_type: str = "majority_combination",
    wandb_run: bool = True,
    score_threshold=0.3,
    s3_bucket: str = S3_BUCKET,
    input_path: str = INPUT_PATH,
    concept_ids: str = CONCEPT_IDS,
    vectors_filepath: str = VECTORS_FILEPATH,
    model_path: str = MODEL_PATH,
):
    valid_inputs = ["majority_combination", "most_probable"]  # "majority_label",

    if model_type not in valid_inputs:
        raise ValueError(
            f"Invalid input. Expected one of {valid_inputs}, got '{model_type}'"
        )

    # Load the data
    training_data_filename = (
        f"openalex_data_{concept_ids}_year-2019-2020-2021-2022-2023_train.csv"
    )

    openalex_data = S3.download_obj(
        s3_bucket,
        path_from=f"{input_path}{training_data_filename}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )

    # Filter the data using a score threshold (0.3 is the threshold used by OpenAlex)
    openalex_data_wide = (
        openalex_data[openalex_data["score"] >= score_threshold]
        # Squash sub-categories into one tuple per work (rather than one row per sub-category per work)
        .groupby(["openalex_id", "text"])["sub_category"]
        .agg(lambda x: tuple(set(x)))
        .reset_index()
    )
    # Set the index - useful later for creating training/validation split
    openalex_data_wide = openalex_data_wide.set_index("openalex_id")

    top_combinations, most_common_combination = find_most_frequent_labels(
        openalex_data_wide, label_col="sub_category", head=20
    )

    # Load embeddings
    embeddings = S3.download_obj(
        s3_bucket,
        path_from=vectors_filepath,
        download_as="dataframe",
    )

    embeddings = embeddings.set_index("openalex_id")

    openalex_data_wide = openalex_data_wide.join(
        embeddings, on="openalex_id", how="left"
    )

    Y, mlb = classification_utils.add_binarise_labels(
        openalex_data_wide, label_column="sub_category", not_valid_label=None
    )

    # Split IDs into random train and test subsets
    unique_ids = openalex_data_wide.index.unique()

    train_ids, val_ids = train_test_split(unique_ids, test_size=0.1, random_state=SEED)

    X_train = (
        openalex_data_wide[openalex_data_wide.index.isin(train_ids)][
            "miniLM_384_vector"
        ]
        .apply(pd.Series)
        .values
    )

    Y_train = Y[Y.index.isin(train_ids)]

    if model_type == "majority_combination":
        most_common_combination_one_hot = mlb.transform([top_combinations.index[0]])
        classifier = MostCommonClassifier(labels=most_common_combination_one_hot)

    elif model_type == "most_probable":
        # Assign probabilities using the training set
        train_df = openalex_data[
            (openalex_data["score"] >= score_threshold)
            & (openalex_data["openalex_id"].isin(train_ids))
        ]
        label_probabilities = get_label_probabilities(
            train_df,
            "sub_category",
            len(train_df["openalex_id"].unique()),
            targets=Y_train.columns,
        )
        # sort the index of label_probabilities so that it matches the order of columns in Y_train and Y_val
        label_probabilities.sort_index(inplace=True)
        classifier = MostProbableClassifier(label_probabilities=label_probabilities)

    baseline_predictions = classifier.predict(X_train)

    metrics = classification_utils.create_average_metrics(
        Y_train, baseline_predictions, average="samples"
    )
    logging.info(metrics)

    if wandb_run:
        # Initialize a wandb run and log score threshold with wandb
        run = wandb.init(
            project="ISS supervised ML",
            job_type="Baseline modeling",
            save_code=True,
            config={
                "baseline_model_type": model_type,
                "score_threshold": score_threshold,
            },
        )
        # We will use set the wandb description to be the dataset name (which includes details of concepts and years)
        # - we set the description, and not the name, as there is a limit on the length of artifact names.
        # wandb artifact names also cannot include "|" so we replace this.
        wandb_name = training_data_filename.replace("|", "_")
        # add reference to this data in wandb
        wb.add_ref_to_data(
            run,
            "openalex_train_data_raw",
            wandb_name,
            S3_BUCKET,
            training_data_filename,
        )
        # Log metrics
        for key, value in metrics.items():
            wandb.log({f"{key}": value})
        # Save and log the model and metrics
        model_path = f"{model_path}/baseline_most_probable.pkl"
        wb.log_model(run, f"baseline_{model_type}", classifier, model_path)
        # End the weights and biases run
        wandb.finish()


if __name__ == "__main__":
    run_baseline_model(
        model_type=MODEL_TYPE,
        wandb_run=WANDB,
        score_threshold=SCORE_THRESHOLD,
        s3_bucket=S3_BUCKET,
        input_path=INPUT_PATH,
        concept_ids=CONCEPT_IDS,
        vectors_filepath=VECTORS_FILEPATH,
        model_path=MODEL_PATH,
    )
