"""
Get predictions from a baseline model

Usage:

python discovery_child_development/pipeline/models/baseline_model.py --model_type most_probable --wandb True

model_type can either be 'most_probable' or 'majority_combination' and determines the type of baseline classifier used:
* majority_combination = predicts the same combination of labels for every single new datapoint
* most_probable = generates predictions based on the distribution of labels in the training set

wandb determines whether a run gets logged on wandb when the script is run.

Further variables can be amended in the script (these are in the script because we anticipate they won't need to be changed often)
"""
import argparse
import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any, Iterable, List, Tuple, Union
import wandb

## nesta ds
from nesta_ds_utils.loading_saving import S3 as nesta_s3

## project code
from discovery_child_development import (
    PROJECT_DIR,
    logging,
    S3_BUCKET,
    config,
    taxonomy_config,
)
from discovery_child_development.getters import taxonomy
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import wandb as wb
from discovery_child_development.utils import utils

MODEL_PATH = PROJECT_DIR / taxonomy_config["models_path"]
utils.create_directory_if_not_exists(MODEL_PATH)
SEED = config["seed"]
# Set the seed
np.random.seed(SEED)


class MostCommonClassifier(BaseEstimator, ClassifierMixin):
    """
    This class implements a classifier that predicts the same combination of labels for every single new datapoint.
    The combination of labels provided during initialization is used as the prediction for all input samples.

    Attributes:
        labels (Iterable): The combination of labels that will be predicted for each new datapoint.
    """

    def __init__(self, labels: List[int]) -> None:
        """
        Initializes the MostCommonClassifier with the provided combination of labels.

        Args:
            labels (Iterable): The combination of labels that will be predicted for each new datapoint.
        """
        self.labels = labels

    def fit(self, X, y=None):
        # Nothing happens here because it doesn't learn from the data
        return self

    def predict(self, X: Iterable[Any]) -> np.ndarray:
        """
        Predicts the same combination of labels for all input samples.

        Args:
            X (Iterable): The input data. The length of X determines the number of predictions to be generated.

        Returns:
            np.ndarray: A numpy array containing the predictions. Each row corresponds to a sample in X,
                        and the values are the provided combination of labels.
        """
        # Returns the most common label combination for all input
        return np.tile(self.labels, (len(X), 1))


def generate_predictions(
    labels: Union[List[str], pd.Index], label_probabilities: pd.Series
) -> List[np.int64]:
    """When you pass in a set of labels and the probabilities of those labels, this function will use the probabilities
    to randomly generate a prediction for a single datapoint.

    Args:
        labels (Union[List[str], pd.Index]): An iterable which can be a list of strings or the index of a pd.Series,
                                             containing the labels for which predictions are to be made.
        label_probabilities (pd.Series): Series where the index is the labels, and the values are the probabilities of those labels.
        Can be generated with get_label_probabilities().

    Returns:
        List[np.int64]: A list of integer predictions (0 or 1) for each label, generated based on the given probabilities.
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
    """
    This class implements a classifier that generates predictions based on the most probable outcomes for each label.

    Attributes:
        labels (pd.Index): The index of the label probabilities, representing the labels for which predictions are to be made.
        label_probabilities (pd.Series): A pandas Series where the index corresponds to the labels and the values are
                                         the probabilities of each label being 1.
    """

    def __init__(self, label_probabilities: pd.Series) -> None:
        """
        Initializes the MostProbableClassifier with the provided label probabilities.

        Args:
            label_probabilities (pd.Series): A pandas Series where the index corresponds to the labels and the values are
                                             the probabilities of each label being 1.
        """
        self.labels = label_probabilities.index
        self.label_probabilities = label_probabilities

    def predict(self, X: Iterable[Any]) -> pd.DataFrame:
        """
        Generates predictions for each sample in the input based on the most probable outcomes for each label.

        Args:
            X (Iterable): An iterable containing the samples for which predictions are to be made. The length of X determines
                          the number of predictions to be generated.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the predictions. Each row corresponds to a sample in X, and each column
                          corresponds to a label. The values are the predicted outcomes (0 or 1) for each label.
        """
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
) -> Tuple[pd.Series, Union[Tuple, List]]:
    """
    Finds the most frequent combinations of labels in a specified column of a dataframe, where each entry in the column
    contains a list or tuple of labels. The function returns the top combinations and the most frequent combination of labels.

    Args:
        df (pd.DataFrame): A dataframe with the following required columns:
        - id (int or str): A unique identifier for each text. It can be either an integer or a string.
        - text (str): The document text. This column contains the actual text of the documents.
        - sub_category (tuple): A tuple (or list) of labels for the documents. Each tuple contains the labels
                                that are assigned to the corresponding document. There is no set length to the tuple/list.

            Example of dataframe structure:
            |   id   |         text         |    sub_category     |
            |--------|----------------------|---------------------|
            |   1    | "Document text 1..." | ("label1", "label2")|
            |   2    | "Document text 2..." | ("label3", "label4")|
            |  ...   |         ...          |         ...         |
        label_col (str, optional): The name of the column in the dataframe that contains the lists or tuples of labels.
                                   Defaults to "sub_category".
        head (int, optional): The number of top combinations to return. Defaults to 20.

    Returns:
        Tuple[pd.Series, Union[Tuple, List]]: A tuple containing two elements:
                                              1. A pandas Series containing the counts of the top combinations of labels.
                                              2. The most frequent combination of labels, which can be a tuple or a list.

    Note:
        The type of the most frequent labels (tuple or list) returned depends on the form of the entries in df[label_col].
    """
    sub_category_combinations = df[label_col].value_counts()

    top_combinations = sub_category_combinations.head(head)

    labels = top_combinations.index[0]

    return top_combinations, labels


def find_most_common_row(df: pd.DataFrame) -> Tuple[str, int]:
    """
    Finds the most common row pattern in a one-hot-encoded dataframe. Each row is converted into a string representation,
    and the function identifies the pattern that occurs most frequently, along with the count of its occurrences.

    Args:
        df (pd.DataFrame): The one-hot-encoded dataframe to analyze.

    Returns:
        Tuple[str, int]: A tuple containing two elements:
                         1. A string representing the most common row pattern (eg '00000010000001').
                         2. An integer indicating the number of times this pattern occurs in the dataframe.
    """

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
    df: pd.DataFrame,
    label_col: str = "sub_category",
    n: int = 1000,
    targets: Union[List[Any], pd.Index] = None,
) -> pd.Series:
    """Get label probabilities from long-form data

    Args:
        df (pd.DataFrame): a long-form dataframe with an ID column and a label column (where each ID can have multiple labels)
        label_col (str, optional): The column that contains the label. Defaults to "sub_category".
        n (int, optional): Denominator used when calculating the prevalence of labels. Should be the number of unique IDs. Defaults to 1000.
        targets (Union[List[Any], pd.Index], optional): Supply a list or the column names if you have a one-hot-encoded dataset of the set of possible labels.
                                                This should be used if you're calculating probabilities on the training set, but the training set does not contain all possible labels. Defaults to None.

    Returns:
        pd.Series: A series where the index is the labels, and the values are the corresponding probabilities.

    Note:
        Be careful to make sure that the output of this function is in the same order as the columns as your Y
        dataframe! You may need to sort the index of the output of this function.
    """
    label_probabilities = (
        pd.DataFrame(df[label_col].value_counts())
        # Normalise using the number of unique documents as the denominator (each label can appear at most once per document because of
        # how we cleaned the data above, so we have made sure that the max "prob" possible is 1)
        .assign(prob=lambda df: df[label_col] / n)
        # drop the count column, so it is just index and probability
        .drop(columns=label_col)
        # convert to a Series
        .squeeze()
    )

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
    s3_bucket: str = S3_BUCKET,
    model_path: str = MODEL_PATH,
) -> None:
    """
    Runs a baseline model for classification based on a specified model type, either using a majority combination
    or the most probable label approach. The function integrates with Weights & Biases (wandb) for experiment tracking.

    This function performs the following steps:
    - Validates the model type.
    - Optionally initializes a wandb run for experiment tracking.
    - Binarizes labels.
    - Initializes and trains the baseline classifier (either MostCommonClassifier or MostProbableClassifier).
    - Predicts and evaluates the model on the training set.
    - Logs metrics, model, and confusion matrix to wandb, if enabled.

    Parameters:
    - model_type (str): Type of baseline model to run. Options are "majority_combination" or "most_probable".
    - wandb_run (bool): If True, initializes a wandb run for experiment tracking.
    - s3_bucket (str): The S3 bucket name to download data from.
    - model_path (str): Path to save the trained model.

    Returns:
    None

    Raises:
    ValueError: If an invalid model type is specified.
    """

    valid_inputs = ["majority_combination", "most_probable"]

    if model_type not in valid_inputs:
        raise ValueError(
            f"Invalid input. Expected one of {valid_inputs}, got '{model_type}'"
        )

    train_df, train_df_filename = taxonomy.get_training_data("train")
    train_df = train_df[["id", "text", "labels"]]

    if wandb_run:
        # Initialize a wandb run and log score threshold with wandb
        run = wandb.init(
            project="ISS supervised ML",
            job_type="Taxonomy classifier",
            save_code=True,
            tags=[f"baseline_{model_type}"],
        )
        # We will use set the wandb description to be the dataset name (which includes details of concepts and years)
        # - we set the description, and not the name, as there is a limit on the length of artifact names.
        # add reference to this data in wandb
        wb.add_ref_to_data(
            run,
            "gpt_labelled_openalex_patents",
            train_df_filename,
            s3_bucket,
            train_df_filename,
        )

    # Load embeddings
    # These are not actually used for prediction, but this is the form our input data will take
    # in future when we're using an actual classifier instead of a dummy one.
    embeddings_train = taxonomy.get_sentence_embeddings("train")

    # This dataframe gers
    if model_type == "most_probable":
        train_df_long = train_df.explode("labels")

    train_df = train_df.merge(embeddings_train, on="id", how="left")

    # The multilabel binarizer splits the sub-category tuple into binary labels.
    # Y has a column for each unique sub-category in the data, and one row per OpenAlex ID.
    Y_train, mlb = classification_utils.add_binarise_labels(
        train_df, label_column="labels", not_valid_label=None
    )

    # We will only get metrics on the training set for now, because the baseline should be
    # the best possible score we can get from a probability/majority-based dummy classifier,
    # and we assume the metrics will be slightly better on the training set.

    X_train = train_df["miniLM_384_vector"].apply(pd.Series).values

    if model_type == "majority_combination":
        top_combinations, _ = find_most_frequent_labels(
            train_df, label_col="labels", head=20
        )

        most_common_combination_one_hot = mlb.transform([top_combinations.index[0]])
        classifier = MostCommonClassifier(labels=most_common_combination_one_hot)

    elif model_type == "most_probable":
        # Assign probabilities using the training set.
        # `get_label_probabilities()` requires a long-form dataset so we use the one that we set aside earlier.
        label_probabilities = get_label_probabilities(
            train_df_long,
            "labels",
            len(train_df_long["id"].unique()),
            targets=Y_train.columns,
        )
        # sort the index of label_probabilities so that it matches the order of columns in Y_train and Y_val
        label_probabilities.sort_index(inplace=True)
        classifier = MostProbableClassifier(label_probabilities=label_probabilities)

    baseline_predictions = classifier.predict(X_train)

    metrics = classification_utils.create_average_metrics(
        Y_train, baseline_predictions, average="macro"
    )
    logging.info(metrics)

    confusion_matrix = classification_utils.create_confusion_matrix(
        Y_train, baseline_predictions, mlb.classes_, proportions=False
    )

    if wandb_run:
        # Log metrics
        wandb.run.summary["macro_avg_f1"] = metrics["f1"]
        wandb.run.summary["accuracy"] = metrics["accuracy"]
        wandb.run.summary["macro_avg_precision"] = metrics["precision"]
        wandb.run.summary["macro_avg_recall"] = metrics["recall"]

        # Save and log the model and metrics
        model_path = f"{model_path}/baseline_{model_type}.pkl"
        wb.log_model(run, f"baseline_{model_type}", classifier, model_path)

        # Log confusion matrix
        wb_confusion_matrix = wandb.Table(
            data=confusion_matrix, columns=confusion_matrix.columns
        )
        run.log({"confusion_matrix": wb_confusion_matrix})

        # End the weights and biases run
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--wandb",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Do you want to log this as a run on wandb? (default: False)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="most_probable",
        choices=["majority_combination", "most_probable"],
        help='Specify the model type (default: "most_probable")',
    )

    args = parser.parse_args()

    run_baseline_model(
        model_type=args.model_type,
        wandb_run=args.wandb,
        s3_bucket=S3_BUCKET,
        model_path=MODEL_PATH,
    )
