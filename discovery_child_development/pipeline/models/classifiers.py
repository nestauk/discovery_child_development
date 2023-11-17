"""
Run three versions of the taxonomy classifier: knn, random forest or one-vs-rest logistic regression.

Usage:

python discovery_child_development/pipeline/models/classifiers.py --wandb True

wandb determines whether a run gets logged on wandb when the script is run.

"""

import argparse
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import wandb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## nesta ds
from nesta_ds_utils.loading_saving import S3

## project code
from discovery_child_development import PROJECT_DIR, logging, config, S3_BUCKET
from discovery_child_development.getters import openalex as oa
from discovery_child_development.pipeline.models import baseline_model as bm
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import cluster_analysis_utils as cau
from discovery_child_development.utils import wandb as wb

load_dotenv()

CONCEPT_IDS = "|".join(config["openalex_concepts"])
INPUT_PATH = f"data/openAlex/processed/openalex_data_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023_train.csv"

DATA_PATH_LOCAL = PROJECT_DIR / "inputs/data/"
FIG_PATH = PROJECT_DIR / "outputs/figures/"
MODEL_PATH = PROJECT_DIR / "outputs/models/"
SEED = 42
# Set the seed
np.random.seed(SEED)


def ensure_path_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


for path in [DATA_PATH_LOCAL, FIG_PATH, MODEL_PATH]:
    ensure_path_exists(path)

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

    args = parser.parse_args()
    logging.info(args)

    # Load the data. Just the training set by default
    openalex_data, training_file_name = oa.get_labelled_data()
    logging.info(training_file_name)

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

    # Load embeddings
    embeddings = oa.get_sentence_embeddings()

    openalex_data_wide = openalex_data_wide.join(
        embeddings, on="openalex_id", how="left"
    )

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
        openalex_data_wide[openalex_data_wide.index.isin(train_ids)][
            "miniLM_384_vector"
        ]
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

    for model in ["one_vs_rest", "knn", "random_forest"]:
        # Initialise wandb run
        if args.wandb:
            # Initialize a wandb run and log score threshold with wandb
            run = wandb.init(
                project="ISS supervised ML",
                job_type="Taxonomy classifier",
                save_code=True,
                tags=[model],
            )
            # Log the dataset
            wandb_name = training_file_name.split("/")[-1].replace("|", "_")
            # add reference to this data in wandb
            wb.add_ref_to_data(
                run,
                "openalex_train_data_raw",
                wandb_name,
                S3_BUCKET,
                training_file_name,
            )

        if model == "one_vs_rest":
            lg = LogisticRegression(penalty="l2", random_state=SEED)
            classifier = OneVsRestClassifier(lg, n_jobs=2)
        elif model == "knn":
            classifier = KNeighborsClassifier()
        elif model == "random_forest":
            classifier = RandomForestClassifier(random_state=SEED)

        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_val)

        metrics = classification_utils.create_average_metrics(
            Y_val, predictions, average="macro"
        )
        logging.info(metrics)

        confusion_matrix = classification_utils.create_confusion_matrix(
            Y_val, predictions, mlb.classes_, proportions=False
        )

        if args.wandb:
            # Log metrics
            wandb.run.summary["macro_avg_f1"] = metrics["f1"]
            wandb.run.summary["accuracy"] = metrics["accuracy"]
            wandb.run.summary["macro_avg_precision"] = metrics["precision"]
            wandb.run.summary["macro_avg_recall"] = metrics["recall"]

            # Save and log the model and metrics
            model_path = f"{MODEL_PATH}/{model}.pkl"
            wb.log_model(run, f"{model}", classifier, model_path)

            # Log confusion matrix
            wb_confusion_matrix = wandb.Table(
                data=confusion_matrix, columns=confusion_matrix.columns
            )
            run.log({"confusion_matrix": wb_confusion_matrix})

            # End the weights and biases run
            wandb.finish()
