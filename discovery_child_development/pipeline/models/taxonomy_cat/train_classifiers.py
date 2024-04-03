"""
Script to train binary classifiers for taxonomy categories.

Usage
python discovery_child_development/pipeline/models/taxonomy_cat/train_classifiers.py --topic ai2
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import wandb
import json
import argparse

from nesta_ds_utils.loading_saving import S3
from discovery_child_development.utils.jsonl_utils import load_jsonl
from discovery_child_development import PROJECT_DIR, logging, config, S3_BUCKET
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils.general_utils import replace_binary_labels
from discovery_child_development.getters.openalex import get_sentence_embeddings
from discovery_child_development.utils.testing_examples_utils import (
    testing_examples_simple,
)
from discovery_child_development.utils import wandb as wb

from dotenv import load_dotenv

load_dotenv()

# Paths for saving trained models
MODEL_PATH = PROJECT_DIR / "outputs/models/taxonomy_cat/binary"
MODEL_PATH.mkdir(parents=True, exist_ok=True)
S3_MODEL_PATH = "models/taxonomy_cat/binary/"
# Path to sentence embeddings
VECTORS_PATH = "data/outputs/vectors/"
VECTORS_FILE = "sentence_vectors_384_labelled.parquet"
# Path to topic information
PATH_TO_TOPICS = (
    PROJECT_DIR
    / "discovery_child_development/pipeline/labelling/taxonomy_cat/prompts/topics.json"
)
# Path automatically labelled data folder
LABELS_DIR = PROJECT_DIR / "outputs/labels/taxonomy_cat"
# Path to manually labelled data (for evaluations)
EVALS_DIR = PROJECT_DIR / "outputs/labels/evals_data"
LABELS_TAXONOMY = EVALS_DIR / "taxonomy_labels_eval_annotated.jsonl"

# Setting the seed
SEED = config["seed"]
np.random.seed(SEED)

# PARAMS
WANDB_RUN = True
MODELS_SIMPLE = ["log_regression", "knn", "random_forest", "sgd", "svm"]


def parse_arguments():
    """Parse the arguments passed to the script."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the arguments for the script")
    # Add the arguments, and define defaults
    parser.add_argument("--topic", type=str, help="Taxonomy category", default="ai2")
    parser.add_argument(
        "--balance", type=bool, help="Balance the training set", default=False
    )
    # Parse the arguments
    return parser.parse_args()


if __name__ == "__main__":
    # Define the arguments
    args = parse_arguments()
    try:
        topic = args.topic
    except ValueError:
        raise ValueError("You must provide a valid topic")
    balance = args.balance

    # Load dataset sentence embeddings (all-MiniLM-L6-v2)
    embeddings_all = (
        get_sentence_embeddings(
            s3_bucket=S3_BUCKET,
            filepath=VECTORS_PATH,
            filename=VECTORS_FILE,
            id="id",
        )
        .reset_index()
        # Simplify the id by removing https
        .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
        .set_index("id")
    )
    # Load information about the topic
    topics = json.load(open(PATH_TO_TOPICS, "r"))
    topic_info = topics[topic]
    topic_name = topic_info["name"]

    # Load manually labelled data
    eval_df = (
        pd.DataFrame(load_jsonl(LABELS_TAXONOMY))
        .query("prediction == @topic_name")
        .drop(columns=["prediction"])
        .rename(columns={"answer": "labels"})
        .assign(
            labels=lambda df: df.labels.map(
                {"reject": "Not-relevant", "accept": "Relevant"}
            )
        )
        .query("labels == 'Relevant' or labels == 'Not-relevant'")
    )[["id", "labels", "text"]]
    # Load automatically labelled data
    labels_df = (
        pd.DataFrame(load_jsonl(LABELS_DIR / f"taxonomy_cat_{topic}.jsonl"))
        .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
        .query("id not in @eval_df.id")
        .rename(columns={"prediction": "labels"})
    )[["id", "labels", "text"]]

    if balance:
        # Balance the training set
        labels_df = (
            labels_df.groupby("labels")
            .apply(lambda x: x.sample(labels_df.labels.value_counts().min()))
            .reset_index(drop=True)
        )
        # log the number of samples
        logging.info(f"Balanced training set for topic {topic}")
        logging.info(labels_df.labels.value_counts())

    # Choose training and validation sets
    training_set = (
        labels_df.sample(frac=0.8, random_state=0)
        .merge(embeddings_all, on="id", how="left")
        .pipe(replace_binary_labels, replace_cat=["Relevant", "Not-relevant"])
        .dropna(subset=["miniLM_384_vector"])
    )
    validation_set = (
        labels_df.query("id not in @training_set.id")
        .merge(embeddings_all, on="id", how="left")
        .pipe(replace_binary_labels, replace_cat=["Relevant", "Not-relevant"])
        .dropna(subset=["miniLM_384_vector"])
    )
    # Prepare training and validation sets for the training
    X_train = training_set["miniLM_384_vector"].apply(pd.Series).values
    X_val = validation_set["miniLM_384_vector"].apply(pd.Series).values

    Y_train = training_set["labels"]
    Y_val = validation_set["labels"]

    for model in MODELS_SIMPLE:
        # Initialise wandb run
        if WANDB_RUN:
            # Initialize a wandb run
            run = wandb.init(
                project="ISS supervised ML",
                job_type="Taxonomy category classifier",
                save_code=True,
                tags=[model, f"cat_{topic}"],
            )

        # Creating the classifier
        if model == "log_regression":
            classifier = LogisticRegression(penalty="l2", random_state=SEED)
        elif model == "knn":
            classifier = KNeighborsClassifier()
        elif model == "random_forest":
            classifier = RandomForestClassifier(random_state=SEED)
        elif model == "sgd":
            classifier = SGDClassifier(random_state=SEED)
        elif model == "svm":
            classifier = LinearSVC(random_state=SEED)

        # Fitting the model
        classifier.fit(X_train, Y_train)
        # Predicting on the validation set
        predictions = classifier.predict(X_val)

        # Creating metrics
        metrics = classification_utils.create_average_metrics(
            Y_val, predictions, average="binary"
        )
        logging.info(f"Metrics for {model} model:")
        logging.info(metrics)
        logging.info(f"------")

        # Check on manually evaluated data
        examples = replace_binary_labels(
            eval_df, replace_cat=["Relevant", "Not-relevant"]
        )
        _, metrics_manual = testing_examples_simple(
            list(examples.text), list(examples.labels), classifier
        )
        print(metrics_manual)

        # Save model to S3
        model_path = f"{S3_MODEL_PATH}taxonomy_cat_classifier_{topic}_{model}.pkl"
        S3.upload_obj(
            obj=classifier,
            bucket=S3_BUCKET,
            path_to=model_path,
        )

        if WANDB_RUN:
            # Log metrics
            for metric in ["f1", "accuracy", "precision", "recall"]:
                wandb.run.summary[metric] = metrics[metric]
            for metric in ["f1", "accuracy", "precision", "recall"]:
                wandb.run.summary[f"{metric}_eval"] = metrics_manual[metric]
            # Adding reference to this model in wandb
            wb.add_ref_to_data(
                run=run,
                name=f"taxonomy_cat_classifier_{topic}_{model}",
                description=f"{model} model for taxonomy category {topic}",
                bucket=S3_BUCKET,
                filepath=model_path,
            )

            # End the weights and biases run
            wandb.finish()
