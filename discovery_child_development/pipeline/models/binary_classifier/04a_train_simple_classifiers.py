"""
Run five versions of the binary classifier: logistic regression, knn, random forest, SGD classifier, SVM.

Usage:

python discovery_child_development/pipeline/models/binary_classifier/04a_train_simple_classifiers.py

Optional arguments:
--wandb : Determines whether a run gets logged on wandb (default: False)
--identifier : Choose which split of the training data you want (default: 50, 50/50 relevant/non-relevant). Options are "20", "50", "all".

"""
# Import packages
import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import wandb

## Nesta DS utils
from nesta_ds_utils.loading_saving import S3

## Import from project
from discovery_child_development import PROJECT_DIR, logging, config, S3_BUCKET
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils.general_utils import replace_binary_labels
from discovery_child_development.getters.binary_classifier.gpt_labelled_datasets import (
    get_labelled_data_for_classifier,
)
from discovery_child_development.getters.openalex import get_sentence_embeddings
from discovery_child_development.utils import wandb as wb

load_dotenv()

MODEL_PATH = PROJECT_DIR / "outputs/models/"
S3_PATH = "models/binary_classifier/gpt_labelled_"

PATH_FROM = "data/labels/binary_classifier/processed/"
VECTORS_PATH = "data/labels/binary_classifier/vectors/"
VECTORS_FILE = "sentence_vectors_384_labelled.parquet"

# Setting the seed
SEED = config["seed"]
np.random.seed(SEED)

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

    labelled_text_training = get_labelled_data_for_classifier(
        set_type="train", path_from=PATH_FROM
    )
    labelled_text_validation = get_labelled_data_for_classifier(
        set_type="validation", path_from=PATH_FROM
    )

    # Embeddings from all-MiniLM-L6-v2
    # Not relevant
    embeddings_all = get_sentence_embeddings(
        s3_bucket=S3_BUCKET, filepath=VECTORS_PATH, filename=VECTORS_FILE
    )

    # Create training and validation sets
    training_set = labelled_text_training.merge(embeddings_all, on="id", how="left")
    validation_set = labelled_text_validation.merge(embeddings_all, on="id", how="left")
    training_set = replace_binary_labels(training_set)
    validation_set = replace_binary_labels(validation_set)

    # Setting up the training and validation sets
    X_train = training_set["miniLM_384_vector"].apply(pd.Series).values
    X_val = validation_set["miniLM_384_vector"].apply(pd.Series).values

    Y_train = training_set["labels"]
    Y_val = validation_set["labels"]

    for model in ["log_regression", "knn", "random_forest", "sgd", "svm"]:
        # Initialise wandb run
        if args.wandb:
            # Initialize a wandb run
            run = wandb.init(
                project="ISS supervised ML",
                job_type="Binary classifier - base models",
                save_code=True,
                tags=[model, "gpt_labelled"],
            )
            # Add reference to this data in wandb
            wb.add_ref_to_data(
                run=run,
                name="binary_train_data_raw_",
                description=f"Binary classifier training data",
                bucket=S3_BUCKET,
                filepath=f"{PATH_FROM}gpt_labelled_train.csv",
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
        logging.info(metrics)

        # Creating confusion matrix
        confusion_matrix = classification_utils.plot_confusion_matrix(
            Y_val, predictions, None, "Relevant works"
        )

        # Save model to S3
        S3.upload_obj(
            obj=classifier,
            bucket=S3_BUCKET,
            path_to=f"{S3_PATH}binary_classifier_{model}.pkl",
        )

        if args.wandb:
            # Log metrics
            wandb.run.summary["f1"] = metrics["f1"]
            wandb.run.summary["accuracy"] = metrics["accuracy"]
            wandb.run.summary["precision"] = metrics["precision"]
            wandb.run.summary["recall"] = metrics["recall"]

            # Adding reference to this model in wandb
            wb.add_ref_to_data(
                run=run,
                name=f"binary_classifier_{model}_" + args.identifier,
                description=f"{model} model trained on binary classifier training data",
                bucket=S3_BUCKET,
                filepath=f"{S3_PATH}binary_classifier_{model}.pkl",
            )

            # Log confusion matrix
            wb_confusion_matrix = wandb.Table(
                data=confusion_matrix, columns=confusion_matrix.columns
            )
            run.log({"confusion_matrix": wb_confusion_matrix})

            # End the weights and biases run
            wandb.finish()
