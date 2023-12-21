"""
Run five versions of the binary classifier: logistic regression, knn, random forest, SGD classifier, SVM.

Usage:

python discovery_child_development/pipeline/openalex/binary_classifier/05a_train_simple_classifiers.py

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
from discovery_child_development.getters.binary_classifier.binary_classifier_datasets import (
    get_data_for_classifier,
)
from discovery_child_development.getters.openalex import get_sentence_embeddings
from discovery_child_development.utils import wandb as wb

load_dotenv()

CONCEPT_IDS = "|".join(config["openalex_concepts"])
MODEL_PATH = PROJECT_DIR / "outputs/models/"
S3_PATH = "models/binary_classifier/"
YEARS = [str(y) for y in config["openalex_years"]]
YEARS = "-".join(YEARS)

PATH_FROM = "data/openAlex/processed/binary_classifier/"
VECTORS_PATH = "data/openAlex/vectors/"
VECTORS_FILE_NOT_RELEVANT = "sentence_vectors_384_broad.parquet"
VECTORS_FILE_RELEVANT = "sentence_vectors_384.parquet"

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

    parser.add_argument(
        "--identifier",
        type=str,
        default="50",
        help="Choose which split of the training data you want (default: 50, 50/50 relevant/non-relevant)",
    )

    args = parser.parse_args()
    logging.info(args)

    openalex_text_training = get_data_for_classifier(
        identifier=args.identifier, set_type="train"
    )
    openalex_text_validation = get_data_for_classifier(
        identifier=args.identifier, set_type="validation"
    )

    # Embeddings from all-MiniLM-L6-v2
    # Not relevant
    embeddings_all_not = get_sentence_embeddings(
        filepath=VECTORS_PATH, filename=VECTORS_FILE_NOT_RELEVANT
    )
    # Relevant
    embeddings_all = get_sentence_embeddings(
        filepath=VECTORS_PATH, filename=VECTORS_FILE_RELEVANT
    )
    # Combining the two
    embeddings_all_combined = pd.concat(
        [embeddings_all.reset_index(), embeddings_all_not.reset_index()],
        ignore_index=True,
    )

    # Create training and validation sets
    training_set = openalex_text_training.merge(
        embeddings_all_combined, on="openalex_id", how="left"
    )
    validation_set = openalex_text_validation.merge(
        embeddings_all_combined, on="openalex_id", how="left"
    )
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
                reinit=True,
                project="ISS supervised ML",
                job_type="Binary classifier - base models",
                save_code=True,
                tags=["all-MiniLM-L6-v2", model, "openealex training data"],
            )
            # Add reference to this data in wandb
            wb.add_ref_to_data(
                run=run,
                name="binary_train_data_raw_" + args.identifier,
                description=f"Binary classifier training data, {args.identifier} split",
                bucket=S3_BUCKET,
                filepath=f"{PATH_FROM}openalex_data_{CONCEPT_IDS}_year-{YEARS}_{args.identifier}_train.csv",
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
            path_to=f"{S3_PATH}binary_classifier_{model}_{args.identifier}.pkl",
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
                filepath=f"{S3_PATH}binary_classifier_{model}_{args.identifier}.pkl",
            )

            # Log confusion matrix
            wb_confusion_matrix = wandb.Table(
                data=confusion_matrix, columns=confusion_matrix.columns
            )
            run.log({"confusion_matrix": wb_confusion_matrix})

            # End the weights and biases run
            wandb.finish()
