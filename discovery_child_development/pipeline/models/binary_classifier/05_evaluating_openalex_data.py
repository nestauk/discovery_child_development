"""
Create a testing dataframe for further fine-tuning/adding in further data for the classifier.

Usage:

python discovery_child_development/pipeline/models/binary_classifier/05_evaluating_openalex_data.py

Optional arguments:
--production : Determines whether you wish to use the production/non-production model (default: True)
--wandb : Determines whether a run gets logged on wandb (default: False)

"""
import pandas as pd
import numpy as np
import wandb
import argparse
from discovery_child_development.utils.huggingface_pipeline import (
    load_model,
    load_training_args,
    load_trained_model,
)
from discovery_child_development.getters.binary_classifier.binary_classifier_model import (
    get_binary_classifier_models,
)
from discovery_child_development.getters.openalex import get_abstracts
from discovery_child_development.getters.openalex_broad_concepts import (
    get_abstracts_broad,
)
from discovery_child_development.getters.binary_classifier.gpt_labelled_datasets import (
    get_labelled_data_for_classifier,
)
from discovery_child_development.utils.testing_examples_utils import (
    testing_examples_huggingface,
)
from nesta_ds_utils.loading_saving import S3
from discovery_child_development.utils import wandb as wb
from discovery_child_development.utils import classification_utils
from discovery_child_development import (
    logging,
    S3_BUCKET,
    config,
    binary_config,
    PROJECT_DIR,
)

# Set the seed
SEED = config["seed"]
np.random.seed(SEED)

# Paths
S3_PATH = "models/binary_classifier/"
PATH_TO = f"{PROJECT_DIR}/outputs/data/models/"
SAVE_PATH = "data/labels/binary_classifier/test_text/"

sample_size = binary_config["openalex_sample_size"]

if __name__ == "__main__":
    # Set up the command line arguments
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--production",
        type=bool,
        default=True,
        help="Do you want to run the code in production? (default: True)",
    )

    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Do you want to log this as a run on wandb? (default: False)",
    )
    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    if args.wandb:
        logging.info("Logging run on wandb")
        run = wandb.init(
            project="ISS supervised ML",
            job_type="Binary classifier - huggingface",
            save_code=True,
            tags=[
                "huggingface",
                "binary_classifier",
                "sentence_embeddings",
                "openalex",
            ],
        )

    # Output filename
    OUTPUT_FILENAME = (
        f"gpt_labelled_binary_classifier_distilbert_production_{args.production}.tar.gz"
    )

    # Retreiving the model and saving it locally
    get_binary_classifier_models(
        filename=OUTPUT_FILENAME, s3_path=S3_PATH, path_to=PATH_TO
    )

    # Loading the model
    model_folder = f"{PATH_TO}gpt_labelled_binary_classifier_distilbert_production_{args.production}"
    model = load_model(model_path=model_folder, config=binary_config, num_labels=2)

    # Train model with early stopping
    training_args = load_training_args(output_dir=S3_PATH, config=binary_config)
    trainer = load_trained_model(
        model=model,
        args=training_args,
        config=binary_config,
    )

    # Trialling the model on the openalex concepts

    # Get labelled training data
    labelled_data = get_labelled_data_for_classifier(set_type="train")
    labelled_data_ids = labelled_data.id.unique()

    # Get abstracts
    abstracts = get_abstracts().query("id not in @labelled_data_ids")
    abstracts_broad = get_abstracts_broad().query("id not in @labelled_data_ids")

    # Collecting sample of results
    relevant = abstracts.sample(sample_size, random_state=SEED).assign(labels=1)
    not_relevant = abstracts_broad.sample(sample_size, random_state=SEED).assign(
        labels=0
    )
    test_set = pd.concat([relevant, not_relevant])

    predictions, metrics = testing_examples_huggingface(
        trainer, test_set[["labels", "text"]], binary_config
    )

    # Adding predictions to the test set
    test_set = test_set.assign(predictions=predictions)

    # Reseting the index
    test_set = test_set.reset_index(drop=True)

    # Creating a confusion matrix
    confusion_matrix = classification_utils.plot_confusion_matrix(
        test_set.labels, predictions, None, "Relevant works"
    )

    if args.wandb:
        # Log metrics
        wandb.run.summary["f1"] = metrics["test_f1"]
        wandb.run.summary["accuracy"] = metrics["test_accuracy"]
        wandb.run.summary["precision"] = metrics["test_precision"]
        wandb.run.summary["recall"] = metrics["test_recall"]

        # Log confusion matrix
        wb_confusion_matrix = wandb.Table(
            data=confusion_matrix, columns=confusion_matrix.columns
        )
        run.log({"confusion_matrix": wb_confusion_matrix})

        # End the weights and biases run
        wandb.finish()

    # Save data results to S3
    S3.upload_obj(
        test_set,
        S3_BUCKET,
        f"{SAVE_PATH}gpt_labelled_results_openalex_sample_size_{sample_size}.csv",
    )
