"""
Run the inference pipeline.

Usage:

python discovery_child_development/pipeline/models/binary_classifier/05_inference_OpenAlex_Patents.py


"""
import pandas as pd
import numpy as np
from discovery_child_development import (
    PROJECT_DIR,
    binary_config,
    config,
    S3_BUCKET,
    labelling_config,
    logging,
)
from nesta_ds_utils.loading_saving import S3
from discovery_child_development.utils.huggingface_pipeline import (
    load_model,
    load_training_args,
    load_trained_model,
)
from discovery_child_development.getters.binary_classifier.binary_classifier_model import (
    get_binary_classifier_models,
)
from discovery_child_development.utils.huggingface_pipeline import (
    predictions_huggingface,
)
from discovery_child_development.getters.unlabelled_data import (
    get_data_for_relevance_classifier,
)

from discovery_child_development.getters import patents, openalex

# Model vars
production = True

# Set the seed
SEED = config["seed"]
np.random.seed(SEED)

# Paths
S3_PATH = "models/binary_classifier/"
PATH_TO = f"{PROJECT_DIR}/outputs/data/models/"
MODEL_FILENAME = (
    f"gpt_labelled_binary_classifier_distilbert_production_{production}.tar.gz"
)
OUTPUT_FILENAME = labelling_config["OUTPUT_FILENAME"]

if __name__ == "__main__":
    # Save the model locally
    logging.info("Downloading the model...")
    get_binary_classifier_models(
        filename=MODEL_FILENAME, s3_path=S3_PATH, path_to=PATH_TO
    )
    model_folder = (
        f"{PATH_TO}gpt_labelled_binary_classifier_distilbert_production_{production}"
    )

    # Load the model
    logging.info("Loading the model...")
    model = load_model(model_path=model_folder, config=binary_config, num_labels=2)

    # Train model with early stopping
    training_args = load_training_args(**binary_config["training_args"])
    trainer = load_trained_model(
        model=model,
        args=training_args,
        config=binary_config,
    )

    # Get the labelled data
    logging.info("Getting the labelled data...")
    patents_df = patents.get_and_process_patents_from_s3()
    openalex_df = openalex.get_abstracts()
    data_for_labelling = pd.concat(
        [patents_df[["id", "text"]], openalex_df[["id", "text"]]], ignore_index=True
    ).dropna(subset=["text"])

    # Get the predictions
    logging.info("Getting the predictions...")
    predictions = predictions_huggingface(
        trainer=trainer, text_data=data_for_labelling, config=binary_config
    )

    # Save the predictions
    logging.info("Saving the predictions...")
    S3.upload_obj(
        predictions,
        S3_BUCKET,
        f"data/outputs/binary_classifier/{OUTPUT_FILENAME}.csv",
    )
