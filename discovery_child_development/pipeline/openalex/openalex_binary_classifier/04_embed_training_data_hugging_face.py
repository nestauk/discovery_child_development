"""
Embed the OpenAlex abstracts using the distilbert-base-uncased transformer for the huggingface model.
--------------

For the existing dataset of OpenAlex docs (already preprocessed with 01_preprocess_openalex_broad.py)
* create embeddings using HuggingFace distilbert-base-uncased model
* create dataframe that is compatible with the huggingface model
* save the dataframes to a pickle file

Usage:

python discovery_child_development/pipeline/openalex/binary_classifier/04_embed_training_data_hugging_face.py

Optional arguments:
    --production : Determines whether to create the embeddings for the full dataset or a test sample (default: True)
    --wandb : Determines whether a run gets logged on wandb (default: False)
    --identifier : Choose which split of the training data you want (default: 50, 50/50 relevant/non-relevant). Options are 20, 50, all.

"""

import wandb
import numpy as np
import argparse
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import PROJECT_DIR, logging, S3_BUCKET, config
from discovery_child_development.getters import openalex as oa
from discovery_child_development.utils.huggingface_pipeline import df_to_hf_ds
from discovery_child_development.utils.general_utils import replace_binary_labels
from discovery_child_development.getters.binary_classifier.binary_classifier_datasets import (
    get_data_for_classifier,
)
from discovery_child_development import binary_config, config

# Set up
CONCEPT_IDS = "|".join(config["openalex_concepts"])
VECTORS_PATH = "data/openAlex/vectors/distilbert/"
VECTORS_FILE = "sentence_vectors_384"
SEED = config["seed"]
NUM_SAMPLES = config["embedding_sample_size"]
# Set the seed
np.random.seed(SEED)

if __name__ == "__main__":
    # Set up the command line arguments
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--production",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Do you want to run the code in production? (default: True)",
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
    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    # Loading the training and validation data
    openalex_text_training = get_data_for_classifier(
        identifier=args.identifier, set_type="train"
    )
    openalex_text_validation = get_data_for_classifier(
        identifier=args.identifier, set_type="validation"
    )

    # Rename the label column to 0/1
    openalex_text_training = replace_binary_labels(openalex_text_training)
    openalex_text_validation = replace_binary_labels(openalex_text_validation)

    if args.wandb:
        run = wandb.init(
            project="ISS supervised ML",
            job_type="Binary classifier - huggingface",
            save_code=True,
            tags=["huggingface", "binary_classifier", "sentence_embeddings"],
        )

    # Small sample for testing
    if not args.production:
        openalex_text_training = openalex_text_training.sample(
            NUM_SAMPLES, random_state=SEED
        )
        openalex_text_validation = openalex_text_validation.sample(
            NUM_SAMPLES, random_state=SEED
        )
        VECTORS_FILE = VECTORS_FILE + "_test"

    training_embeddings = df_to_hf_ds(
        openalex_text_training,
        config=binary_config,
        non_label_cols=["openalex_id", "title", "abstract", "text"],
        text_column="text",
        problem_type=False,
    )
    validation_embeddings = df_to_hf_ds(
        openalex_text_validation,
        config=binary_config,
        non_label_cols=["openalex_id", "title", "abstract", "text"],
        text_column="text",
        problem_type=False,
    )

    # Save the model as a pickle file
    logging.info("Saving the embeddings...")
    S3.upload_obj(
        training_embeddings,
        S3_BUCKET,
        f"{VECTORS_PATH}{VECTORS_FILE}_{args.identifier}_train.pkl",
    )
    S3.upload_obj(
        validation_embeddings,
        S3_BUCKET,
        f"{VECTORS_PATH}{VECTORS_FILE}_{args.identifier}_validation.pkl",
    )
