"""
Embed the GPT labelled abstracts using the distilbert-base-uncased transformer for the huggingface model.
--------------

For the existing dataset of GPD labelled docs
* create embeddings using HuggingFace distilbert-base-uncased model
* create dataframe that is compatible with the huggingface model
* save the dataframes to a pickle file

Usage:

python discovery_child_development/pipeline/models/binary_classifier/03_embed_training_data_hugging_face.py

Optional arguments:
    --production : Determines whether to create the embeddings for the full dataset or a test sample (default: True)

"""

import numpy as np
import argparse
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import logging, S3_BUCKET, config
from discovery_child_development.getters.binary_classifier.gpt_labelled_datasets import (
    get_labelled_data_for_classifier,
)
from discovery_child_development.utils.huggingface_pipeline import df_to_hf_ds
from discovery_child_development.utils.general_utils import replace_binary_labels
from discovery_child_development import binary_config, config

# Set up
SEED = config["seed"]
NUM_SAMPLES = config["embedding_sample_size"]
# Set the seed
np.random.seed(SEED)
VECTORS_PATH = "data/labels/binary_classifier/vectors/"
VECTORS_FILE = "distilbert_sentence_vectors_384_labelled"

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

    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    # Loading the training and validation data
    labelled_text_training = get_labelled_data_for_classifier(set_type="train")
    labelled_text_validation = get_labelled_data_for_classifier(set_type="validation")

    # Rename the label column to 0/1
    labelled_text_training = replace_binary_labels(
        labelled_text_training, replace_cat=["Relevant", "Not-relevant"]
    )
    labelled_text_validation = replace_binary_labels(
        labelled_text_validation, replace_cat=["Relevant", "Not-relevant"]
    )

    # Small sample for testing
    if not args.production:
        labelled_text_training = labelled_text_training.sample(
            NUM_SAMPLES, random_state=SEED
        )
        labelled_text_validation = labelled_text_validation.sample(
            NUM_SAMPLES, random_state=SEED
        )
        VECTORS_FILE = VECTORS_FILE + "_test"

    training_embeddings = df_to_hf_ds(
        labelled_text_training,
        config=binary_config,
        non_label_cols=["id", "source"],
        text_column="text",
        problem_type=False,
    )
    validation_embeddings = df_to_hf_ds(
        labelled_text_validation,
        config=binary_config,
        non_label_cols=["id", "source"],
        text_column="text",
        problem_type=False,
    )

    # Save the model as a pickle file
    logging.info("Saving the embeddings...")
    S3.upload_obj(
        training_embeddings,
        S3_BUCKET,
        f"{VECTORS_PATH}{VECTORS_FILE}_train.pkl",
    )
    S3.upload_obj(
        validation_embeddings,
        S3_BUCKET,
        f"{VECTORS_PATH}{VECTORS_FILE}_validation.pkl",
    )
