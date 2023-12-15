"""
Prepare labelled data for training a binary classifier
--------------

For the existing dataset of GPT labelled data
* create a test set of 100 docs (50 patents, 50 openalex) for testing the classifier after validation
* create 3 training/validation set for the classifier
* save the training/validation/test set to S3

Usage:

python discovery_child_development/pipeline/models/binary_classifier/02_binary_classifier_training_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import logging, config, S3_BUCKET
from discovery_child_development.getters.labels import get_relevance_labels

OUT_PATH = "data/labels/binary_classifier/processed/"

# needed for train-validation split
SEED = config["seed"]

if __name__ == "__main__":
    # Loading in the data
    logging.info("Loading in the data...")
    labelled_data = (
        get_relevance_labels()
        # Rename the prediction column to labels
        .rename(columns={"prediction": "labels"})
        # Replace the 'Not-specified' label with 'Not-relevant'
        .replace({"labels": {"Not-specified": "Not-relevant"}})
    )

    # Create test set for the classifier
    logging.info("Creating test set...")
    # 50/50 split of EY seed list and broader concepts
    test_data = pd.concat(
        [
            labelled_data.query("source=='patents'").sample(50, random_state=SEED),
            labelled_data.query("source=='openalex'").sample(50, random_state=SEED),
        ]
    )

    # Upload test data to S3
    logging.info("Uploading test data to S3...")
    S3.upload_obj(
        test_data,
        S3_BUCKET,
        f"{OUT_PATH}gpt_labelled_test.csv",
    )
    # Remove test data from the training data
    labelled_data = labelled_data[~labelled_data["id"].isin(test_data["id"].unique())]

    # Training/Validation sets for the classifier
    logging.info("Splitting into a training/validation sets...")

    # 1. Create a training/validation set
    # Split IDs into random train and validation subsets for each of the 3 datasets
    logging.info("Beginning train-validation split...")
    unique_ids = labelled_data["id"].unique()
    train_ids, validation_ids = train_test_split(
        unique_ids, test_size=0.1, random_state=SEED
    )

    # Retain only the train and validation data, remove works with no text
    train_df = (
        labelled_data[labelled_data["id"].isin(train_ids)]
        .dropna(subset=["text"])
        .reset_index(drop=True)
    )
    validation_df = (
        labelled_data[labelled_data["id"].isin(validation_ids)]
        .dropna(subset=["text"])
        .reset_index(drop=True)
    )

    # write to s3
    logging.info("Uploading to S3...")
    S3.upload_obj(
        train_df,
        S3_BUCKET,
        f"{OUT_PATH}gpt_labelled_train.csv",
    )
    S3.upload_obj(
        validation_df,
        S3_BUCKET,
        f"{OUT_PATH}gpt_labelled_validation.csv",
    )
    logging.info(f"Training/validation set uploaded!")
