"""
Prepare labelled data for training a classifier
--------------

For the existing dataset of OpenAlex docs (already preprocessed with 01_preprocess_openalex_broad.py and 01_preprocess_openalex.py)
* create a test set of 1000 docs (500 relevant, 500 not relevant) for testing the classifier after validation
* create 3 training/validation sets for the classifier:
    1. 50% of the data is from the EY seed list, 50% is from the broader concepts (relavant/non-relevant)
    2. 20% of the data is from the EY seed list, 80% is from the broader concepts (relavant/non-relevant)
    3. All of the data is from the EY seed list and broader concepts (relavant - ~11% /non-relevant - ~89%)
* save the training/validation/test sets to S3

Usage:

python discovery_child_development/pipeline/binary_classifier/03_binary_classifier_training_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import logging, config, S3_BUCKET
from discovery_child_development.getters import openalex, openalex_broad_concepts

CONCEPT_IDS = "|".join(config["openalex_concepts"])
BROAD_CONCEPT_IDS = "|".join(config["openalex_broad_concepts"])
YEARS = [str(y) for y in config["openalex_years"]]
YEARS = "-".join(YEARS)

OUT_PATH = "data/openAlex/processed/binary_classifier/"

# needed for train-validation split
SEED = config["seed"]

if __name__ == "__main__":
    # Loading in the data
    logging.info("Loading in the data...")
    openalex_data = openalex.get_abstracts()
    openalex_broad_data = openalex_broad_concepts.get_abstracts_broad()

    # Create test set for the classifier
    logging.info("Creating test set...")
    # 50/50 split of EY seed list and broader concepts
    test_data = pd.concat(
        [openalex_broad_data.sample(500), openalex_data.sample(500)],
        keys=["not relevant", "relevant"],
    )
    # Upload test data to S3
    logging.info("Uploading test data to S3...")
    S3.upload_obj(
        test_data,
        S3_BUCKET,
        f"{OUT_PATH}openalex_data_{CONCEPT_IDS}_year-{YEARS}_test.csv",
    )
    # Remove test data from the training data
    openalex_data = openalex_data[~openalex_data["id"].isin(test_data["id"].unique())]
    openalex_broad_data = openalex_broad_data[
        ~openalex_broad_data["id"].isin(test_data["id"].unique())
    ]

    # Training/Validation sets for the classifier
    # Split into 3 different training/validation sets to see how the classifier performs on unbalanced data
    logging.info("Splitting into 3 different training/validation sets...")

    # 1. 50% of the data is from the EY seed list, 50% is from the broader concepts (relavant/non-relevant)
    classifier_data_50 = (
        pd.concat(
            [openalex_broad_data.sample(openalex_data.shape[0]), openalex_data],
            keys=["not relevant", "relevant"],
        )
        .reset_index(level=[0])
        .rename(columns={"level_0": "label", "id": "openalex_id"})
    )
    # 2. 20% of the data is from the EY seed list, 80% is from the broader concepts (relavant/non-relevant)
    classifier_data_20 = (
        pd.concat(
            [openalex_broad_data.sample(openalex_data.shape[0] * 4), openalex_data],
            keys=["not relevant", "relevant"],
        )
        .reset_index(level=[0])
        .rename(columns={"level_0": "label", "id": "openalex_id"})
    )
    # 3. All of the data is from the EY seed list and broader concepts (relavant - ~11% /non-relevant - ~89%)
    classifier_data_all = (
        pd.concat(
            [openalex_broad_data, openalex_data], keys=["not relevant", "relevant"]
        )
        .reset_index(level=[0])
        .rename(columns={"level_0": "label", "id": "openalex_id"})
    )

    # Split IDs into random train and validation subsets for each of the 3 datasets
    logging.info("Beginning train-validation split...")
    list_of_data = [classifier_data_50, classifier_data_20, classifier_data_all]
    list_of_identifiers = ["50", "20", "all"]
    for k, (data, identifier) in enumerate(zip(list_of_data, list_of_identifiers)):
        unique_ids = data["openalex_id"].unique()

        train_ids, validation_ids = train_test_split(
            unique_ids, test_size=0.1, random_state=SEED
        )

        # Retain only the train and validation data, remove research works with no abstract/title
        train_df = (
            data[data["openalex_id"].isin(train_ids)]
            .dropna(subset=["text"])
            .reset_index(drop=True)
        )
        validation_df = (
            data[data["openalex_id"].isin(validation_ids)]
            .dropna(subset=["text"])
            .reset_index(drop=True)
        )

        # write to s3
        logging.info("Uploading to S3...")
        S3.upload_obj(
            train_df,
            S3_BUCKET,
            f"{OUT_PATH}openalex_data_{CONCEPT_IDS}_year-{YEARS}_{identifier}_train.csv",
        )
        S3.upload_obj(
            validation_df,
            S3_BUCKET,
            f"{OUT_PATH}openalex_data_{CONCEPT_IDS}_year-{YEARS}_{identifier}_validation.csv",
        )
        logging.info(f"Dataset: {k} Complete!")
