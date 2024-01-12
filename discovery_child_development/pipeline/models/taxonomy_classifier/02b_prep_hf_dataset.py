"""
Prepare HuggingFace datasets

Usage:

python discovery_child_development/pipeline/models/binary_classifier/03_embed_training_data_hugging_face.py

Optional arguments:
    --production : Determines whether to create a dataset from the full sample, or just a small sample (default: False)

"""
from discovery_child_development import S3_BUCKET, config, taxonomy_config, logging
from discovery_child_development.getters import taxonomy_classifier
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import huggingface_pipeline as hf

from nesta_ds_utils.loading_saving import S3 as nesta_s3

import argparse
from datasets import DatasetDict
import numpy as np
import pandas as pd
from typing import Any, Iterable

HF_PATH = taxonomy_config["s3_hf_ds_path"]
HF_FILE = taxonomy_config["s3_hf_ds_file"]

NUM_SAMPLES = 100


def create_dataset(
    train_val_df: pd.DataFrame,
    Y_train_val: pd.DataFrame,
    ids: Iterable,
    split: str = "train",
    s3_bucket: str = S3_BUCKET,
    hf_path: str = HF_PATH,
    hf_file: str = HF_FILE,
    production: bool = False,
) -> None:
    """
    Creates a Hugging Face dataset from given dataframes, and uploads it to an S3 bucket.

    Args:
    train_val_df (pd.DataFrame): A dataframe containing training/validation data. Must contain columns "id", "text", "source".
    Y_train_val (pd.DataFrame): A dataframe containing the labels for the training/validation data. Must contain a column "id". This should already be one-hot encoded using sklearn's MultilabelBinarizer.
    ids (Iterable): An iterable of IDs to filter the dataframes by - the unique ID determines whether a datapoint is in the train or validation set.
    split (str): The type of dataset split ('train' or 'validation').
    s3_bucket (str): The name of the S3 bucket to upload the dataset.
    hf_path (str): The path in the S3 bucket for the dataset.
    hf_file (str): The file name in the S3 bucket.
    production (bool): Flag to determine the filepath format for production or testing - if False, 'test_' gets pasted into the filename.

    Returns:
    None: This function does not return anything.
    """

    Y = Y_train_val[train_val_df.index.isin(ids)]

    # prepare the input: the texts
    X = train_val_df[train_val_df.index.isin(ids)][["source", "text"]]

    dataset_df = pd.merge(X, Y, on="id").reset_index()

    hf_ds = hf.df_to_hf_ds(
        dataset_df,
        config=taxonomy_config,
        non_label_cols=["id", "text", "source"],
        text_column="text",
    )

    # Save the datasets as a pickle files
    logging.info(f"Saving HF {split} dataset...")

    if production:
        filepath = f"{hf_path}{hf_file.replace('SPLIT', split)}"
    else:
        filepath = f"{hf_path}test_{hf_file.replace('SPLIT', split)}"

    nesta_s3.upload_obj(
        hf_ds,
        s3_bucket,
        filepath,
    )


if __name__ == "__main__":
    # Set up the command line arguments
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--production",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Do you want to run the code in production? (default: False)",
    )

    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    # Loading the training and validation data
    train_df, _ = taxonomy_classifier.get_training_data("train")
    val_df, _ = taxonomy_classifier.get_training_data("val")

    if not args.production:
        train_df = train_df.sample(NUM_SAMPLES)
        val_df = val_df.sample(NUM_SAMPLES)

    train_ids = train_df["id"].unique()
    val_ids = val_df["id"].unique()
    train_val_df = pd.concat([train_df, val_df])[
        ["id", "text", "source", "labels"]
    ].set_index("id")

    # prepare the target: the labels
    Y_train_val, mlb = classification_utils.add_binarise_labels(
        train_val_df, label_column="labels", not_valid_label=None
    )

    create_dataset(
        train_val_df,
        Y_train_val,
        ids=train_ids,
        split="train",
        s3_bucket=S3_BUCKET,
        hf_path=HF_PATH,
        hf_file=HF_FILE,
        production=args.production,
    )
    create_dataset(
        train_val_df,
        Y_train_val,
        ids=val_ids,
        split="val",
        s3_bucket=S3_BUCKET,
        hf_path=HF_PATH,
        hf_file=HF_FILE,
        production=args.production,
    )
