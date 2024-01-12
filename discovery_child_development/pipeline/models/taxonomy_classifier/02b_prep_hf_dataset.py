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
from discovery_child_development import S3_BUCKET, config, taxonomy_config, logging
from discovery_child_development.getters import taxonomy_classifier
from discovery_child_development.utils import classification_utils
from discovery_child_development.utils import huggingface_pipeline as hf

from nesta_ds_utils.loading_saving import S3 as nesta_s3

import argparse
from datasets import DatasetDict
import numpy as np
import pandas as pd

HF_PATH = taxonomy_config["s3_hf_ds_path"]
HF_FILE = taxonomy_config["s3_hf_ds_file"]

NUM_SAMPLES = 100


def create_dataset(
    train_val_df,
    Y_train_val,
    ids,
    split="train",
    s3_bucket=S3_BUCKET,
    hf_path=HF_PATH,
    hf_file=HF_FILE,
    production=False,
):
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
        help="Do you want to run the code in production? (default: True)",
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
