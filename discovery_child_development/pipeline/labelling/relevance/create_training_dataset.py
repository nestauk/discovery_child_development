"""
Script to prepare the data for training the relevance model.

Usage:
    python discovery_child_development/pipeline/labelling/relevance/create_training_dataset.py
"""
import pandas as pd
from pathlib import Path

from nesta_ds_utils.loading_saving import S3

from discovery_child_development import logging, PROJECT_DIR, S3_BUCKET
from discovery_child_development.getters import get_dataset
from discovery_child_development.utils.utils import (
    load_jsonl,
    get_yaml_config,
)

# Get labelling config params
CONFIG = get_yaml_config(Path(__file__).resolve().parent / "config.yaml")
# Define paths to the outputs
OUTPUT_FILEPATH = PROJECT_DIR / CONFIG["local_output_directory"]
# OUTPUT_FILENAME = CONFIG["output_filename"]
OUTPUT_FILENAME = "testing"

if "__main__" == __name__:
    # Create the dataset for labelling
    S3.download_file(
        path_from=CONFIG["s3_directory"] + OUTPUT_FILENAME + ".jsonl",
        bucket=S3_BUCKET,
        path_to=str(OUTPUT_FILEPATH / OUTPUT_FILENAME) + ".jsonl",
    )
    labelled_df = pd.DataFrame(
        load_jsonl(str(OUTPUT_FILEPATH / OUTPUT_FILENAME) + ".jsonl")
    )
    logging.info(
        f"Downloaded {len(labelled_df)} labelled data points from {CONFIG['s3_directory']+OUTPUT_FILENAME+'.jsonl'}"
    )
    # Add text data to the labells dataframe
    df_with_text = []
    for dataset in labelled_df.source.unique():
        texts_df = get_dataset(dataset)
        df = labelled_df.query(f"source == '{dataset}'")
        df_with_text.append(
            df.copy().merge(texts_df[["id", "text"]], on="id", how="left")
        )
        logging.info(f"Added text data for {len(df)} samples of {dataset} dataset")
    labelled_df = pd.concat(df_with_text, ignore_index=True)

    S3.upload_obj(
        bucket=S3_BUCKET,
        path_to=CONFIG["s3_directory"] + OUTPUT_FILENAME + ".csv",
        obj=labelled_df,
    )
    logging.info(
        f"Uploaded {len(labelled_df)} training data points as {OUTPUT_FILENAME}.csv to {CONFIG['s3_directory']}"
    )
