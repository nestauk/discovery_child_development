"""
Preprocesses the output from metaflow: splits it into a concepts metadata file, and an OpenAlex abstracts file.

Additional cleaning steps:
- NA valyes in 'abstract_inverted_index' and 'title' are removed.
- Works that are not in English are removed.

Usage:
python discovery_child_development/pipeline/01_preprocess_openalex.py
"""

from dotenv import load_dotenv
import pandas as pd
import os

from nesta_ds_utils.loading_saving import S3

from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.utils.io import import_config
from discovery_child_development.utils import openalex_utils

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]

PARAMS = import_config("config.yaml")

CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])
YEARS_LIST = [str(y) for y in PARAMS["openalex_years"]]
YEARS = "-".join(YEARS_LIST)

# paths for saving data
S3_PATH = "metaflow"

OUTPUT_FILENAME_CONCEPTS = f"concepts_metadata_{CONCEPT_IDS}_year-{YEARS}.csv"
OUTPUT_FILEPATH_CONCEPTS = "data/openAlex/concepts/"
OUTPUT_FILENAME_WORKS = f"openalex_abstracts_{CONCEPT_IDS}_year-{YEARS}.csv"
OUTPUT_FILEPATH_WORKS = "data/openAlex/"

INPUT_FILES = [
    f"openalex-works_production-True_concept-{CONCEPT_IDS}_year-{year}.json"
    for year in YEARS_LIST
]

if __name__ == "__main__":
    openalex_df = openalex_utils.concat_json_files(INPUT_FILES, S3_BUCKET, S3_PATH)

    # Retain only works in English
    openalex_en = openalex_df[openalex_df["language"] == "en"]

    logging.info(
        f"Number of works lost because they were not in English: {len(openalex_df)-len(openalex_en)}"
    )

    # Retain only works where abstract and title are not null
    logging.info(
        f"Number of NAs in 'abstract_inverted_index' before cleaning: {openalex_en['abstract_inverted_index'].isna().sum()}"
    )
    logging.info(
        f"Number of NAs in 'title' before cleaning: {openalex_en['title'].isna().sum()}"
    )

    openalex_en = openalex_en[openalex_en["abstract_inverted_index"].notnull()]
    openalex_en = openalex_en[openalex_en["title"].notnull()]

    logging.info(f"Remaining number of works after removing NAs: {len(openalex_en)}")

    concepts_df = openalex_utils.create_concepts_metadata(openalex_en)

    logging.info("Saving concepts metadata to S3...")
    # Write the concepts metadata to s3
    S3.upload_obj(
        concepts_df,
        S3_BUCKET,
        f"{OUTPUT_FILEPATH_CONCEPTS}{OUTPUT_FILENAME_CONCEPTS}",
    )

    openalex_en_abstracts = openalex_utils.create_text_data(
        openalex_en[["id", "title", "abstract_inverted_index"]]
    )

    logging.info("Saving OpenAlex text data to S3...")
    # Write to s3
    S3.upload_obj(
        openalex_en_abstracts,
        S3_BUCKET,
        f"{OUTPUT_FILEPATH_WORKS}{OUTPUT_FILENAME_WORKS}",
    )
