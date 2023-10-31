"""
Preprocesses the output from metaflow: splits it into a concepts metadata file, and an OpenAlex abstracts file.

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


def create_concepts_metadata(df):
    data_list = []
    for index, row in df[["id", "title", "publication_year", "concepts"]].iterrows():
        for concept in row["concepts"]:
            data_list.append(
                {
                    "openalex_id": row["id"],
                    "title": row["title"],
                    "year": row["publication_year"],
                    "concept_id": concept["id"],
                    "wikidata": concept["wikidata"],
                    "display_name": concept["display_name"],
                    "level": concept["level"],
                    "score": concept["score"],
                }
            )

    out_df = pd.DataFrame(data_list)

    return out_df


def create_text_data(df):
    # Deinvert the abstract and stick together the title and abstract. This mimics preprocessing done to create [this dataset](https://huggingface.co/datasets/colonelwatch/abstracts-embeddings).

    df.loc[:, "abstract"] = df["abstract_inverted_index"].apply(
        lambda x: openalex_utils.deinvert_abstract(x)
    )

    df.loc[:, "text"] = df["title"] + " " + df["abstract"]

    return df[["id", "title", "abstract", "text"]]


if __name__ == "__main__":
    openalex_df = pd.DataFrame()

    for file in INPUT_FILES:
        openalex_data = S3.download_obj(S3_BUCKET, f"{S3_PATH}/{file}", "dict")

        logging.info(f"Number of works in {file}: {len(openalex_data)}")

        year_df = pd.DataFrame(openalex_data)

        openalex_df = pd.concat([openalex_df, year_df])

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

    concepts_df = create_concepts_metadata(openalex_en)

    logging.info("Saving concepts metadata to S3...")
    # Write the concepts metadata to s3
    S3.upload_obj(
        concepts_df,
        S3_BUCKET,
        f"{OUTPUT_FILEPATH_CONCEPTS}{OUTPUT_FILENAME_CONCEPTS}",
    )

    openalex_en_abstracts = create_text_data(
        openalex_en[["id", "title", "abstract_inverted_index"]]
    )

    logging.info("Saving OpenAlex text data to S3...")
    # Write to s3
    S3.upload_obj(
        openalex_en_abstracts,
        S3_BUCKET,
        f"{OUTPUT_FILEPATH_WORKS}{OUTPUT_FILENAME_WORKS}",
    )
