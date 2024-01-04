"""
Preprocesses the output from metaflow: creating an OpenAlex abstracts file.

Additional cleaning steps:
- NA values in 'abstract_inverted_index' and 'title' are removed.
- Works that are not in English are removed.

Usage:
python discovery_child_development/pipeline/binary_classifier/01_preprocess_openalex_broad.py
"""
import pandas as pd
from nesta_ds_utils.loading_saving import S3

from discovery_child_development import logging, S3_BUCKET, config
from discovery_child_development.utils.preprocess_openalex_utils import create_text_data
from discovery_child_development.getters.openalex import get_abstracts

if __name__ == "__main__":
    CONCEPT_LIST = config["openalex_broad_concepts"]
    CONCEPT_IDS = "|".join(CONCEPT_LIST)
    YEARS_LIST = [str(y) for y in config["openalex_years"]]
    YEARS = "-".join(YEARS_LIST)

    # paths for saving data
    S3_PATH = "metaflow/openalex_broad_concepts"

    OUTPUT_FILENAME_CONCEPTS = f"concepts_metadata_{CONCEPT_IDS}_year-{YEARS}.csv"
    OUTPUT_FILEPATH_CONCEPTS = "data/openAlex/concepts/broad_concepts/"
    OUTPUT_FILENAME_WORKS = f"openalex_abstracts_{CONCEPT_IDS}_year-{YEARS}.csv"
    OUTPUT_FILEPATH_WORKS = "data/openAlex/abstracts/broad_concepts/"

    INPUT_FILES = [
        f"openalex-works_production-True_concept-{concept}_year-{year}_sample-10000_seed-42.json"
        for year in YEARS_LIST
        for concept in CONCEPT_LIST
    ]

    openalex_df = pd.DataFrame()

    for file in INPUT_FILES:
        openalex_data = S3.download_obj(S3_BUCKET, f"{S3_PATH}/{file}", "dict")

        logging.info(f"Number of works in {file}: {len(openalex_data)}")

        year_df = pd.DataFrame(openalex_data)

        openalex_df = pd.concat([openalex_df, year_df])

    # Retain only works in English
    openalex_en = openalex_df[openalex_df["language"] == "en"].reset_index(drop=True)

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

    openalex_en = openalex_en[
        (openalex_en["abstract_inverted_index"].notnull())
        & (openalex_en["title"].notnull())
    ].reset_index(drop=True)

    logging.info(f"Remaining number of works after removing NAs: {len(openalex_en)}")

    # Removing works which are in the EY seed list
    logging.info("Removing works which are in the EY seed list...")
    ey_seed_works = get_abstracts().id.unique()

    openalex_en = openalex_en[~openalex_en.id.isin(ey_seed_works)].reset_index(
        drop=True
    )

    logging.info(
        f"Remaining number of works after removing EY seed works: {len(openalex_en)}"
    )

    # Creating abstracts metadata
    openalex_en_abstracts = create_text_data(
        openalex_en[["id", "title", "abstract_inverted_index"]]
    )

    openalex_en_abstracts = openalex_en_abstracts[
        openalex_en_abstracts.text.notnull()
    ].reset_index(drop=True)
    logging.info(
        f"Remaining number of works after removing NAs: {len(openalex_en_abstracts)}"
    )

    # Removing duplicates (based on id)
    logging.info("Removing duplicates...")
    openalex_en_abstracts = openalex_en_abstracts.drop_duplicates(
        subset=["id"], keep="first"
    ).reset_index(drop=True)
    logging.info(
        f"Remaining number of works after removing duplicates: {len(openalex_en_abstracts)}"
    )

    logging.info("Saving OpenAlex text data to S3...")
    # Write the text data to s3
    S3.upload_obj(
        openalex_en_abstracts,
        S3_BUCKET,
        f"{OUTPUT_FILEPATH_WORKS}{OUTPUT_FILENAME_WORKS}",
    )
