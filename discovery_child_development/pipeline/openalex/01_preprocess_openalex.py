"""
Preprocesses the output from EY concepts metaflow: splits it into a concepts metadata file, and an OpenAlex abstracts file.

Additional cleaning steps:
- NA values in 'abstract_inverted_index' and 'title' are removed.
- Works that are not in English are removed.

Usage:
python discovery_child_development/pipeline/openalex/01_preprocess_openalex.py
"""
from nesta_ds_utils.loading_saving import S3 as nesta_s3
import pandas as pd
from dotenv import load_dotenv
import datetime

from discovery_child_development import S3_BUCKET, config, logging
from discovery_child_development.utils import openalex_utils as openalex_utils
from discovery_child_development.utils import utils

if __name__ == "__main__":
    API_ROOT = config["openalex_keywords_api_root"]
    KEYWORD_PATH = "metaflow/openalex_keyword_search"
    CONCEPTS_PATH = "metaflow/openalex_concepts"
    YEARS = config["openalex_years"]
    KEYWORDS = config["openalex_keywords"]

    load_dotenv()

    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"data/openAlex/openalex_works_concepts_{TIMESTAMP}/"

    OUTPUT_FILENAME_CONCEPTS = "concepts_metadata.csv"
    OUTPUT_FILENAME_WORKS = "openalex_abstracts.csv"

    keywords_folder = utils.get_latest_subfolder(
        S3_BUCKET, KEYWORD_PATH, "production_True"
    )
    concepts_folder = utils.get_latest_subfolder(
        S3_BUCKET, CONCEPTS_PATH, "production_True"
    )

    oa_keywords_data = nesta_s3.download_obj(
        S3_BUCKET,
        path_from=f"{keywords_folder}openalex_keywords_combined.json",
        download_as="dict",
    )
    oa_keywords_df = pd.DataFrame(oa_keywords_data)
    oa_keywords_df["extracted_date_time"] = utils.parse_timestamp_from_folder_name(
        keywords_folder
    )

    INPUT_FILES = [f"openalex-works_year-{year}.json" for year in YEARS]

    openalex_concepts_df = openalex_utils.concat_json_files(
        INPUT_FILES, S3_BUCKET, concepts_folder
    )
    openalex_concepts_df[
        "extracted_date_time"
    ] = utils.parse_timestamp_from_folder_name(concepts_folder)

    keywords_ids = set(oa_keywords_df["id"].unique())
    concept_works_ids = set(openalex_concepts_df["id"].unique())

    keyword_ids_to_keep = list(keywords_ids - concept_works_ids)

    oa_keywords_df = oa_keywords_df[oa_keywords_df["id"].isin(keyword_ids_to_keep)]

    oa_merged_df = pd.concat([openalex_concepts_df, oa_keywords_df]).reset_index(
        drop=True
    )

    oa_cleaned = openalex_utils.clean_openalex_data(oa_merged_df)

    concepts_df = openalex_utils.create_concepts_metadata(oa_cleaned)

    concepts_filename = utils.list_objects_in_subfolder(
        S3_BUCKET, concepts_folder, "concepts"
    )
    keywords_filename = utils.list_objects_in_subfolder(
        S3_BUCKET, keywords_folder, "keywords.+\.txt"
    )
    api_filename = utils.list_objects_in_subfolder(
        S3_BUCKET, keywords_folder, "api_calls"
    )

    utils.copy_s3_object(
        S3_BUCKET,
        f"{concepts_folder}{concepts_filename[0]}",
        f"{OUTPUT_DIR}{concepts_filename[0]}",
    )
    utils.copy_s3_object(
        S3_BUCKET,
        f"{keywords_folder}{keywords_filename[0]}",
        f"{OUTPUT_DIR}{keywords_filename[0]}",
    )
    utils.copy_s3_object(
        S3_BUCKET,
        f"{keywords_folder}{api_filename[0]}",
        f"{OUTPUT_DIR}{api_filename[0]}",
    )

    logging.info("Saving concepts metadata to S3...")
    # Write the concepts metadata to s3
    nesta_s3.upload_obj(
        concepts_df,
        S3_BUCKET,
        f"{OUTPUT_DIR}{OUTPUT_FILENAME_CONCEPTS}",
    )
    logging.info(f"Saved concepts metadata to {OUTPUT_DIR}{OUTPUT_FILENAME_CONCEPTS}")

    openalex_en_abstracts = openalex_utils.create_text_data(
        oa_cleaned[["id", "title", "abstract_inverted_index"]]
    )

    logging.info("Saving OpenAlex text data to S3...")
    # Write the text data to s3
    nesta_s3.upload_obj(
        openalex_en_abstracts,
        S3_BUCKET,
        f"{OUTPUT_DIR}{OUTPUT_FILENAME_WORKS}",
    )
    logging.info(f"Saved abstracts to {OUTPUT_DIR}{OUTPUT_FILENAME_WORKS}")
