"""
Prepare labelled data for training a classifier
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

from nesta_ds_utils.loading_saving import S3

from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.getters import taxonomy, openalex
from discovery_child_development.utils.io import import_config

S3_BUCKET = os.environ["S3_BUCKET"]

PARAMS = import_config("config.yaml")

CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])
YEARS = [str(y) for y in PARAMS["openalex_years"]]
YEARS = "-".join(YEARS)

OUT_PATH = "data/openAlex/processed/"

# needed for train-test split
SEED = PARAMS["seed"]

if __name__ == "__main__":
    taxonomy_data = taxonomy.get_taxonomy()

    # check that there are no duplicate concepts - this should return 0
    duplicates = len(taxonomy_data) - len(taxonomy_data["display_name"].unique())
    if duplicates > 0:
        raise ValueError(
            f"There are {duplicates} duplicate concepts in the taxonomy data."
        )

    # check that the number of unique names and the number of unique ids are the same
    if len(taxonomy_data["display_name"].unique()) != len(
        taxonomy_data["concept_id"].unique()
    ):
        raise ValueError(
            "The number of unique names does not match the number of unique IDs."
        )

    # Get the IDs of concepts that we will use to filter the OpenAlex data
    taxonomy_concept_ids = taxonomy_data["concept_id"].unique()

    openalex_data = openalex.get_abstracts()

    openalex_concepts = openalex.get_concepts_metadata()

    openalex_concepts_subset = openalex_concepts[
        openalex_concepts["concept_id"].isin(taxonomy_concept_ids)
    ].copy()
    logging.info(f"N rows lost: {len(openalex_concepts)-len(openalex_concepts_subset)}")

    # merge taxonomy
    openalex_concepts_subset = pd.merge(
        openalex_concepts_subset,
        taxonomy_data[["sub_category", "concept_id"]],
        how="left",
        on="concept_id",
    )

    # Check whether any works have been lost because they were not tagged with any concepts from the taxonomy
    n_works_lost = len(openalex_concepts["openalex_id"].unique()) - len(
        openalex_concepts_subset["openalex_id"].unique()
    )
    logging.info(
        f"{n_works_lost} works lost because they were not tagged with any concepts from the taxonomy"
    )

    # Merge the abstracts, concepts metadata and taxonomy sub-categories
    openalex_data = openalex_concepts_subset[
        ["openalex_id", "concept_id", "sub_category", "display_name", "level", "score"]
    ].merge(
        openalex_data[["id", "text"]], left_on="openalex_id", right_on="id", how="left"
    )

    openalex_data = openalex_data.drop(columns=["id"], axis=1)

    # Split IDs into random train and test subsets
    unique_ids = openalex_data["openalex_id"].unique()

    train_ids, test_ids = train_test_split(unique_ids, test_size=0.1, random_state=SEED)

    train_df = openalex_data[openalex_data["openalex_id"].isin(train_ids)]
    test_df = openalex_data[openalex_data["openalex_id"].isin(test_ids)]

    # write to s3
    S3.upload_obj(
        train_df,
        S3_BUCKET,
        f"{OUT_PATH}openalex_data_{CONCEPT_IDS}_year-{YEARS}_train.csv",
    )
    S3.upload_obj(
        test_df,
        S3_BUCKET,
        f"{OUT_PATH}openalex_data_{CONCEPT_IDS}_year-{YEARS}_test.csv",
    )
