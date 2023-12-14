"""
Prepare labelled data for training a classifier
"""
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import logging, config, S3_BUCKET
from discovery_child_development.getters import taxonomy, openalex

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = (
    f"data/openAlex/processed/taxonomy_classifier/openalex_train_test_{TIMESTAMP}/"
)

# needed for train-test split
SEED = config["seed"]


def identify_multiple_sub_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Check which title/concept combinations map to multiple subcategories.

    One concept can be mapped to multiple subcategories. In order to check that the taxonomy
    and the concept metadata have been merged correctly, we implement this check to make sure
    that some OpenAlex title-concept combinations map to multiple subcategories from the taxonomy.

    Args:
        df (pd.DataFrame): Concepts metadata dataframe.

    Returns:
        Tuple[pd.DataFrame, int]: A dataframe with the title-concept combinations and the number of sub-categories they map to;
                and the number of rows in that dataframe (number of unique title-concept combinations that map to multiple sub-categories)
    """
    # Count the number of subcategories for each title/concept_id pair
    grouped = df.groupby(["concept_id", "openalex_id", "title"])[
        "sub_category"
    ].nunique()

    # Find the rows where there are multiple subcategories
    multiple_subcats = grouped[grouped > 1].reset_index()

    # Count how many works are mapped to multiple subcategories now
    count = multiple_subcats.shape[0]

    return multiple_subcats, count


if __name__ == "__main__":
    taxonomy_data = taxonomy.get_taxonomy()

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

    _, rows_with_multiple_subcats = identify_multiple_sub_categories(
        openalex_concepts_subset
    )
    logging.info(
        f"{rows_with_multiple_subcats} title/concept combinations are mapped to multiple subcategories"
    )

    # Check whether any works have been lost because they were not tagged with any concepts from the taxonomy
    n_works_lost = len(openalex_concepts["openalex_id"].unique()) - len(
        openalex_concepts_subset["openalex_id"].unique()
    )
    logging.info(
        f"{n_works_lost} works lost because they were not tagged with any concepts from the taxonomy"
    )

    # Merge the abstracts, concepts metadata and taxonomy sub-categories
    logging.info("Merging concepts metadata with text data...")
    openalex_data = (
        openalex_concepts_subset[
            [
                "openalex_id",
                "concept_id",
                "sub_category",
                "display_name",
                "level",
                "score",
            ]
        ]
        .merge(
            openalex_data[["id", "text"]],
            left_on="openalex_id",
            right_on="id",
            how="left",
        )
        .drop(columns=["id"], axis=1)
    )

    # Split IDs into random train and test subsets
    logging.info("Beginning train-test split...")
    unique_ids = openalex_data["openalex_id"].unique()

    train_ids, test_ids = train_test_split(unique_ids, test_size=0.1, random_state=SEED)

    train_df = openalex_data[openalex_data["openalex_id"].isin(train_ids)]
    test_df = openalex_data[openalex_data["openalex_id"].isin(test_ids)]

    # write to s3
    logging.info("Uploading to S3...")
    S3.upload_obj(
        train_df,
        S3_BUCKET,
        f"{OUT_PATH}openalex_data_train.csv",
    )
    S3.upload_obj(
        test_df,
        S3_BUCKET,
        f"{OUT_PATH}openalex_data_test.csv",
    )
    logging.info("Complete!")
