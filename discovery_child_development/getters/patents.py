"""
Getters for patent data
"""
import os
import pandas as pd

from nesta_ds_utils.loading_saving import S3
from discovery_child_development import config
from discovery_child_development.utils.keywords import (
    process_keywords,
    check_keyword_hits,
)

PATENTS_PATH = config["patents_data_path"]
PATENTS_DATA_VERSION = config["patents_data_version"]


def get_patents_from_s3(
    patents_data_version: str = PATENTS_DATA_VERSION,
) -> pd.DataFrame:
    """Get google patents data from S3

    Args:
        patent_data_version (str, optional): Version of patent data to download.

    Returns:
        pd.DataFrame: Google patents data
    """
    return S3.download_obj(
        os.environ["S3_BUCKET"],
        f"{PATENTS_PATH}{patents_data_version}/{patents_data_version}.parquet",
        download_as="dataframe",
    )


def get_keywords_from_s3(
    patents_data_version: str = PATENTS_DATA_VERSION,
) -> pd.DataFrame:
    """Get keywords from S3 that are used to find patents

    Args:
        patent_data_version (str, optional): Version of patent data to download.

    Returns:
        pd.DataFrame: Keywords
    """
    return process_keywords(
        S3.download_obj(
            os.environ["S3_BUCKET"],
            f"{PATENTS_PATH}{patents_data_version}/keywords_query.txt",
            download_as="list",
        )
    )


def _get_and_process_patents_from_s3(
    patents_data_version: str = PATENTS_DATA_VERSION,
) -> pd.DataFrame:
    """Get google patents data from S3 and do light processing by combining title and abstract and removing patents without text

    Args:
        patents_data_version (str, optional): Version of patent data to download.

    Returns:
        pd.DataFrame: Google patents data
    """
    data_raw_df = get_patents_from_s3(patents_data_version)
    # Load keywords
    keywords = get_keywords_from_s3(patents_data_version)
    # Process and filter the data
    return (
        data_raw_df
        # Remove patents without text
        .dropna(subset=["abstract"])
        # Combine title and abstract
        .assign(text=lambda df: df["title"] + ". " + df["abstract"])
        # Check which patents have keyword hits in the same sentence
        .assign(has_hits=lambda df: check_keyword_hits(df.text, keywords))
        .query("has_hits == True")
        .rename(columns={"publication_number": "id"})
    )[["text", "id"]]


def get_and_process_patents_from_s3(
    patents_data_version: str = PATENTS_DATA_VERSION,
) -> pd.DataFrame:
    """Get google patents data from S3 and do light processing by combining title and abstract and removing patents without text

    Args:
        patents_data_version (str, optional): Version of patent data to download.

    Returns:
        pd.DataFrame: Google patents data
    """
    return S3.download_obj(
        os.environ["S3_BUCKET"],
        f"{PATENTS_PATH}{patents_data_version}/{patents_data_version}_processed.parquet",
        download_as="dataframe",
    )
