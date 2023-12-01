"""
Getters for patent data
"""
import os
import pandas as pd

from nesta_ds_utils.loading_saving import S3
from discovery_child_development import config
from discovery_child_development.utils.keywords import process_keywords

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
    """Get google patents data from S3

    Args:
        patent_data_version (str, optional): Version of patent data to download.

    Returns:
        pd.DataFrame: Google patents data
    """
    return process_keywords(
        S3.download_obj(
            os.environ["S3_BUCKET"],
            f"{PATENTS_PATH}{patents_data_version}/keywords_query.txt",
            download_as="list",
        )
    )
