"""
Getters for Gateway to Research data
"""
import os
import pandas as pd

from nesta_ds_utils.loading_saving import S3

DATA_PATH = "data/GtR/"
DATA_VERSION = "GtR_20240109"


def get_gtr_from_s3(
    table: str,
    data_version: str = DATA_VERSION,
) -> pd.DataFrame:
    """Get Gateway to Research data from S3

    Args:
        table (str): Table name to download, one of: organisations, funds, persons, projects
        data_version (str, optional): Version of data to download.

    Returns:
        pd.DataFrame: Google patents data
    """
    return S3.download_obj(
        os.environ["S3_BUCKET"],
        f"{DATA_PATH}{data_version}/gtr_{table}.json",
        download_as="dict",
    )
