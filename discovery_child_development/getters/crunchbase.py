"""
Getters for Crunchbase data
"""
import os
import pandas as pd

from nesta_ds_utils.loading_saving import S3

DATA_PATH = "data/crunchbase/"
DATA_VERSION = "Crunchbase_2024-05-12"


def get_cb_from_s3(
    table: str,
    data_version: str = DATA_VERSION,
) -> pd.DataFrame:
    """Get Crunchbase data from S3

    Args:
        table (str): Table name to download
        data_version (str, optional): Version of data to download.

    Returns:
        pd.DataFrame: Crunchbase data
    """
    return S3.download_obj(
        os.environ["S3_BUCKET"],
        f"{DATA_PATH}{data_version}/{table}.parquet",
        download_as="dataframe",
    )
