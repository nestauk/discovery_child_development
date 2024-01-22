"""Module for getting unlabeled data from S3 to be labelled"""
import pandas as pd
from discovery_child_development import S3_BUCKET
from nesta_ds_utils.loading_saving import S3


def get_unprocessed_data(config: dict) -> pd.DataFrame:
    """Get unprocessed data from S3

    Args:
        config (dict): The config dictionary for the inference pipeline

    Returns:
        pd.DataFrame: A DataFrame to be labelled. Columns must include the following variables:
                - id: The ID of the work
                - text: The text of the work (title + abstract)
    """
    extension = config["data_extension"]
    if extension in ["csv", "parquet", "xlsx", "xlsm"]:
        return S3.download_obj(
            bucket=S3_BUCKET,
            path_from=f"{config['DATA_PATH']}.{extension}",
            download_as="dataframe",
            kwargs_reading={"index_col": 0},
        )
    elif extension == "json":
        json_file = S3.download_obj(
            bucket=S3_BUCKET,
            path_from=f"{config['DATA_PATH']}.{extension}",
            download_as="dict",
        )
        # Convert to dataframe
        return pd.DataFrame(json_file)
    else:
        # Raise an error if the extension is not supported
        raise ValueError(
            f"The extension {extension} is not supported. Please use one of the following: csv, parquet, xlsx, xlsm, json"
        )


def get_data_for_relevance_classifier(config: dict) -> pd.DataFrame:
    """Get data from S3

    Args:
        config (dict): The config dictionary for the inference pipeline

    Returns:
        pd.DataFrame: A DataFrame to be labelled. Columns must include the following variables:
                - id: The ID of the work
                - text: The text of the work (title + abstract)
    """
    extension = config["extension"]
    if extension in ["csv", "parquet", "xlsx", "xlsm"]:
        return S3.download_obj(
            bucket=S3_BUCKET,
            path_from=f"{config['S3_PATH']}{config['FNAME']}.{extension}",
            download_as="dataframe",
            kwargs_reading={"index_col": 0},
        )
    elif extension == "json":
        json_file = S3.download_obj(
            bucket=S3_BUCKET,
            path_from=f"{config['S3_PATH']}{config['FNAME']}.{extension}",
            download_as="dict",
        )
        # Convert to dataframe
        return pd.DataFrame(json_file)
    else:
        # Raise an error if the extension is not supported
        raise ValueError(
            f"The extension {extension} is not supported. Please use one of the following: csv, parquet, xlsx, xlsm, json"
        )
