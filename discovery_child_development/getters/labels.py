"""Module for getting labelled data from S3"""
import pandas as pd
from pathlib import Path
from discovery_child_development import S3_BUCKET, PROJECT_DIR, get_yaml_config
from discovery_child_development.utils import jsonl_utils
from nesta_ds_utils.loading_saving import S3


def get_relevance_labels() -> pd.DataFrame:
    """Get relevance labels from S3

    Returns:
        pd.DataFrame: A DataFrame containing the relevance labels.
            Columns are:
                - id: The ID of the work
                - source: The source of the data (so far: openalex or patent)
                - text: The text of the work (title + abstract)
                - prediction: Relevant (is about preschool-age child development),
                    Not-relevant, Not-specified (might be about child development but age unclear)
                - model: The model used to make the prediction
                - timestamp: The timestamp of the prediction
    """
    config = get_yaml_config(
        PROJECT_DIR
        / "discovery_child_development/pipeline/labelling/relevance/config.yaml"
    )
    local_path = f'{config["local_output_directory"]}/{config["output_filename"]}.jsonl'
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    data = jsonl_utils.download_file_from_s3(
        bucket_name=S3_BUCKET,
        s3_file_name=f'{config["s3_directory"]}{config["output_filename"]}.jsonl',
        local_file=f"{str(PROJECT_DIR)}/{local_path}",
    )
    return pd.DataFrame(data)


def get_taxonomy_labels(raw: bool = False) -> pd.DataFrame:
    """Get relevance labels from S3

    If raw=True, then downloads the unprocessed outputs from OpenAI API.
    Otherwises, fetches the slighty cleaned prediction outptus. In each case,
    you can use the 'predictions' column for the labels.

    Returns:
        pd.DataFrame: A DataFrame containing the relevance labels.
            Columns are:
                - id: The ID of the work
                - source: The source of the data (so far: openalex or patent)
                - text: The text of the work (title + abstract)
                - prediction: Relevant (is about preschool-age child development),
                    Not-relevant, Not-specified (might be about child development but age unclear)
                - model: The model used to make the prediction
                - timestamp: The timestamp of the prediction
    """
    config = get_yaml_config(
        PROJECT_DIR
        / "discovery_child_development/pipeline/labelling/taxonomy_v2/config.yaml"
    )
    if raw:
        fname = config["output_filename"] + "_raw.jsonl"
        local_path = f'{config["local_output_directory"]}/{fname}'
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        data = jsonl_utils.download_file_from_s3(
            bucket_name=S3_BUCKET,
            s3_file_name=f'{config["s3_directory"]}{fname}',
            local_file=f"{str(PROJECT_DIR)}/{local_path}",
        )
        return pd.DataFrame(data)
    else:
        return S3.download_obj(
            bucket=S3_BUCKET,
            path_from=f'{config["s3_directory"]}{config["output_filename"]}.parquet',
            download_as="dataframe",
        )


def get_detection_management_labels() -> pd.DataFrame:
    """Get relevance labels from S3

    Returns:
        pd.DataFrame: A DataFrame containing the relevance labels.
            Columns are:
                - id: The ID of the work
                - source: The source of the data (so far: openalex or patent)
                - text: The text of the work (title + abstract)
                - prediction: Relevant (is about preschool-age child development),
                    Not-relevant, Not-specified (might be about child development but age unclear)
                - model: The model used to make the prediction
                - timestamp: The timestamp of the prediction
    """
    config = get_yaml_config(
        PROJECT_DIR
        / "discovery_child_development/pipeline/labelling/detection_management/config.yaml"
    )
    local_path = f'{config["local_output_directory"]}/{config["output_filename"]}.jsonl'
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    data = jsonl_utils.download_file_from_s3(
        bucket_name=S3_BUCKET,
        s3_file_name=f'{config["s3_directory"]}{config["output_filename"]}.jsonl',
        local_file=local_path,
    )
    return pd.DataFrame(data)
