"""
Utils functions for working with weights and biases
"""

import wandb
import pandas as pd


def log_dataset(
    run: wandb.sdk.wandb_run.Run,
    df: pd.DataFrame,
    local_path_to_data: str,
    name: str,
    description: str,
) -> None:
    """Log your dataset on wandb. You have to point wandb to a local file, so this function
    saves the dataframe to a local parquet, then logs it to wandb.

    Args:
        run (wandb.sdk.wandb_run.Run): The run object that you created with `wandb.init()`
        df (pd.DataFrame): pandas dataframe
        local_path_to_data (str): path to save the dataframe to
        name (str): Name that the dataset should have on wandb
        description (str): Description that the dataset should have on wandb
    """
    df.to_parquet(local_path_to_data, index=False)

    # Create an Artifact
    artifact = wandb.Artifact(name=name, type="data", description=description)
    # Add the file to the Artifact
    artifact.add_file(local_path_to_data)
    # Log the Artifact as part of the run
    run.log_artifact(artifact)


def add_ref_to_data(
    run: wandb.sdk.wandb_run.Run,
    name: str,
    description: str,
    bucket: str,
    filepath: str,
) -> None:
    """Add a wandb reference to a dataset that lives on S3. See https://docs.wandb.ai/guides/artifacts/track-external-files

    Args:
        run (wandb.sdk.wandb_run.Run): The run object that you created with `wandb.init()`
        name (str): Name that the dataset should have on wandb
        description (str): Description that the dataset should have on wandb
        bucket (str): Name of the S3 bucket
        filepath (str): Path to the file in the bucket
    """
    artifact = wandb.Artifact(name=name, type="data", description=description)
    artifact.add_reference(f"s3://{bucket}/{filepath}")
    run.log_artifact(artifact)
