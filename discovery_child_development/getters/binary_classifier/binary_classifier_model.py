from nesta_ds_utils.loading_saving import S3
from discovery_child_development import config, S3_BUCKET
from discovery_child_development.utils.general_utils import extract_tarfile


def get_binary_classifier_models(
    filename: str,
    s3_path: str,
    path_to: str,
):
    """Downloads the binary classifier model from S3.

    Args:
        filename (str): The name of the model.
        s3_path (str): The path to the model on S3.
        path_to (str): The path to the model on the local machine.

    Returns:
        Model: The binary classifier model.
    """

    # Downloads the tar.gz file from S3
    S3.download_file(
        path_from=f"{s3_path}{filename}",
        bucket=S3_BUCKET,
        path_to=f"{path_to}{filename}",
    )

    # Unzips the tar.gz file
    extract_tarfile(filename, path_to)
