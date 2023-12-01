import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import datetime
import re


def list_subfolders_in_s3(bucket_name, parent_folder):
    """
    List all subfolders within a specified parent directory in an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        parent_folder (str): The parent directory path in the S3 bucket.

    Returns:
        list: A list of subfolder paths.
    """
    s3 = boto3.client("s3")

    # Ensure the parent_folder format is correct
    if not parent_folder.endswith("/"):
        parent_folder += "/"

    response = s3.list_objects_v2(
        Bucket=bucket_name, Prefix=parent_folder, Delimiter="/"
    )
    subfolders = [content["Prefix"] for content in response.get("CommonPrefixes", [])]

    return subfolders


def list_objects_in_subfolder(bucket_name, subfolder, search_string=None):
    s3 = boto3.client("s3")
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=subfolder)
    objects = [obj["Key"].split("/")[-1] for obj in objects["Contents"]]
    if search_string:
        pattern = re.compile(search_string)
        objects = [obj for obj in objects if pattern.search(obj)]

    return objects


def parse_timestamp_from_folder_name(folder_name):
    """Parse the datetime object from the folder name."""

    pattern = r"\d{8}_\d{6}"
    match = re.search(pattern, folder_name)
    if not match:
        return None
    else:
        try:
            return datetime.datetime.strptime(match.group(0), "%Y%m%d_%H%M%S")
        except ValueError:
            return None


def get_latest_subfolder(bucket_name, parent_folder, production=True):
    """
    Get the subfolder with the most recent timestamp that optionally contains a specific string.

    Args:
        bucket_name (str): The name of the S3 bucket.
        parent_folder (str): The parent directory path in the S3 bucket.
        filter_str (str, optional): A string to filter the subfolders. Defaults to None.

    Returns:
        str: The path of the latest subfolder, or None if not found.
    """
    subfolders = list_subfolders_in_s3(bucket_name, parent_folder)

    if production:
        subfolders = [folder for folder in subfolders if "production_True" in folder]

    # Extract timestamps and filter out invalid folder names
    parsed_folders = [
        (folder, parse_timestamp_from_folder_name(folder)) for folder in subfolders
    ]

    valid_folders = [
        (folder, timestamp) for folder, timestamp in parsed_folders if timestamp
    ]

    # Find the folder with the latest timestamp
    if valid_folders:
        latest_folder = max(valid_folders, key=lambda x: x[1])[0]
        return latest_folder
    else:
        return None


def copy_s3_object(bucket_name, source_key, destination_key):
    # Create an S3 client
    s3 = boto3.resource("s3")

    try:
        # Copy the object
        copy_source = {"Bucket": bucket_name, "Key": source_key}
        s3.meta.client.copy(copy_source, bucket_name, destination_key)
        print(
            f"File copied from {source_key} to {destination_key} in bucket {bucket_name}"
        )

    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials")
    except Exception as e:
        print(f"Error occurred: {e}")
