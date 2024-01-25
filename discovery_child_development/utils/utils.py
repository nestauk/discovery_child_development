import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import datetime
import os
import re
from typing import Optional, List
import json
import os
from typing import Generator
import yaml
from pathlib import Path


def list_subfolders_in_s3(bucket_name: str, parent_folder: str) -> List[str]:
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


def list_objects_in_subfolder(
    bucket_name: str, subfolder: str, search_string: Optional[str] = None
) -> List[str]:
    """List objects saved within a subfolder on S3

    Args:
        bucket_name (str): Name of the S3 bucket
        subfolder (str): Subfolder to search within
        search_string (Optional[str], optional): Search string to be compiled as regex. Defaults to None. If None,
        names of all objects within the folder will be returned.

    Returns:
        List[str]: _description_
    """
    s3 = boto3.client("s3")
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=subfolder)
    objects = [obj["Key"].split("/")[-1] for obj in objects["Contents"]]
    if search_string:
        pattern = re.compile(search_string)
        objects = [obj for obj in objects if pattern.search(obj)]

    return objects


def parse_timestamp_from_folder_name(folder_name: str) -> Optional[datetime.datetime]:
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


def get_latest_subfolder(
    bucket_name: str, parent_folder: str, search_string: Optional[str] = None
) -> Optional[str]:
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

    if search_string is not None:
        subfolders = [folder for folder in subfolders if search_string in folder]

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


def copy_s3_object(bucket_name: str, source_key: str, destination_key: str) -> None:
    """Copy an object from a subfolder in an S3 bucket to another path within the same bucket.

    Args:
        bucket_name (str): Name of S3 bucket
        source_key (str): Original path of the object
        destination_key (str): Destination path where the object should be copied to
    """
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


def batch(lst: list, n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_directory_if_not_exists(dir_path: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def current_time() -> str:
    """Return the current time as a string. Used as part of the session UUID."""
    # Get current date and time
    current_datetime = datetime.datetime.now()

    # Convert to a long number format
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")

    return datetime_string


def prepare_url(id: str, source: str) -> str:
    """
    Prepare the URL for the example to be displayed to the user.
    """
    if "http" in id:
        return id
    else:
        if source == "patents":
            # For Google patents remove all non alphanumeric characters
            _id = "".join([c for c in id if c.isalnum()])
            return f"https://patents.google.com/patent/{_id}"
        elif source == "openalex":
            # OpenAlex
            return f"https://openalex.org/{id}"
        else:
            raise ValueError(f"Unknown source: {source}")
