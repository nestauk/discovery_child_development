import boto3
from botocore.exceptions import NoCredentialsError
import json

from discovery_child_development import logging


def upload_file_to_s3(local_file, bucket_name, s3_file_name):
    """
    Upload a file to an S3 bucket

    :param local_file: File to upload
    :param bucket_name: Bucket to upload to
    :param s3_file_name: S3 object name. If not specified then local_file is used
    """
    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        # Upload the file
        s3.upload_file(local_file, bucket_name, s3_file_name)
        print(f"File {local_file} uploaded to {bucket_name}/{s3_file_name}")
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
    except NoCredentialsError:
        print("Credentials not available")


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:
                # Skip empty lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e}")
                # Optionally, continue to next line or handle error differently
    return data


def download_file_from_s3(bucket_name, s3_file_name, local_file):
    """
    Download a file from an S3 bucket

    :param bucket_name: Bucket to download from
    :param s3_file_name: S3 object name
    :param local_file: File path to store the downloaded file
    """
    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        # Download the file
        s3.download_file(bucket_name, s3_file_name, local_file)
        logging.info(
            f"File {s3_file_name} downloaded from {bucket_name} to {local_file}"
        )

        data = load_jsonl(local_file)
    except FileNotFoundError:
        print(f"The file {s3_file_name} was not found in {bucket_name}")
    except NoCredentialsError:
        print("Credentials not available")

    return data
