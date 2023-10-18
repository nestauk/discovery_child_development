"""
This module contains functions for establishing a Google BigQuery client

Usage:
from discovery_child_development.utils.bigquery import create_client
client = create_client()

"""
from google.oauth2.service_account import Credentials
from google.cloud import bigquery

from discovery_child_development import PROJECT_DIR, BUCKET_NAME, logging
from nesta_ds_utils.loading_saving.S3 import download_file

from os import environ, path
import dotenv

dotenv.load_dotenv()


def find_credentials(credentials_env_var: str):
    # Check if the environment variable is set
    if credentials_env_var not in environ:
        raise EnvironmentError("The environment variable is not set.")

    credentials_json = PROJECT_DIR / environ.get(credentials_env_var)

    if not path.isfile(credentials_json):
        logging.info("Credentials not found. Downloading from S3...")
        try:
            download_file(
                path_from=f"credentials/{credentials_json.name}",
                bucket=BUCKET_NAME,
                path_to=str(credentials_json),
            )
        except Exception as e:
            raise Exception(f"Error downloading credentials from S3: {e}")

    return credentials_json


def create_client() -> bigquery.Client:
    """
    Instantiate Google BigQuery client to query data.

    Assumes service account key is saved at a path defined in the
    .env file as GOOGLE_APPLICATION_CREDENTIALS=<path>
    If credentials are not found in the specified location, then
    the function will downloads them from s3.

    Returns instantiated bigquery client with passed credentials.

    Raises:
        EnvironmentError: If the GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.
        Exception: If there's an error downloading the credentials from S3.
    """
    credentials_json = find_credentials("GOOGLE_APPLICATION_CREDENTIALS")

    # Load credentials from the service account key JSON file
    credentials = Credentials.from_service_account_file(credentials_json)

    # Initialize a client with the provided credentials
    client = bigquery.Client(credentials=credentials)
    return client
