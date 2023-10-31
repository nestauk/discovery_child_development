"""
This module contains functions for establishing a Google BigQuery client

Usage:
from discovery_child_development.utils.bigquery import create_client
client = create_client()

"""
from google.oauth2.service_account import Credentials
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
from google.cloud import bigquery
from pathlib import PosixPath

from discovery_child_development import PROJECT_DIR, BUCKET_NAME, logging
from nesta_ds_utils.loading_saving.S3 import download_file

from os import environ, path
import dotenv

dotenv.load_dotenv()


def find_credentials(credentials_env_var: str) -> PosixPath:
    """Find credentials file

    For accessing some Google resources, we need credentials stored in a JSON file in `.credentials/`.
    This function takes the name of an environment variable as input and checks whether the corresponding
    credentials file exists. If not, it downloads the file from S3.

    Args:
        credentials_env_var (str): Name of the env var eg "GOOGLE_APPLICATION_CREDENTIALS". Your .env file should have paths to Google credentials files stored like "GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials-file>".

    Raises:
        EnvironmentError: If this env var is not recorded in `.env`
        Exception: If the function can neither find the credentials file nor download it from S3

    Returns:
        PosixPath: Path to the credentials file
    """
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


def access_google_sheet(google_credentials_json, sheet_id, sheet_name):
    # Define the scope for the Google Sheets API (we only want Google sheets)
    scope = ["https://spreadsheets.google.com/feeds"]

    # Authenticate using the credentials JSON file
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        google_credentials_json, scope
    )

    # Load the data into a pandas DataFrame
    data = g2d.download(
        sheet_id, sheet_name, credentials=credentials, col_names=True, row_names=True
    )

    return data
