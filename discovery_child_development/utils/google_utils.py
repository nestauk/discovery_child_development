"""
This module contains functions for accessing Google Sheets and Google BigQuery resources.

Usage:
import discovery_child_development.utils.google_utils as google_utils

# access data from Google Sheets
data = google_utils.access_google_sheet(<sheet_id>, <sheet_name>)

# access data from Google BigQuery
client = google_utils.create_client()

## Define the SQL query
sql = "< add your query here >"

## Execute the query
query_job = client.query(sql)

## Fetch the results
results = query_job.result()

"""
from google.oauth2.service_account import Credentials
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
from google.cloud import bigquery
from pathlib import PosixPath

from discovery_child_development import PROJECT_DIR, S3_BUCKET, logging
from nesta_ds_utils.loading_saving.S3 import download_file, upload_file, upload_obj

import pandas as pd
import datetime
from typing import List, Union
from pathlib import Path
import re
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
                bucket=S3_BUCKET,
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


def write_like_condition(term: Union[str, List[str]], table: str, field: str) -> str:
    """Create a LIKE condition for a search term or a list of search terms"""
    if len(term) == 1:
        return f'{table}.{field} LIKE "%{term[0]}%"'
    else:
        # Write an AND condition if more than one search term
        return "(" + " AND ".join([f'{table}.{field} LIKE "%{t}%"' for t in term]) + ")"


def create_patents_query(search_terms: List[str]) -> str:
    """Create a query to fetch patent data from BigQuery using search terms:
    the query checks search terms in the title and abstract.

    We access both google_patents_research.publications and google_patents.publications as
    the google_patents_research has abstracts translated into English, plus embeddings (if needed),
    while patents.publication has the grant date and other metadata

    Args:
    search_terms (List[str]): list of search terms

    Returns
    str: query to fetch data from BigQuery
    """

    # Create a list of 'LIKE' conditions for each search term
    like_conditions = {}
    for field in ["title", "abstract"]:
        like_conditions[field] = [
            write_like_condition(term, "gpr", field) for term in search_terms
        ]

    # Join conditions with 'OR'
    combined_conditions_title = " OR ".join(like_conditions["title"])
    combined_conditions_abstract = " OR ".join(like_conditions["abstract"])

    q = f"""
    WITH
    pubs as (
        SELECT DISTINCT
            pub.publication_number
        FROM `patents-public-data.patents.publications` pub
            INNER JOIN `patents-public-data.google_patents_research.publications` gpr ON
            pub.publication_number = gpr.publication_number
        WHERE
            (({combined_conditions_title})
            OR ({combined_conditions_abstract}))
            AND ((pub.grant_date BETWEEN 20130101 AND 20241231) OR (pub.filing_date BETWEEN 20130101 AND 20241231))
    )

    SELECT
        gpr.publication_number,
        pub.application_number,
        pub.family_id,
        url,
        pub.filing_date,
        pub.grant_date,
        pub.publication_date,
        pub.priority_date,
        title,
        title_translated,
        abstract,
        abstract_translated,
        top_terms,
        pub.country_code,
        pub.inventor_harmonized,
        pub.assignee_harmonized,
        pub.cpc,
    FROM `patents-public-data.patents.publications` pub
        INNER JOIN `patents-public-data.google_patents_research.publications` gpr ON
        pub.publication_number = gpr.publication_number
    WHERE
        gpr.publication_number IN (SELECT publication_number FROM pubs)
    """

    return q


def dry_run(client: bigquery.Client, query: str) -> None:
    """Dry run a query to estimate the size of the query"""
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    query_job = client.query(query, job_config=job_config)
    logging.info(
        "This query will process {} GB.".format(
            round(query_job.total_bytes_processed / 1e9, 3)
        )
    )


def upload_query_to_s3(
    query_name: str,
    path: str,
    query_df: pd.DataFrame,
    query: str,
    metadata: List[str] = None,
) -> None:
    """Upload query results to S3. Saves the results in a subfolder
    with the name {query_name}_{timestamp}, with the timestamp added
    to avoid overwriting existing results.

    Args:
        query_name (str): Arbitrary, user-chosen name of the query
        path (str): S3 path to upload to
        query_df (pd.DataFrame): Query results
        query (str): Query string
        metadata (List[str], optional): List of metadata files to upload. Defaults to None.

    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    query_name = f"{query_name}_{timestamp}"

    # Upload the data
    upload_obj(
        query_df,
        S3_BUCKET,
        f"{path}{query_name}/{query_name}.parquet",
    )
    # Upload the query
    upload_obj(
        query,
        S3_BUCKET,
        f"{path}{query_name}/{query_name}_query.txt",
    )
    # Upload any other metadata files
    if type(metadata) is not None:
        for file in metadata:
            filename = Path(file).stem + Path(file).suffix
            upload_file(
                str(file),
                S3_BUCKET,
                f"{path}{query_name}/{filename}",
            )
    logging.info(f"Query results uploaded to {path}{query_name}/")


def access_google_sheet(sheet_id: str, sheet_name: str):
    """
    Accesses a specified Google Sheet and returns its contents as a pandas DataFrame.

    This function authenticates using service account credentials, defines the scope
    for the Google Sheets API, and downloads the sheet contents. The sheet is accessed
    by its unique identifier and a specific sheet name within the spreadsheet.

    Args:
        sheet_id (str): The unique identifier for the Google Sheets file.
        sheet_name (str): The name of the individual sheet within the Google Sheets file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the specified Google Sheet.

    Raises:
        GoogleAuthError: If authentication with Google Sheets API fails.
        DownloadError: If there is an issue downloading the sheet contents.

    Notes:
    - The GOOGLE_SHEETS_CREDENTIALS environment variable must be set with the path to
      the credentials JSON file ie `.credentials/xxxxx.json`.
    - The service account must have the necessary permissions to access the Google Sheet.
    - The function assumes the first row and column of the sheet contain the header and
      index names, respectively.
    """
    # Load the credentials for use with Google Sheets
    google_credentials_json = find_credentials("GOOGLE_SHEETS_CREDENTIALS")

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
