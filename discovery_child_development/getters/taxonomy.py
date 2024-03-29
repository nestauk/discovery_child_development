import pandas as pd

from nesta_ds_utils.loading_saving import S3 as nesta_s3

from discovery_child_development import PROJECT_DIR, config, S3_BUCKET
from discovery_child_development.utils import google_utils

N_WORKS = config["taxonomy"]["n_works"]
SHEET_ID = config["taxonomy"]["sheet_id"]
SHEET_NAME = config["taxonomy"]["sheet_name"]


def clean_string(s: str):
    """Custom string cleaning function for the taxonomy.

    Because the taxonomy data was edited manually in Google Sheets, we need to clean up
    normal human errors like adding one too many spaces, inconsistent capitalisation, typos etc.

    Args:
        s (str): A string to be cleaned eg "Statistical methods ?"

    Returns:
        str: A cleaned string eg "statistical methods"
    """
    s = s.replace(
        "?", ""
    )  # Remove question marks (do this *before* removing trailing spaces)
    s = s.strip()  # Remove leading/trailing whitespace
    s = " ".join(s.split())  # Remove duplicate spaces
    s = s.lower()  # Convert to lowercase
    return s


def get_taxonomy(
    sheet_id: str = SHEET_ID,
    sheet_name: str = SHEET_NAME,
    works_threshold: int = N_WORKS,
) -> pd.DataFrame:
    """
    Retrieves and processes taxonomy data from a specified Google Sheet.

    The data is filtered to include only relevant entries with
    a number of works above a given threshold. It also cleans the 'Sub category'
    column using a custom string cleaning function (defined above).

    Parameters:
        sheet_id (str): The unique identifier for the Google Sheets file. Defaults to SHEET_ID.
        sheet_name (str): The name of the individual sheet within the Google Sheets file. Defaults to SHEET_NAME.
        works_threshold (int): The minimum number of works for the taxonomy entries to be included. Defaults to N_WORKS.

    Returns:
        pandas.DataFrame: A DataFrame containing the cleaned and filtered taxonomy data.
    """
    taxonomy_data = google_utils.access_google_sheet(sheet_id, sheet_name)

    # convert this column so that we can filter on it more easily
    taxonomy_data["n_works"] = taxonomy_data["n_works"].astype("int64")

    taxonomy_data = (
        taxonomy_data
        # keep only concepts/categories that have been marked relevant
        # and only concepts with at least 10 works
        .query(f"`Relevant?` == 'Y' and `n_works` >= {works_threshold}")
        # Remove noise and typos in the "Sub category" column
        .assign(**{"Sub category": lambda df: df["Sub category"].apply(clean_string)})
        # The string cleaning function leaves behind some empty strings (previously "?") so we can now get rid of these
        .query("`Sub category` != ''").rename(columns={"Sub category": "sub_category"})
    )

    return taxonomy_data
