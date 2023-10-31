import os
import pandas as pd

from discovery_child_development import PROJECT_DIR, logging
from discovery_child_development.utils import google_utils
from discovery_child_development.utils.io import import_config


google_utils.find_credentials("GOOGLE_SHEETS_CREDENTIALS")
GOOGLE_SHEETS_CREDENTIALS = os.path.join(
    PROJECT_DIR, os.environ["GOOGLE_SHEETS_CREDENTIALS"]
)

PARAMS = import_config("config.yaml")
N_WORKS = PARAMS["taxonomy"]["n_works"]
SHEET_ID = PARAMS["taxonomy"]["sheet_id"]
SHEET_NAME = PARAMS["taxonomy"]["sheet_name"]


def clean_string(s):
    s = s.replace(
        "?", ""
    )  # Remove question marks (do this *before* removing trailing spaces)
    s = s.strip()  # Remove leading/trailing whitespace
    s = " ".join(s.split())  # Remove duplicate spaces
    s = s.lower()  # Convert to lowercase
    return s


def get_taxonomy(
    google_credentials_json=GOOGLE_SHEETS_CREDENTIALS,
    sheet_id=SHEET_ID,
    sheet_name=SHEET_NAME,
    works_threshold=N_WORKS,
):
    taxonomy_data = google_utils.access_google_sheet(
        google_credentials_json, sheet_id, sheet_name
    )

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
