from enum import Enum
import pandas as pd

from nesta_ds_utils.loading_saving import S3 as nesta_s3

from discovery_child_development import PROJECT_DIR, logging, config, S3_BUCKET
from discovery_child_development.utils import google_utils
from discovery_child_development.utils.utils import create_directory_if_not_exists

N_WORKS = config["taxonomy"]["n_works"]
SHEET_ID = config["taxonomy"]["sheet_id"]
SHEET_NAME = config["taxonomy"]["sheet_name"]

GPT_LABELLED_DATA = "data/labels/taxonomy_classifier/labelled_with_gpt/training_validation_data_patents_openalex_GPT_LABELLED_test.parquet"

LOCAL_PATH_LABELLED_DATA = PROJECT_DIR / "inputs/data/labelling/taxonomy/output"
create_directory_if_not_exists(LOCAL_PATH_LABELLED_DATA)

TAXONOMY_CLASSIFIER_S3_PATH = "data/taxonomy_classifier/"


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


def get_gpt_labelled_sample(
    s3_bucket: str = S3_BUCKET, s3_file: str = GPT_LABELLED_DATA
) -> pd.DataFrame:
    """
    Downloads a dataset of OpenAlex abstracts and patents labelled with GPT.

    This function retrieves a dataset that has been processed by feeding the output
    of the `get_labelling_sample()` function to a GPT model for labelling. The dataset
    contains OpenAlex abstracts and patents, each labelled with GPT. The labels include
    topics and categories relevant to the content of each abstract or patent.

    The dataset is structured as a DataFrame with columns such as 'id', 'text', 'source',
    'cost', 'labels_raw', and 'labels'. Each row represents an individual abstract or
    patent, including its textual content, source (e.g., patents), associated cost for
    processing, raw labels, and processed labels.

    Parameters:
    - s3_bucket (str): The name of the S3 bucket where the data is stored.
    - s3_file (str): The specific file in the S3 bucket to be downloaded.

    Returns:
    - DataFrame: A pandas DataFrame with the following columns:
                'id' - unique identifier for the abstract/patent
                'text' - text of the abstract/patent
                'source' - either OpenAlex or patents
                'cost' - cost of the GPT input and output for this text
                'labels_raw' - raw labels output by GPT
                'labels' - cleaned, processed labels
    """
    return nesta_s3.download_obj(s3_bucket, s3_file, download_as="dataframe")


def get_training_data(
    split: str = "train",
    s3_bucket: str = S3_BUCKET,
    s3_path: str = TAXONOMY_CLASSIFIER_S3_PATH,
) -> pd.DataFrame:
    """
    Downloads and returns labelled data for the taxonomy classifier, either the training/test/validation set.
    This data was created with `discovery_child_development/pipeline/models/taxonomy_classifier/01_train_test_split.py`.

    Parameters:
    - split (str): The dataset split type. Allowed values are 'train', 'test', 'val'.
    - s3_bucket (str): The name of the S3 bucket.
    - s3_path (str): The path within the S3 bucket where the data is located.

    Returns:
    - DataFrame: The downloaded data as a pandas DataFrame.
    """

    if split not in ["train", "test", "val"]:
        raise ValueError(
            f"Invalid value for 'split': {split}. Allowed values are 'train', 'test', 'val'."
        )

    filename = f"taxonomy_labelled_data_{split}.parquet"

    s3_file = s3_path + filename

    return nesta_s3.download_obj(s3_bucket, s3_file, download_as="dataframe")
