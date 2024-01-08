import pandas as pd
from typing import Any, Dict, List, Optional

from nesta_ds_utils.loading_saving import S3 as nesta_s3

from discovery_child_development import PROJECT_DIR, logging, config, S3_BUCKET
from discovery_child_development.utils import google_utils
from discovery_child_development.utils import jsonl_utils as jsonl
from discovery_child_development.utils.utils import create_directory_if_not_exists

N_WORKS = config["taxonomy"]["n_works"]
SHEET_ID = config["taxonomy"]["sheet_id"]
SHEET_NAME = config["taxonomy"]["sheet_name"]

S3_LABELLING_DATA = (
    "data/labels/taxonomy_classifier/training_validation_data_patents_openalex.jsonl"
)
LOCAL_PATH = PROJECT_DIR / "inputs/data/labelling/taxonomy/input"
LOCAL_FILE = f"{LOCAL_PATH}/training_validation_data_patents_openalex.jsonl"
create_directory_if_not_exists(LOCAL_PATH)

GPT_LABELLED_DATA = "data/labels/taxonomy_classifier/labelled_with_gpt/training_validation_data_patents_openalex_GPT_LABELLED.parquet"
PRODIGY_LABELLED_DATA_FILENAME = (
    "training_validation_data_patents_openalex_LABELLED_prodigy.jsonl"
)
PRODIGY_LABELLED_DATA_FILENAME_LOCAL = (
    "training_validation_data_patents_openalex_LABELLED_prodigy_downloaded.jsonl"
)
S3_PRODIGY_DATA_PATH = (
    f"data/labels/taxonomy_classifier/{PRODIGY_LABELLED_DATA_FILENAME}"
)
LOCAL_PATH_LABELLED_DATA = PROJECT_DIR / "inputs/data/labelling/taxonomy/output"
create_directory_if_not_exists(LOCAL_PATH_LABELLED_DATA)
LOCAL_PRODIGY_DATA = LOCAL_PATH_LABELLED_DATA / PRODIGY_LABELLED_DATA_FILENAME_LOCAL


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


def get_labelling_sample(
    s3_bucket: str = S3_BUCKET,
    s3_file: str = S3_LABELLING_DATA,
    local_file: str = LOCAL_FILE,
) -> List[Dict[str, Any]]:
    """
    Balanced sample of OpenAlex and patents data for labelling.

    There should be:
    * 100 * <n_categories> OpenAlex abstracts; the 100 are taken by matching concepts metadata to the taxonomy.
        For example, the 100 samples for the taxonomy category "Nutrition and weights"
        might be tagged with the concepts "Childhood obesity", "Gut flora", "Healthy eating",
        "Malnutrition", as these are some of the concepts that this taxonomy category was derived from.
    * 100 * <n_categories> patents
    * 100 * 10 OpenAlex abstracts that do not have any concepts metadata

    Returns:
    - List[Dict[str, Any]]: A list of records where each record contains the following fields:
        - id: A unique identifier for the text.
        - text: The text content that was labelled.
        - source: The source of the text content (OpenAlex or patents)

    """
    return jsonl.download_file_from_s3(s3_bucket, s3_file, local_file)


def get_gpt_labelled_sample(
    s3_bucket: str = S3_BUCKET, s3_file: str = GPT_LABELLED_DATA
) -> pd.DataFrame:
    """Relatively balanced dataset of OpenAlex abstracts and patents, labelled with GPT.

    The output of the above function `get_labelling_sample()` was fed to GPT for labelling.

    The resulting DataFrame contains the following columns:
    - 'id': A unique identifier for the text.
    - 'text': The text content that was labelled.
    - 'source': The source of the text content (e.g., 'patents').
    - 'cost': The cost associated with the labelling.
    - 'labels_raw': The raw labels assigned by GPT.
    - 'labels': The cleaned labels. For more info on the cleaning that was done, see the function `map_keywords_to_categories()` in `taxonomy_labelling_utils`.

    Parameters:
    - s3_bucket (str): The name of the S3 bucket where the data is stored.
    - s3_file (str): The file path in the S3 bucket.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the GPT-labelled data (see above)
    """
    return nesta_s3.download_obj(s3_bucket, s3_file, download_as="dataframe")


def get_prodigy_labelled_data(
    s3_bucket: str = S3_BUCKET,
    s3_file: str = S3_PRODIGY_DATA_PATH,
    local_file: str = str(LOCAL_PRODIGY_DATA),
) -> pd.DataFrame:
    """
    Get data that has been labelled with Prodigy and stored on S3. It also saves the Prodigy
    data locally as a jsonl file at the path specified by `local_file`.

    Parameters:
    - s3_bucket (str): The name of the S3 bucket where the data is stored.
    - s3_file (str): The file path in the S3 bucket.
    - local_file (str): The local file path where the data will be downloaded.

    Returns:
    - DataFrame: A Pandas DataFrame containing the data from the specified S3 file.

    The resulting DataFrame contains the following columns:
    - 'id': A unique identifier for the text.
    - 'text': The text content that was labelled.
    - 'source': The source of the text content (OpenAlex or patents)
    - 'tokens_input': Number of GPT tokens in the input.
    - 'tokens_output': Number of GPT tokens in the output.
    - 'model': The model used for labelling.
    - 'cost': The cost associated with the labelling.
    - 'options': The options provided for labelling.
    - 'accept': The labels accepted during labelling.
    - 'model_output': The output of the model -> Evaluate the model by checking how many of these labels match the 'accept' labels.
    - '_input_hash': A hash of the input data.
    - '_task_hash': A hash of the task.
    - '_view_id': The view ID of the task.
    - 'config': Configuration used for the labelling task.
    - 'answer': 'accept' or 'ignore' - 'ignore' indicates that the text would have been filtered out by the relevance classifier in the eventual pipeline
    - '_timestamp': The timestamp when the labelling was done.
    - '_annotator_id': The ID of the annotator.
    - '_session_id': The session ID for the labelling task.

    """
    return pd.DataFrame(jsonl.download_file_from_s3(s3_bucket, s3_file, local_file))
