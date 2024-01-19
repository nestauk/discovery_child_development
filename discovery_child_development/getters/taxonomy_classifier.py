from datasets import Dataset
import pandas as pd
from typing import Any, Dict, List, Optional

from nesta_ds_utils.loading_saving import S3 as nesta_s3

from discovery_child_development import (
    PROJECT_DIR,
    logging,
    config,
    S3_BUCKET,
    taxonomy_config,
)
from discovery_child_development.utils.utils import create_directory_if_not_exists
from discovery_child_development.utils import jsonl_utils as jsonl

# get_labelling_sample
S3_LABELLING_DATA = (
    "data/labels/taxonomy_classifier/training_validation_data_patents_openalex.jsonl"
)
LOCAL_PATH = PROJECT_DIR / "inputs/data/labelling/taxonomy/input"
LOCAL_FILE = f"{LOCAL_PATH}/training_validation_data_patents_openalex.jsonl"
create_directory_if_not_exists(LOCAL_PATH)

# get_gpt_labelled_sample
GPT_LABELLED_DATA = "data/labels/taxonomy_classifier/labelled_with_gpt/training_validation_data_patents_openalex_GPT_LABELLED.parquet"

# get_prodigy_labelled_data
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

# get_training_data
TAXONOMY_CLASSIFIER_S3_PATH = taxonomy_config["s3_data_path"]
TAXONOMY_CLASSIFIER_INPUT_FILENAME = taxonomy_config["s3_filename"]

# get_sentence_embeddings
VECTORS_PATH = taxonomy_config["s3_vectors_path"]
VECTORS_FILE = taxonomy_config["s3_vectors_name"]

# huggingface datasets
HF_PATH = taxonomy_config["s3_hf_ds_path"]
HF_FILE = taxonomy_config["s3_hf_ds_file"]


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


def get_training_data(
    split: str = "train",
    s3_bucket: str = S3_BUCKET,
    s3_path: str = TAXONOMY_CLASSIFIER_S3_PATH,
    s3_filename: str = TAXONOMY_CLASSIFIER_INPUT_FILENAME,
) -> tuple[pd.DataFrame, str]:
    """
    Downloads and returns labelled data for the taxonomy classifier, either the training/test/validation set.
    This data was created with `discovery_child_development/pipeline/models/taxonomy_classifier/01_train_test_split.py`.

    Parameters:
    - split (str): The dataset split type. Allowed values are 'train', 'test', 'val'.
    - s3_bucket (str): The name of the S3 bucket.
    - s3_path (str): The path within the S3 bucket where the data is located.

    Returns:
    - DataFrame: The downloaded data as a pandas DataFrame.
    - str: The file path in the S3 bucket where the data was downloaded from (this is useful for logging on wandb)

    The resulting DataFrame contains the following columns:
    - 'id': A unique identifier for the text.
    - 'text': The text content that was labelled.
    - 'source': The source of the text content (OpenAlex or patents)
    - 'cost': The cost associated with the labelling.
    - 'labels_raw': The raw labels assigned by GPT.
    - 'labels': The cleaned labels. For more info on the cleaning that was done, see the function `map_keywords_to_categories()` in `taxonomy_labelling_utils`.
    """

    if split not in ["train", "test", "val"]:
        raise ValueError(
            f"Invalid value for 'split': {split}. Allowed values are 'train', 'test', 'val'."
        )

    s3_file = s3_path + s3_filename.replace("SPLIT", split)

    return nesta_s3.download_obj(s3_bucket, s3_file, download_as="dataframe"), s3_file


def get_sentence_embeddings(
    split: str = "train",
    s3_bucket: str = S3_BUCKET,
    vectors_path: str = VECTORS_PATH,
    vectors_file: str = VECTORS_FILE,
) -> pd.DataFrame:
    """
    Download sentence embeddings from an S3 bucket and return them as a pandas DataFrame.

    Parameters:
    split (str): The data split to use:'train', 'test', or 'val'.
    s3_bucket (str): The name of the S3 bucket where the sentence embeddings are stored.
    vectors_path (str): The path within the S3 bucket where the vectors files are located.
    vectors_file (str): The filename template for the vectors file. It must contain 'SPLIT' which will be replaced by the 'split' parameter.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the sentence embeddings.

    The dataframe has the columns:
    - 'id': A unique identifier for the text.
    - 'miniLM_384_vector': Each element is a Series of 384 floats, representing the sentence embedding.
    """

    filepath = f"{vectors_path}{vectors_file.replace('SPLIT', split)}"

    return nesta_s3.download_obj(
        s3_bucket,
        path_from=filepath,
        download_as="dataframe",
    )


def get_hf_ds(
    s3_bucket: str = S3_BUCKET,
    s3_path: str = HF_PATH,
    s3_file: str = HF_FILE,
    set_type: str = "train",
    production: bool = False,
) -> Optional[Dataset]:
    """
    Downloads the train or validation HuggingFace dataset from the S3 bucket.

    Args:
    s3_bucket (str): The name of the S3 bucket.
    s3_path (str): The path within the S3 bucket.
    s3_file (str): The file name in the S3 bucket, with 'SPLIT' as a placeholder for the dataset type.
    set_type (str): The type of the dataset to download ('train' or 'val').
    production (bool): If True, downloads the full dataset. If False, downloads a smaller dataset.

    Returns:
    datasets.Dataset or None: The downloaded Hugging Face dataset, or None if the download fails.
    """
    if production:
        filepath = f"{s3_path}{s3_file.replace('SPLIT', set_type)}"
    else:
        filepath = f"{s3_path}test_{s3_file.replace('SPLIT', set_type)}"

    hf_ds = nesta_s3.download_obj(s3_bucket, path_from=filepath)

    return hf_ds
