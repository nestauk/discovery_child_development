from dotenv import load_dotenv
from nesta_ds_utils.loading_saving import S3
import pandas as pd
from typing import Optional, Tuple

from discovery_child_development import PROJECT_DIR, S3_BUCKET, config, logging
from discovery_child_development.utils import utils

load_dotenv()

FILEPATH_PROCESSED = utils.get_latest_subfolder(
    S3_BUCKET, "data/openAlex/processed/taxonomy_classifier"
)
TEST_TRAIN_FILENAME = "openalex_data_train.csv"


SCORE_THRESHOLD = 0.3

INPUT_DATA_PATH = "data/openAlex"
INPUT_DATA_PATH = utils.get_latest_subfolder(S3_BUCKET, INPUT_DATA_PATH)
ABSTRACTS_FILENAME = "openalex_abstracts.csv"
CONCEPTS_METADATA_FILENAME = "concepts_metadata.csv"

VECTORS_FILEPATH = utils.get_latest_subfolder(S3_BUCKET, "data/openAlex/vectors")
VECTORS_FILENAME = "sentence_vectors_384.parquet"


def get_abstracts(
    bucket: str = S3_BUCKET, input_path: str = INPUT_DATA_PATH
) -> pd.DataFrame:
    """Downloads OpenAlex text data (titles and abstracts) from S3.

    Args:
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.
        input_path (str, optional): Path to the folder where the data is stored. Defaults to INPUT_DATA_PATH.

    Returns:
        pandas.DataFrame: A pandas dataframe with following columns:
            - id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - title (str): the title of the paper
            - abstract (str): the abstract of the paper
            - text (str): the concatenation of title and abstract
    """
    openalex_data = S3.download_obj(
        bucket,
        path_from=f"{input_path}{ABSTRACTS_FILENAME}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )
    return openalex_data


def get_concepts_metadata(
    bucket: str = S3_BUCKET, input_path: str = INPUT_DATA_PATH
) -> pd.DataFrame:
    """Downloads OpenAlex concepts metadata from S3.

    Args:
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.
        input_path (str, optional): Path to the folder where the data is stored. Defaults to INPUT_DATA_PATH.

    Returns:
        pandas.DataFrame: A dataframe with multiple rows per OpenAlex ID,
            and one row per concept per ID. The columns are:
            - openalex_id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - title (str): the title of the paper
            - year (int): the year of the paper
            - concept_id (str): the id of the concept eg "https://openalex.org/C71924100"
            - wikidata (str): the wikidata id of the concept eg "https://www.wikidata.org/wiki/Q11190"
            - display_name (str): the display name of the concept eg "Medicine"
            - level (int): the level of the concept in the hierarchy. 0 = least granular, 5 = most granular.
            - score (float): the score of the concept for this paper. Higher score means more relevant.

             ```
            openalex_id                       title                                               year  concept_id                    wikidata                               display_name                 level  score
            0  https://openalex.org/W4249228678  REPRINT OF: Relationship of Childhood Abuse an...  2019  https://openalex.org/C71924100  https://www.wikidata.org/wiki/Q11190  Medicine                     0     0.669585
            1  https://openalex.org/W4249228678  REPRINT OF: Relationship of Childhood Abuse an...  2019  https://openalex.org/C2992354236 https://www.wikidata.org/wiki/Q43414  Sexual abuse                4     0.583067
            2  https://openalex.org/W4249228678  REPRINT OF: Relationship of Childhood Abuse an...  2019  https://openalex.org/C118552586  https://www.wikidata.org/wiki/Q7867   Psychiatry                  1     0.529665
            3  https://openalex.org/W4249228678  REPRINT OF: Relationship of Childhood Abuse an...  2019  https://openalex.org/C190385971  https://www.wikidata.org/wiki/Q373494 Injury prevention          3     0.483131
            4  https://openalex.org/W4249228678  REPRINT OF: Relationship of Childhood Abuse an...  2019  https://openalex.org/C187155963  https://www.wikidata.org/wiki/Q629029 Occupational safety and...  2     0.471979
            ```
    """
    openalex_concepts = S3.download_obj(
        bucket,
        path_from=f"{input_path}{CONCEPTS_METADATA_FILENAME}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )

    return openalex_concepts


def get_labelled_data(
    filepath: str = FILEPATH_PROCESSED,
    filename: str = TEST_TRAIN_FILENAME,
    score_threshold: float = SCORE_THRESHOLD,
    s3_bucket: str = S3_BUCKET,
    train: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Downloads preprocessed OpenAlex data from S3 (either the training&validation, or the test set),
    and filters it to only include papers with a score above a threshold.

    Args:
        filepath (str, optional): Path to the data within the bucket. Defaults to FILEPATH_PROCESSED.
        filename (str, optional): Name of the csv. Defaults to TEST_TRAIN_FILENAME.
        score_threshold (float, optional): Get rid of concept/paper combinations where the OA algorithm's confidence in a concept is below this threshold. Defaults to SCORE_THRESHOLD.
        s3_bucket (str, optional): Name of the bucket where data is stored. Defaults to S3_BUCKET.
        train (bool, optional): Should this be the training set or the test set? Defaults to True (training set).

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing a dataframe with one row per paper/concept combination; and the file path as a string.
    """
    filepath = f"{filepath}{filename}"

    if train == True:
        filepath = filepath
    else:
        filepath = str.replace(filepath, "train", "test")

    openalex_data = S3.download_obj(
        s3_bucket,
        path_from=filepath,
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )

    openalex_filtered = openalex_data[openalex_data["score"] >= score_threshold]

    return openalex_filtered, filepath


def get_sentence_embeddings(
    s3_bucket: str = S3_BUCKET,
    filepath: str = VECTORS_FILEPATH,
    filename: str = VECTORS_FILENAME,
) -> pd.DataFrame:
    # Load embeddings
    embeddings = S3.download_obj(
        s3_bucket,
        path_from=f"{filepath}{filename}",
        download_as="dataframe",
    )

    embeddings = embeddings.set_index("openalex_id")
    return embeddings
