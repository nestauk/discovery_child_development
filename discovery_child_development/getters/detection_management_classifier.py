from nesta_ds_utils.loading_saving import S3
from discovery_child_development import S3_BUCKET

TRAINING_DATA_PATH = "data/labels/detection_management_classifier/processed/"


def get_training_data(
    set_type: str = "train",
    bucket: str = S3_BUCKET,
    path_from: str = TRAINING_DATA_PATH,
):
    """Downloads the GPT labelled training/validation/test data from S3. Contains all the data used
    for each dataset for training the classifier.

    Args:
        set_type (str, optional): Whether to use the training/validation/testing set. Defaults to "training". Options are "train", "validation", "test".
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.

    Returns:
        pandas.DataFrame: A pandas dataframe with following columns:
            - id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - source (str): the source of the paper eg "openalex" or "patents"
            - title (str): the title of the paper
            - labels (str): the label of the paper
    """
    abstracts_filename = f"gpt_labelled_{set_type}.csv"
    return S3.download_obj(
        bucket,
        path_from=f"{path_from}{abstracts_filename}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )


def get_hf_dataset(
    vectors_path: str,
    vectors_file: str,
    identifier: str = "all",
    bucket: str = S3_BUCKET,
    production: bool = True,
    set_type: str = "train",
):
    """Downloads OpenAlex embeddings for the training set from S3. Contains all the data used
    for training the simple binary classifier.

    Args:
        identifier (str, optional): The identifier for the training data. Defaults to "all". Options are "50", "20", "all".
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.
        vectors_path (str, optional): The path to the vectors. Defaults to VECTORS_PATH.
        vectors_file (str, optional): The name of the vectors file. Defaults to VECTORS_FILE.
        production (bool, optional): Whether to use the production data. Defaults to True.
        set_type (str, optional): Whether to use the training/validation/testing set. Defaults to "train". Options are "train", "validation".

    Returns:
        pandas.DataFrame: Dictionary with following keys:
            - openalex_id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - input_ids (list): the input ids of the paper
            - attention_mask (list): the attention mask of the paper
            - label (int): the label of the paper
    """
    if production:
        if identifier in ["20", "50", "all"]:
            embedding_filename = f"{vectors_file}_{identifier}_{set_type}.pkl"
        else:
            embedding_filename = f"{vectors_file}_{set_type}.pkl"
    else:
        if identifier in ["20", "50", "all"]:
            embedding_filename = f"{vectors_file}_test_{identifier}_{set_type}.pkl"
        else:
            embedding_filename = f"{vectors_file}_test_{set_type}.pkl"

    openalex_data = S3.download_obj(
        bucket, path_from=f"{vectors_path}{embedding_filename}"
    )
    return openalex_data
