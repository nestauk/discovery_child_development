from nesta_ds_utils.loading_saving import S3
from discovery_child_development import S3_BUCKET

TRAINING_DATA_PATH = "data/labels/detection_management_classifier/processed/"


def get_training_data(
    set_type="train",
    bucket=S3_BUCKET,
    path_from=TRAINING_DATA_PATH,
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
