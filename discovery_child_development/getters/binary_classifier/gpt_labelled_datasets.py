from nesta_ds_utils.loading_saving import S3
from discovery_child_development import config, S3_BUCKET

PATH_FROM = "data/labels/binary_classifier/processed/"


def get_labelled_data_for_classifier(
    set_type="train",
    bucket=S3_BUCKET,
    path_from=PATH_FROM,
):
    """Downloads the GPT labelled training/validation/test data from S3. Contains all the data used
    for each dataset for training the binary classifier.

    Args:
        set_type (str, optional): Whether to use the training/validation/testing set. Defaults to "training". Options are "train", "validation", "test".
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.

    Returns:
        pandas.DataFrame: A pandas dataframe with following columns:
            - id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - source (str): the source of the paper eg "openalex" or "patents"
            - title (str): the title of the paper
            - labels (str): the label of the paper eg "Relevant" or "Not-relevant"
    """
    abstracts_filename = f"gpt_labelled_{set_type}.csv"
    openalex_data = S3.download_obj(
        bucket,
        path_from=f"{path_from}{abstracts_filename}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )
    return openalex_data
