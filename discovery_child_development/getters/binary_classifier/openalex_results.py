from nesta_ds_utils.loading_saving import S3
from discovery_child_development import config, S3_BUCKET

PATH_FROM = "data/openAlex/test_text/"


def get_openalex_results(
    bucket: str = S3_BUCKET,
    path_from: str = PATH_FROM,
    sample_size: int = 500,
):
    """Downloads the results for a sample of the openalex data from S3, using the huggingface pipeline on the GPT labelled data.
    See discovery_child_development/pipeline/models/binary_classifier

    Args:
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.
        path_from (str, optional): Path to the data. Defaults to PATH_FROM.
        sample_size (int, optional): Number of samples from each of relevant/not relevant samples. Defaults to 500.

    Returns:
        pandas.DataFrame: A pandas dataframe with following columns:
            - id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - title (str): the title of the paper
            - abstract (str): the abstract of the paper
            - text (str): the title/abstract of the paper
            - labels (str): the label of the paper eg 1:"Relevant" or 0:"Not-relevant"
            - prediction (str): the prediction of the model eg 1:"Relevant" or 0:"Not-relevant"

    """
    filename = f"gpt_labelled_results_sample_size_{sample_size}.csv"
    openalex_data = S3.download_obj(
        bucket,
        path_from=f"{path_from}{filename}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )
    return openalex_data
