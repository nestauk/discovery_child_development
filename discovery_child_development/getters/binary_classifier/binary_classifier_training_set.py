from nesta_ds_utils.loading_saving import S3
from discovery_child_development import config, S3_BUCKET

CONCEPT_IDS = "|".join(config["openalex_concepts"])
BROAD_CONCEPT_IDS = "|".join(config["openalex_broad_concepts"])
YEARS = [str(y) for y in config["openalex_years"]]
YEARS = "-".join(YEARS)

PATH_FROM = "data/openAlex/processed/binary_classifier/"


def get_text_training(
    concepts=CONCEPT_IDS,
    years=YEARS,
    identifier="all",
    bucket=S3_BUCKET,
    path_from=PATH_FROM,
):
    """Downloads OpenAlex text data (titles and abstracts) from S3. Contains all the data used
    for training the binary classifier.

    Args:
        concepts (str, optional): The concept IDs used in the metaflow. Defaults to CONCEPT_IDS.
        years (str, optional): The years for which we have data eg "2019_2020". Defaults to YEARS.
        identifier (str, optional): The identifier for the training data. Defaults to "all". Options are "50", "20", "all".
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.

    Returns:
        pandas.DataFrame: A pandas dataframe with following columns:
            - id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - title (str): the title of the paper
            - abstract (str): the abstract of the paper
            - text (str): the concatenation of title and abstract
    """
    abstracts_filename = f"openalex_data_{concepts}_year-{years}_{identifier}_train.csv"
    openalex_data = S3.download_obj(
        bucket,
        path_from=f"{path_from}{abstracts_filename}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )
    return openalex_data
