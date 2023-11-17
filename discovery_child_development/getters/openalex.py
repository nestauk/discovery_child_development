from dotenv import load_dotenv
from nesta_ds_utils.loading_saving import S3
import os

from discovery_child_development import PROJECT_DIR, S3_BUCKET, config, logging

load_dotenv()

CONCEPT_IDS = "|".join(config["openalex_concepts"])
YEARS = [str(y) for y in config["openalex_years"]]
YEARS = "-".join(YEARS)
FILEPATH_PROCESSED = (
    f"data/openAlex/processed/openalex_data_{CONCEPT_IDS}_year-{YEARS}_train.csv"
)
VECTORS_FILEPATH = "data/openAlex/vectors/sentence_vectors_384.parquet"
SCORE_THRESHOLD = 0.3


def get_abstracts(concepts=CONCEPT_IDS, years=YEARS, bucket=S3_BUCKET):
    """Downloads OpenAlex text data (titles and abstracts) from S3.

    Args:
        concepts (str, optional): The concept IDs used in the metaflow. Defaults to CONCEPT_IDS.
        years (str, optional): The years for which we have data eg "2019_2020". Defaults to YEARS.
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.

    Returns:
        pandas.DataFrame: A pandas dataframe with following columns:
            - id (str): the id of the paper eg "https://openalex.org/W4249228678"
            - title (str): the title of the paper
            - abstract (str): the abstract of the paper
            - text (str): the concatenation of title and abstract
    """

    abstracts_filename = f"openalex_abstracts_{concepts}_year-{years}.csv"
    openalex_data = S3.download_obj(
        bucket,
        path_from=f"data/openAlex/{abstracts_filename}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )
    return openalex_data


def get_concepts_metadata(concepts=CONCEPT_IDS, years=YEARS, bucket=S3_BUCKET):
    """_summary_

    Args:
        concepts (str, optional): The concept IDs used in the metaflow. Defaults to CONCEPT_IDS.
        years (str, optional): The years for which we have data eg "2019_2020". Defaults to YEARS.
        bucket (str, optional): Name of the bucket where the data is stored. Defaults to S3_BUCKET.

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
    concepts_file = f"concepts_metadata_{concepts}_year-{years}.csv"

    openalex_concepts = S3.download_obj(
        bucket,
        path_from=f"data/openAlex/concepts/{concepts_file}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )

    return openalex_concepts


def get_labelled_data(
    filepath=FILEPATH_PROCESSED,
    score_threshold=SCORE_THRESHOLD,
    s3_bucket=S3_BUCKET,
    train=True,
):
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


def get_sentence_embeddings(s3_bucket=S3_BUCKET, filepath=VECTORS_FILEPATH):
    # Load embeddings
    embeddings = S3.download_obj(
        s3_bucket,
        path_from=filepath,
        download_as="dataframe",
    )

    embeddings = embeddings.set_index("openalex_id")
    return embeddings
