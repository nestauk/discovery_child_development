from dotenv import load_dotenv
from nesta_ds_utils.loading_saving import S3
import os

from discovery_child_development.utils.io import import_config

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]

PARAMS = import_config("config.yaml")
CONCEPT_IDS = "|".join(PARAMS["openalex_concepts"])
YEARS = [str(y) for y in PARAMS["openalex_years"]]
YEARS = "-".join(YEARS)


def get_abstracts(concepts=CONCEPT_IDS, years=YEARS, bucket=S3_BUCKET):
    abstracts_filename = f"openalex_abstracts_{concepts}_year-{years}.csv"
    openalex_data = S3.download_obj(
        bucket,
        path_from=f"data/openAlex/{abstracts_filename}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )
    return openalex_data


def get_concepts_metadata(concepts=CONCEPT_IDS, years=YEARS, bucket=S3_BUCKET):
    concepts_file = f"concepts_metadata_{concepts}_year-{years}.csv"

    openalex_concepts = S3.download_obj(
        bucket,
        path_from=f"data/openAlex/concepts/{concepts_file}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )

    return openalex_concepts
