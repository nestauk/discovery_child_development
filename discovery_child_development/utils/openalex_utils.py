from typing import Dict, List, Union
from itertools import chain
import boto3
from nesta_ds_utils.loading_saving import S3 as nesta_s3
import pandas as pd

from discovery_child_development import S3_BUCKET, logging


def deinvert_abstract(inverted_abstract: Dict[str, List]) -> Union[str, None]:
    """Convert inverted abstract into normal abstract

    Args:
        inverted_abstract: a dict where the keys are words
        and the values lists of positions

    Returns:
        A str that reconstitutes the abstract or None if the deinvered abstract
        is empty

    """

    if len(inverted_abstract) == 0:
        return None
    else:
        abstr_empty = (max(chain(*inverted_abstract.values())) + 1) * [""]

        for word, pos in inverted_abstract.items():
            for p in pos:
                abstr_empty[p] = word

        return " ".join(abstr_empty)


def generate_keyword_queries(
    root: str, keywords: List[str], years: List[int]
) -> List[str]:
    """
    Generates a list of API queries based on given keywords and years.

    This function constructs queries by combining a root API endpoint with
    specified keywords and a filter for publication years. Each query is
    formed by appending a keyword and a year filter to the root endpoint.

    Args:
    root (str): The root URL or endpoint of the API.
    keywords (List[str]): A list of keywords to include in the queries.
    years (List[int]): A list of years to filter the publications by.

    Returns:
    List[str]: A list of complete API query strings.
    """
    queries = []
    k = " OR ".join(keywords)
    temp_root = root.replace("KEYWORD", k)
    for year in years:
        queries.append(f"{temp_root}&filter=publication_year:{year}")
    return queries


def concat_json_files(input_files, s3_bucket, s3_path) -> pd.DataFrame:
    df = pd.DataFrame()

    for file in input_files:
        openalex_data = nesta_s3.download_obj(s3_bucket, f"{s3_path}/{file}", "dict")

        logging.info(f"Number of works in {file}: {len(openalex_data)}")

        year_df = pd.DataFrame(openalex_data)

        df = pd.concat([df, year_df])
    return df


def extract_concept_data(row: pd.Series) -> pd.DataFrame:
    """
    Extracts concept data from a row of the input DataFrame and transforms it into a DataFrame.

    In the input DataFrame, there is a 'concepts' column consisting of
    a list of dictionaries, each representing a concept. It converts this list into a DataFrame and adds
    additional columns based on other metadata from the input row.

    Parameters:
    - row (pd.Series): A row from the input DataFrame, expected to contain the following fields:
        - id (str): The OpenAlex ID for the publication.
        - title (str): The title of the publication.
        - publication_year (int/str): The year of publication.
        - concepts (List[Dict]): A list of dictionaries, each with concept metadata.

    Returns:
    - pd.DataFrame: A DataFrame where each row is a concept from the input 'concepts' list, with the
      following columns:
        - concept_id (str): The ID for the concept.
        - wikidata (str): The Wikidata ID for the concept.
        - display_name (str): The display name for the concept.
        - level (int): The level of the concept.
        - score (float): The score of the concept.
        - openalex_id (str): The OpenAlex ID for the publication, taken from the 'id' field of the input row.
        - title (str): The title of the publication, taken from the 'title' field of the input row.
        - year (int/str): The year of publication, taken from the 'publication_year' field of the input row.
    """

    concepts_df = pd.DataFrame(row["concepts"])
    concepts_df["openalex_id"] = row["id"]
    concepts_df["title"] = row["title"]
    concepts_df["year"] = row["publication_year"]
    concepts_df.rename(columns={"id": "concept_id"}, inplace=True)
    return concepts_df


def create_concepts_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame containing metadata for concepts extracted from an input DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame with at least the following columns:
        - id (str): The OpenAlex ID for the publication.
        - title (str): The title of the publication.
        - publication_year (int/str): The year of publication.
        - concepts (List[Dict]): A list of dictionaries, each representing a concept
          with the following keys: id, wikidata, display_name, level, and score.

    Returns:
    - pd.DataFrame: A DataFrame where each row contains the metadata for a single concept
      associated with a publication. The columns are:
        - openalex_id (str): The OpenAlex ID for the publication.
        - title (str): The title of the publication.
        - year (int/str): The year of publication.
        - concept_id (str): The ID for the concept.
        - wikidata (str): The Wikidata ID for the concept.
        - display_name (str): The display name for the concept.
        - level (int): The level of the concept.
        - score (float): The score of the concept.
    """

    concept_data_frames = df.apply(extract_concept_data, axis=1)
    out_df = pd.concat(concept_data_frames.values.tolist(), ignore_index=True)

    # Select and reorder the columns
    out_df = out_df[
        [
            "openalex_id",
            "title",
            "year",
            "concept_id",
            "wikidata",
            "display_name",
            "level",
            "score",
        ]
    ]

    return out_df


def create_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenates title + abstract into a new column 'text'.

    Deinvert the abstract and stick together the title and abstract. This mimics preprocessing done to create
    [this dataset](https://huggingface.co/datasets/colonelwatch/abstracts-embeddings).

    Args:
        df (pd.DataFrame): DataFrame with at least the following columns:
            - "id": The OpenAlex ID for the publication.
            - "title": The title of the publication.
            - "abstract_inverted_index": The inverted abstract of the publication.

    Returns:
        pd.DataFrame: A dataframe with the following columns:
            - "id": The OpenAlex ID for the publication.
            - "title": The title of the publication.
            - "abstract": The abstract of the publication as plain text.
            - "text": The concatenation of title and abstract.
    """
    # Deinvert the abstract and stick together the title and abstract. This mimics preprocessing done to create
    # [this dataset](https://huggingface.co/datasets/colonelwatch/abstracts-embeddings).

    df.loc[:, "abstract"] = df["abstract_inverted_index"].apply(
        lambda x: deinvert_abstract(x)
    )

    df.loc[:, "text"] = df["title"] + ". " + df["abstract"]

    return df[["id", "title", "abstract", "text"]]


def save_keywords_to_s3(
    keywords: List[str], path: str, timestamp: str, file_prefix: str
) -> None:
    """
    Save the KEYWORDS list to a .txt file and upload it to S3.

    Args:
        keywords (List[str]): List of keywords to save.
        path (str): S3 path to upload to.
        timestamp (str): Timestamp to create a unique filename.
    """
    if isinstance(keywords, list):
        keywords_str = "\n".join(keywords)
    else:
        keywords_str = keywords

    filename = f"{file_prefix}_{timestamp}.txt"
    custom_path = f"{path}/{filename}"

    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=S3_BUCKET, Key=custom_path, Body=keywords_str.encode("utf-8")
    )
