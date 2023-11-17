"""
Preprocesses the output from metaflow: splits it into a concepts metadata file, and an OpenAlex abstracts file.

Additional cleaning steps:
- NA valyes in 'abstract_inverted_index' and 'title' are removed.
- Works that are not in English are removed.

Usage:
python discovery_child_development/pipeline/01_preprocess_openalex.py
"""
import pandas as pd
from discovery_child_development.utils import openalex_utils


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
        lambda x: openalex_utils.deinvert_abstract(x)
    )

    df.loc[:, "text"] = df["title"] + ". " + df["abstract"]

    return df[["id", "title", "abstract", "text"]]
