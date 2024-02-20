from .openalex import get_abstracts
from .openalex_broad_concepts import get_abstracts_broad
from .patents import get_and_process_patents_from_s3
from .labels import get_relevance_labels
from .taxonomy_classifier import get_labelling_sample
from pandas import DataFrame
from discovery_child_development import S3_BUCKET
from nesta_ds_utils.loading_saving.S3 import download_obj


def get_dataset(dataset: str) -> DataFrame:
    """Get the specified dataset

    Args:
        dataset (str): The dataset to be fetched, can be one of "openalex", "patents", "openalex_broad"

    Returns:
        DataFrame: A DataFrame containing the data from the specified dataset

    """
    if dataset == "openalex":
        return get_abstracts()
    elif dataset == "patents":
        return get_and_process_patents_from_s3()
    elif dataset == "openalex_broad":
        return get_abstracts_broad()
    elif dataset == "taxonomy_labelling_sample":
        return DataFrame(get_labelling_sample())
    elif dataset == "test_relevant_data":
        return get_relevance_labels().query("prediction == 'Relevant'")


def get_sentence_embeddings(
    filepath: str,
    filename: str,
    id: str = "id",
    s3_bucket: str = S3_BUCKET,
) -> DataFrame:
    """Get the sentence embeddings from S3

    Args:
        s3_bucket (str, optional): The S3 bucket where the embeddings are stored. Defaults to S3_BUCKET.
        filepath (str): The filepath of the embeddings
        filename (str): The filename of the embeddings
        id (str): The column name of the id column

    Returns:
        DataFrame: A DataFrame containing the sentence embeddings
    """
    embeddings = download_obj(
        s3_bucket,
        path_from=f"{filepath}{filename}",
        download_as="dataframe",
    )
    embeddings = embeddings.set_index(id)
    return embeddings
