from .openalex import get_abstracts
from .openalex_broad_concepts import get_abstracts_broad
from .patents import get_and_process_patents_from_s3
from .taxonomy import get_labelling_sample
from .labels import get_relevance_labels
from pandas import DataFrame


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
