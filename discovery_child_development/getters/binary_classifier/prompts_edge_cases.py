import pandas as pd
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import S3_BUCKET


def get_examples():
    """Gets some examples of the data; including some edge cases and prompts used for labelling.

    Returns:
        pandas.DataFrame: Dataframe containing examples of the data.
    """
    return S3.download_obj(
        bucket=S3_BUCKET,
        path_from="data/openAlex/test_text/relevance_classifier_tests.csv",
        download_as="dataframe",
        kwargs_reading={"engine": "python"},
    )
