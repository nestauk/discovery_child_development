"""Module for getting labelled data from S3"""
import pandas as pd
from nesta_ds_utils.loading_saving import S3
from discovery_child_development import S3_BUCKET


def get_relevance_labels(filename: str = "relevance_labels_20231212") -> pd.DataFrame:
    """Get relevance labels from S3

    Returns:
        pd.DataFrame: A DataFrame containing the relevance labels.
            Columns are:
                - id: The ID of the work
                - source: The source of the data (so far: openalex or patent)
                - text: The text of the work (title + abstract)
                - prediction: Relevant (is about preschool-age child development),
                    Not-relevant, Not-specified (might be about child development but age unclear)
    """
    return S3.download_obj(
        bucket=S3_BUCKET,
        path_from=f"data/labels/afs_relevance/{filename}.csv",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
    )
