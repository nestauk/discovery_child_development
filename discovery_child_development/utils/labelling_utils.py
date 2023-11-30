import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, Generator
import os
import random


def add_binarise_labels(
    df: pd.DataFrame, label_column: str, not_valid_label: str = None
) -> pd.DataFrame:
    """Add label dummy columns to dataframe.

    Args:
        df: Dataframe to add dummy columns to.
        label_column: Column with labels to turn into dummy column.
            The label column must have values in a list.
        not_valid_label: Label that indicates that the record is
            not relevant or valid. If a record has this label,
            all of its other labels will be set to 0. The dummy
            column relating to this label will be removed.

    Returns:
        Dataframe with additional dummy label columns
    """
    mlb = MultiLabelBinarizer()
    dummy_cols = pd.DataFrame(
        mlb.fit_transform(df[label_column]), columns=mlb.classes_, index=df.index
    )

    if not_valid_label is not None:
        valid_cols = [col for col in dummy_cols.columns if col != not_valid_label]
        # Set all other labels to 0 if row has not valid label
        dummy_cols = dummy_cols[valid_cols].mask(dummy_cols[not_valid_label] == 1, 0)
    else:
        valid_cols = dummy_cols.columns

    return dummy_cols[valid_cols], mlb


def create_category_description_string(
    categories: dict, randomise: bool = False
) -> str:
    """Create the category descriptions for the prompt

    Args:
        categories (Dict): The categories, in the format {category: description}
        randomise (bool, optional): Whether to randomise the order of the categories. Defaults to False.

    Returns:
        str: The category descriptions with each category and description in a new line
    """
    category_descriptions = ""
    all_categories = list(categories.keys())
    if randomise:
        all_categories = random.sample(all_categories, len(all_categories))
    # randomise the order categories so that the order is not always the same
    for category in all_categories:
        category_descriptions += f"{category}: {categories[category]}\n"
    return category_descriptions


def batch(lst: list, n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def create_directory_if_not_exists(dir_path: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
