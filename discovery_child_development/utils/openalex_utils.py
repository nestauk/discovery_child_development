from typing import Dict, List, Any, Union
from itertools import chain


def uninvert_index(index) -> str:
    """Uninvert an index to plain text

    Args:
        index (dict): The abstract in inverted index format, as returned by the OpenAlex API

    Returns:
        str: A plain text version of the abstract
    """
    # Check if index is None first
    if isinstance(index, str) or index is None:
        uninverted_index = ""
    else:
        inverted_index = index
        uninverted_index = {}

        # Initialize an empty list with None values
        text_list = [None] * (
            max([idx for indices in inverted_index.values() for idx in indices]) + 1
        )

        # Populate the list with words from the inverted index
        for word, indices in inverted_index.items():
            for idx in indices:
                text_list[idx] = word

        # Convert the list to plain text
        uninverted_index = " ".join(text_list)

    return uninverted_index


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
