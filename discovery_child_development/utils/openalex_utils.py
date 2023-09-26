from typing import Dict, List, Union
from itertools import chain


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
