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
