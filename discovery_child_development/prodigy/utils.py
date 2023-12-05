def flatten_dictionary(d):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The input dictionary.
    - parent_key: The prefix to be added to keys.
    - sep: The separator to use between keys.

    Returns:
    - A flattened dictionary.
    """
    items = []
    for theme, nested_dict in d.items():
        for category in nested_dict:
            items.append((category, nested_dict[category]))
    return dict(items)
