"""
Utility functions for dealing with keywords
"""
from typing import List
import re


def get_keywords(file: str) -> List[List[str]]:
    """Get keywords from a txt file. The keywords stored in a text file as
     'keyword_1\nkeyword_2 + keyword_3\n' will be returned as a list of lists
     ['keyword_1', ['keyword_2', 'keyword_3']] which will then be interpreted as
     'keyword_1' OR ('keyword_2' AND 'keyword_3')

    Args:
        file (str): path to a txt file with keywords

    Returns:
        List[str]: list of lists of keywords
    """
    with open(file) as f:
        keywords = f.readlines()
    keywords = [[word.strip() for word in line.split("+")] for line in keywords]
    return keywords


def save_keywords(keywords: List[List[str]], filepath: str) -> None:
    """Save keywords to a txt file

    Args:
        keywords (List[List[str]]): list of lists of keywords
        filepath (str): path to a txt file
    """
    with open(filepath, "w") as f:
        for sublist in keywords:
            f.write(" + ".join(sublist) + "\n")


def replace_word(lst: List[List[str]], old_word: str, new_word: str) -> List[List[str]]:
    """Replace a word in a list of lists of strings"""
    pattern = re.compile(rf"\b{old_word}\b")
    new_lst = [[pattern.sub(new_word, item) for item in sublist] for sublist in lst]
    return new_lst


def deduplicate_keywords(lst: List[List[str]]) -> List[List[str]]:
    """Deduplicate a list of lists of strings"""
    deduplicated_set = set(tuple(sublist) for sublist in lst)
    deduplicated_lst = [list(item) for item in deduplicated_set]
    return deduplicated_lst
