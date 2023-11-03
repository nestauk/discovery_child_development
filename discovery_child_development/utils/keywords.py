"""
Utility functions for dealing with keywords
"""
from typing import List
import re
import nltk
import pandas as pd
import numpy as np


def load_tokenizer():
    """Load nltk tokenizer"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer


def process_keywords(keywords: List[str]) -> List[List[str]]:
    """Process a list of keywords and keyword combinations

    Args:
        keywords (List[str]): list of keywords in the format ['keyword_1', 'keyword_2 + keyword_3']

    Returns:
        List[List[str]]: list of lists of keywords in the format ['keyword_1', ['keyword_2', 'keyword_3']]
    """
    return [[word.strip() for word in line.split("+")] for line in keywords]


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
    return process_keywords(keywords)


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


# def check_articles_for_comma_terms(text: str, terms: str):
#     """Return True if text contains comma terms"""
#     terms = [term.strip() for term in terms.split(",")]
#     sentences_with_terms = find_sentences_with_terms(text, terms, all_terms=True)
#     return len(sentences_with_terms) >= 1


def find_sentences_with_terms(
    text: str, terms: List[str], all_terms: bool = True
) -> List[str]:
    """Find sentences which contain specified search terms

    Args:
        text (str): text to search
        terms (List[str]): list of terms to search for
        all_terms (bool, optional): whether to search for all terms or any term. Defaults to True.

    Returns:
        List[str]: list of sentences containing the specified terms
    """
    tokenizer = load_tokenizer()
    # Split text into sentences
    sentences = tokenizer.tokenize(text)
    # Keep sentences with terms
    sentences_with_terms = []
    # Number of terms in the query
    n_terms = len(terms)
    for sentence in sentences:
        terms_detected = 0
        # cCeck all terms
        for term in terms:
            if term in sentence.lower():
                terms_detected += 1
        # Check if all terms were found
        if (all_terms and (terms_detected == n_terms)) or (
            (not all_terms) and (terms_detected > 0)
        ):
            sentences_with_terms.append(sentence)
    return sentences_with_terms


def check_keyword_hits(texts: pd.Series, keywords: list) -> List[bool]:
    """Check if a text contains any of the keywords or keyword combinatons

    Args:
        texts (pd.Series): Series of texts to search
        keywords (list): list of keywords or keyword combinations
    """
    hits = [
        texts.apply(lambda x: find_sentences_with_terms(x.lower(), keyword))
        .apply(len)
        .astype(bool)
        for keyword in keywords
    ]
    return list(np.array(hits).sum(axis=0).astype(bool))
