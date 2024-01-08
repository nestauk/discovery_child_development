import json
from pathlib import Path
import tiktoken
from typing import Any, Dict, List, Set, Union

from discovery_child_development import PROJECT_DIR
from discovery_child_development.utils.openai_utils import (
    FunctionTemplate,
    MessageTemplate,
)
from pandas import DataFrame

PATH_TO_PROMPTS = (
    PROJECT_DIR / "discovery_child_development/pipeline/labelling/taxonomy/prompts"
)
PATH_USER = PATH_TO_PROMPTS / "user.json"
PATH_SYSTEM = PATH_TO_PROMPTS / "system.json"
PATH_FUNCTION = PATH_TO_PROMPTS / "function.json"
PATH_CATEGORIES = PATH_TO_PROMPTS / "taxonomy_categories.json"


def flatten_dictionary(d: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    This gets used for the taxonomy categories: there are 6 category headings (eg "Health") and categories
    within those (eg "Oral health"). For the purposes of labelling the data, we don't care about the category headings,
    just the categories themselves.

    Parameters:
    - d (Dict[str, Dict[str, Any]]): The input dictionary.

    Returns:
    - Dict[str, Any]: A flattened dictionary.
    """
    items = []
    for _, nested_dict in d.items():
        for category in nested_dict:
            items.append((category, nested_dict[category]))
    return dict(items)


def get_labels_from_gpt_response(response) -> List[str]:
    """
    Extract labels from the GPT-3 response.

    Parameters:
    - response: The GPT-3 response.

    Returns:
    - List[str]: A list of labels.
    """
    # Get the labels from the response
    output = response.choices[0].message.function_call.arguments

    # Check that the response is a valid json
    try:
        decoded_response = json.loads(output)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # If it is not a valid json, provide a default label
        # TODO: handle this better
        decoded_response = {"label": "==NONE=="}

    # GPT gives back a string of comma-separated labels, so we split this into a list
    return [label.strip() for label in decoded_response["label"].split(",")]


def load_categories(
    categories_path: Union[str, Path] = PATH_CATEGORIES
) -> Dict[str, Any]:
    """
    Load and flatten category data from a JSON file.

    This function reads category data from a specified JSON file,
    then flattens the nested dictionary structure into a single-level
    dictionary using the flatten_dictionary function.

    Parameters:
    - categories_path (Union[str, Path]): The file path to the JSON file containing categories,
                                          which can be a string or a pathlib.Path object.
                                          Defaults to PATH_CATEGORIES.

    Returns:
    - Dict[str, Any]: A flattened dictionary of categories.
    """
    with open(categories_path) as json_file:
        categories = json.load(json_file)

    categories_flat = flatten_dictionary(categories)

    return categories_flat


def format_categories_for_prompt(categories_flat: Dict[str, str]) -> str:
    """
    This function takes the (flattened) dict of categories and converts them into a single string
    so that they can be pasted into the OpenAI prompt. The output format is:
    'category 1': 'keyword 1, keyword 2, keyword 3'
    'category 2': 'keyword 4, keyword 5, keyword 6'

    Parameters:
    - categories_flat (Dict[str, str]): A dictionary of categories where the keys are
                                        category names and the values are their descriptions.

    Returns:
    - str: A formatted string representing the categories, suitable for use in prompts.
    """
    category_list = []
    for category in categories_flat.keys():
        string = f"'{category}': {categories_flat[category]}"
        category_list.append(string)

    category_list = "\n".join(category_list)

    return category_list


def make_keyword_dict(categories_flat: Dict[str, str]) -> Dict[str, str]:
    """
    This function inverts the categories dict. This is because GPT often responds with a keyword
    rather than with the appropriate category name. We use this function when processing the
    GPT response so that if the GPT response includes a keyword, we can map that keyword to the category name.
    See also the function below, `map_keywords_to_categories()`.

    Parameters:
    - categories_flat (Dict[str, str]): A dictionary where keys are category names and values are
                                        comma-separated strings of keywords.

    Returns:
    - Dict[str, str]: A dictionary where each key is a keyword, and each value is a category name.
    """
    new_category_dict = {}

    for key, value in categories_flat.items():
        # `all_terms` includes the original category name, a lowercase version of the category name, and each of the keywords.
        # If any of these is in the GPT output, it will be mapped to the original category name.
        all_terms = [key] + [key.lower()] + [item.strip() for item in value.split(",")]
        for term in all_terms:
            new_category_dict[term] = key

    return new_category_dict


def map_keywords_to_categories(
    test_string: str,
    new_category_dict: Dict[str, str],
    category_names: Union[Set[str], List[str]],
) -> str:
    """
    Map a string to its corresponding category label.

    This function checks if the input string is a category name or is a keyword that needs to be converted to the appropriate
    category name:
    - If it is already category name, it returns the test string itself as the label.
    - If it is in the dict new_category_dict (created by the function `make_keyword_dict()`), it uses that dict
    to get the corresponding category name.
    - If it's neither, it returns 'no label'.

    Parameters:
    - test_string (str): The input string.
    - new_category_dict (Dict[str, str]): A dictionary mapping keywords to category names.
    - category_names (Union[Set[str], List[str]]): A collection of category names.

    Returns:
    - str: The corresponding category category name or 'no label' if no match is found.
    """
    if test_string in category_names:
        label = test_string
    elif test_string in new_category_dict.keys():
        label = new_category_dict[test_string]
    else:
        label = "no label"

    return label


def format_function(
    categories_flat: Dict[str, str], path: Union[str, Path] = PATH_FUNCTION
):
    """
    Format a function template to include the desired output values.

    The keys from the dict catgories_flat are pasted into the function as the available output options.
    The string "no label" is added to the list of available category names, in case no category fits.

    Parameters:
    - categories_flat (Dict[str, str]): A dictionary of category names and their descriptions.
    - path (Union[str, Path]): The file path to the function template, either as a string or a pathlib.Path object.
                               Defaults to PATH_FUNCTION.

    Returns:
    - A modified instance of FunctionTemplate.
    """
    if isinstance(path, Path):
        path = str(path)

    function = FunctionTemplate.load(path).to_prompt()

    function["parameters"]["properties"]["label"]["enum"] = list(
        categories_flat.keys()
    ) + ["no label"]

    return function


def build_prompt(text, categories_flat, path_system=PATH_SYSTEM, path_user=PATH_USER):
    if isinstance(path_system, Path):
        path_system = str(path_system)

    if isinstance(path_user, Path):
        path_user = str(path_user)

    system = MessageTemplate.load(path_system).to_prompt()
    user_message = MessageTemplate.load(path_user).to_prompt()
    user_message["content"] = user_message["content"].format(
        text=text, categories=format_categories_for_prompt(categories_flat)
    )

    return [system, user_message]


def num_tokens_from_string(string: str, encoding):
    """Gets the number of tokens in a string, using tiktoken encoding."""
    return len(encoding.encode(string))


def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_model_cost(model):
    """
    Get the cost of input and output tokens for a given OpenAI API model.
    The cost is expressed in $ per 1000 tokens.

    Based on https://openai.com/pricing

    """
    if model == "gpt-3.5-turbo-1106":
        input = 0.001
        output = 0.002
    elif model == "gpt-4":
        input = 0.03
        output = 0.06
    return input, output


def unique_list(series):
    """Return a list of unique values in a series."""
    return list(set(series))


def clean_labelled_data(labelled_data: DataFrame, categories: dict) -> DataFrame:
    """Clean the labelled data

    Fixes the cases where LLM has labelled the text with keywords rather than categories,
    and removes the datapoints that have no label.

    Args:
        labelled_data (pd.DataFrame): A DataFrame containing the labelled data
        categories (dict): A dictionary containing the categories

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned labelled data
    """
    category_names = list(categories.keys())
    keyword_to_category_dict = make_keyword_dict(categories)
    return (
        DataFrame(labelled_data)
        .explode("prediction")
        .rename(columns={"prediction": "prediction_raw"})
        .assign(
            prediction=lambda df: df["prediction_raw"].apply(
                lambda x: map_keywords_to_categories(
                    x, keyword_to_category_dict, category_names
                )
            )
        )
        .query("prediction != 'no label'")
        .groupby(["id", "text", "source"])
        .agg({"prediction_raw": unique_list, "prediction": unique_list})
        .reset_index()
    )
