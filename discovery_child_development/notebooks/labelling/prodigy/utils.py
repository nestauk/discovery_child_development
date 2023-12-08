import json
from pathlib import Path

from discovery_child_development import PROJECT_DIR
from discovery_child_development.utils.openai_utils import (
    FunctionTemplate,
    MessageTemplate,
)

PATH_TO_PROMPTS = (
    PROJECT_DIR / "discovery_child_development/notebooks/labelling/prompts/taxonomy"
)
PATH_USER = PATH_TO_PROMPTS / "user.json"
PATH_SYSTEM = PATH_TO_PROMPTS / "system.json"
PATH_FUNCTION = PATH_TO_PROMPTS / "function.json"
PATH_CATEGORIES = PATH_TO_PROMPTS / "taxonomy_categories.json"


def flatten_dictionary(d):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The input dictionary.

    Returns:
    - A flattened dictionary.
    """
    items = []
    for theme, nested_dict in d.items():
        for category in nested_dict:
            items.append((category, nested_dict[category]))
    return dict(items)


def get_labels_from_gpt_response(response):
    """
    Extract labels from the GPT-3 response.

    Parameters:
    - response: The GPT-3 response.

    Returns:
    - A list of labels.
    """
    # Get the labels from the response
    output = response.choices[0].message.function_call.arguments

    return [label.strip() for label in json.loads(output)["label"].split(",")]


def load_categories(categories_path=PATH_CATEGORIES):
    with open(categories_path) as json_file:
        categories = json.load(json_file)

    categories_flat = flatten_dictionary(categories)

    return categories_flat


def format_categories_for_prompt(categories_flat):
    category_list = []
    for category in categories_flat.keys():
        string = f"'{category}': {categories_flat[category]}"
        category_list.append(string)

    category_list = "\n".join(category_list)

    return category_list


def format_function(path=PATH_FUNCTION):
    if isinstance(path, Path):
        path = str(path)

    categories_flat = load_categories()

    function = FunctionTemplate.load(path).to_prompt()

    function["parameters"]["properties"]["label"]["enum"] = list(categories_flat.keys())

    return function


def build_prompt(text, path_system=PATH_SYSTEM, path_user=PATH_USER):
    if isinstance(path_system, Path):
        path_system = str(path_system)

    if isinstance(path_user, Path):
        path_user = str(path_user)

    categories_flat = load_categories()

    system = MessageTemplate.load(path_system).to_prompt()
    user_message = MessageTemplate.load(path_user).to_prompt()
    user_message["content"] = user_message["content"].format(
        text=text, categories=format_categories_for_prompt(categories_flat)
    )

    return [system, user_message]
