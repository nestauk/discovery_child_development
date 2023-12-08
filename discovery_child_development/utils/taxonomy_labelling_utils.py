import json
from pathlib import Path
import tiktoken

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


def format_function(categories_flat, path=PATH_FUNCTION):
    if isinstance(path, Path):
        path = str(path)

    function = FunctionTemplate.load(path).to_prompt()

    function["parameters"]["properties"]["label"]["enum"] = list(categories_flat.keys())

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
