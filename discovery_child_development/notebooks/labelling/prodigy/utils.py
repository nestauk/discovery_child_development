import json

from discovery_child_development import PROJECT_DIR

FUNCTION_PATH = (
    PROJECT_DIR
    / "discovery_child_development/notebooks/labelling/prompts/taxonomy/function.json"
)
CATEGORIES_PATH = (
    PROJECT_DIR
    / "discovery_child_development/notebooks/labelling/prompts/taxonomy/taxonomy_categories.json"
)
PROMPT_TEXT_PATH = (
    PROJECT_DIR
    / "discovery_child_development/notebooks/labelling/prompts/taxonomy/task_prompt.txt"
)


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


def load_categories(categories_path=CATEGORIES_PATH):
    with open(categories_path) as json_file:
        categories = json.load(json_file)

    categories_flat = flatten_dictionary(categories)

    return categories_flat


def format_function(path=FUNCTION_PATH):
    categories_flat = load_categories()

    with open(path) as json_file:
        function_json = json.load(json_file)

    function_json["parameters"]["properties"]["label"]["enum"] = list(
        categories_flat.keys()
    )

    return function_json


def format_categories_for_prompt(categories_flat):
    category_list = []
    for category in categories_flat.keys():
        string = f"{category}: {categories_flat[category]}"
        category_list.append(string)

    category_list = "\n".join(category_list)

    return category_list


def format_message(text, path=PROMPT_TEXT_PATH, categories_flat=load_categories()):
    category_list = format_categories_for_prompt(categories_flat)

    with open(path, "r") as file:
        task_prompt = file.read()

    task_prompt = task_prompt.replace("TEXT_TO_LABEL", text).replace(
        "CATEGORIES_AND_KEYWORDS", category_list
    )

    return task_prompt
