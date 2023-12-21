"""
To test this, create and activate a new prodigy env (using `prodigy_requirements.txt`) and run:
```
prodigy oa_classification test_data discovery_child_development/notebooks/labelling/prodigy/test_sample.jsonl -F discovery_child_development/notebooks/labelling/prodigy/taxonomy_classifier_recipe.py
```
or
```
prodigy oa_classification taxonomy_data inputs/data/labelling/taxonomy/input/training_validation_data_patents_openalex.jsonl -F discovery_child_development/pipeline/labelling/taxonomy/prodigy/taxonomy_classifier_recipe.py
```

To export the data and have it saved locally, run:
```
prodigy db-out taxonomy_data > inputs/data/labelling/taxonomy/output/training_validation_data_patents_openalex_LABELLED.jsonl
aws s3 cp inputs/data/labelling/taxonomy/output/training_validation_data_patents_openalex_LABELLED.jsonl s3://discovery-iss/data/labels/taxonomy_classifier/labelled_with_prodigy/training_validation_data_patents_openalex_LABELLED.jsonl

```

If you have labelled your test examples and want to scrap those labels and start again (eg if you've switched to a different GPT model),
run:
```
prodigy drop taxonomy_data
```

"""
import copy
from pathlib import Path
import prodigy
from prodigy.components.db import connect
from prodigy.components.loaders import JSONL
import random
import tiktoken
from typing import Any, Iterator, Set

from discovery_child_development.utils import taxonomy_labelling_utils as tlu
from discovery_child_development.utils.openai_utils import client

MODEL = "gpt-3.5-turbo-1106"

MODEL_INPUT_COST, MODEL_OUTPUT_COST = tlu.get_model_cost(MODEL)
TEMPERATURE = 0.0
encoding = tiktoken.encoding_for_model(MODEL)

categories_flat = tlu.load_categories()

categories_list = sorted(list(categories_flat.keys()))

new_category_dict = tlu.make_keyword_dict(categories_flat)


def get_existing_ids(
    dataset_name: str = "taxonomy_data", id_field: str = "id"
) -> Set[Any]:
    """This function is used for deduplication. It finds which IDs have already been seen by Prodigy.

    Args:
        dataset_name (str): The name of the prodigy dataset
        id_field (str, optional): The name of the unique ID field in the dataset. Defaults to "id".

    Returns:
        Set[Any]: A set of unique IDs from the specified dataset.
    """
    db = connect()
    all_dataset_names = (
        db.datasets
    )  # get all the datasets that your local prodigy instance has available
    if dataset_name not in all_dataset_names:
        return set()  # Return an empty set if the dataset does not yet exist
    else:
        return {
            example[id_field]
            for example in db.get_dataset(dataset_name)
            if id_field in example
        }


def make_tasks(
    stream: Iterator[dict],
    existing_ids: set,
    model=MODEL,
    temperature=TEMPERATURE,
    model_input_cost=MODEL_INPUT_COST,
    model_output_cost=MODEL_OUTPUT_COST,
) -> Iterator[dict]:
    """Determines what to do with each example in the stream.

    Args:
        stream (Iterator[dict]): Prodigy stream
        existing_ids (set): IDs obtained via the function `get_existing_ids()`
        model (str, optional): Which GPT model to use. Defaults to MODEL.
        temperature (float, optional): The temperature to use for GPT. Defaults to TEMPERATURE.
        model_input_cost (float, optional): The cost per 1000 tokens of the GPT input. Defaults to MODEL_INPUT_COST.
        model_output_cost (float, optional): The cost per 1000 tokens of the GPT output. Defaults to MODEL_OUTPUT_COST.

    Yields:
        Iterator[dict]: The tasks to be displayed to the user.
    """
    for eg in stream:
        # Deduplication: if the OpenAlex/patent ID has already been labelled, skip it
        if eg.get("id") in existing_ids:
            continue
        else:
            task = copy.deepcopy(eg)
            text = eg["text"]  # the raw text
            source = eg["source"]  # openalex or patents

            # Format the prompt with the text to be classified
            prompt = tlu.build_prompt(text, categories_flat)

            # Format the function
            function = tlu.format_function(categories_flat)

            # Call OpenAI API
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=prompt,
                functions=[function],
                function_call={"name": "predict_category"},
            )

            # Process and format the output for Prodigy.
            # The options displayed in the Prodigy "choice" view have to have both an 'id' and a 'text' field.
            options = [
                {"id": category, "text": category} for category in categories_list
            ]
            output_as_list = [
                tlu.map_keywords_to_categories(x, new_category_dict, categories_list)
                for x in tlu.get_labels_from_gpt_response(response)
            ]
            task["options"] = options
            # Set the response from GPT to be already accepted when the task gets displayed (the user can then manually correct these choices)
            task["accept"] = output_as_list

            # These task components will not be displayed to the user, but they are available in the saved data for downstream analysis
            task["model_output"] = tlu.get_labels_from_gpt_response(
                response
            )  # The choices made by GPT (before human labelling)
            task["model"] = model  # GPT model used
            task["source"] = source  # OpenAlex or patents
            task["tokens_input"] = response.usage.prompt_tokens
            task["tokens_output"] = response.usage.completion_tokens
            task["cost"] = (
                model_input_cost * (response.usage.prompt_tokens / 1000)
            ) + (model_output_cost * (response.usage.completion_tokens / 1000))

            yield task


@prodigy.recipe(
    "oa_classification",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .jsonl file", "positional", None, Path),
)
def custom_oa(dataset: str, source: str):
    # Get existing IDs from the dataset
    existing_ids = get_existing_ids(dataset)

    stream = list(JSONL(source))
    # Randomise the order in which examples are presented - otherwise you would see all the OpenAlex abstracts,
    # then all the patents
    random.shuffle(stream)

    stream = make_tasks(stream, existing_ids, model=MODEL)

    return {
        "dataset": dataset,
        "view_id": "choice",
        "stream": stream,
        "config": {
            "choice_style": "multiple"
        },  # this allows the user to select multiple categories
    }
