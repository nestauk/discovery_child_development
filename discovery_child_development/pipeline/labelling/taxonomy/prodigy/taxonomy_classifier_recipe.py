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
aws s3 cp inputs/data/labelling/taxonomy/output/training_validation_data_patents_openalex_LABELLED.jsonl s3://discovery-iss/data/labels/taxonomy_classifier/training_validation_data_patents_openalex_LABELLED.jsonl

```

If you have labelled your test examples and want to scrap those labels and start again (eg if you've switched to a different GPT model),
run:
```
prodigy drop taxonomy_data
```

"""

import prodigy
from prodigy.components.db import connect
from prodigy.components.loaders import JSONL
from pathlib import Path
from typing import Iterator
import copy
import tiktoken
import random

from discovery_child_development.utils import taxonomy_labelling_utils as tlu

from discovery_child_development.utils.openai_utils import client

MODEL = "gpt-3.5-turbo-1106"


def get_model_cost(model):
    # based on https://openai.com/pricing
    if model == "gpt-3.5-turbo-1106":
        input = 0.001
        output = 0.002
    elif model == "gpt-4":
        input = 0.03
        output = 0.06
    return input, output


MODEL_INPUT_COST, MODEL_OUTPUT_COST = get_model_cost(MODEL)
TEMPERATURE = 0.0
encoding = tiktoken.encoding_for_model(MODEL)

categories_flat = tlu.load_categories()

categories_list = sorted(list(categories_flat.keys()))


def get_existing_ids(dataset_name, id_field="id"):
    db = connect()
    all_dataset_names = db.datasets
    if dataset_name not in all_dataset_names:
        return set()  # Return an empty set if the dataset does not exist
    else:
        return {
            example[id_field]
            for example in db.get_dataset(dataset_name)
            if id_field in example
        }


def make_tasks(
    stream: Iterator[dict], existing_ids: set, model=MODEL
) -> Iterator[dict]:
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

            function = tlu.format_function(categories_flat)

            # Call OpenAI API
            response = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                messages=prompt,
                functions=[function],
                function_call={"name": "predict_category"},
            )

            # Process and format the output for Prodigy
            options = [
                {"id": category, "text": category} for category in categories_list
            ]
            output_as_list = tlu.get_labels_from_gpt_response(response)
            task["options"] = options
            task["accept"] = output_as_list

            # These task components will not be displayed to the user, but they are available in the saved data for downstream analysis
            task[
                "model_output"
            ] = output_as_list  # The choices made by GPT (before human labelling)
            task["model"] = model  # GPT model used
            task["source"] = source  # OpenAlex or patents
            task["tokens_input"] = response.usage.prompt_tokens
            task["tokens_output"] = response.usage.completion_tokens
            task["cost"] = (
                MODEL_INPUT_COST * (response.usage.prompt_tokens / 1000)
            ) + (MODEL_OUTPUT_COST * (response.usage.completion_tokens / 1000))

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
    random.shuffle(stream)

    stream = make_tasks(stream, existing_ids, model=MODEL)

    return {
        "dataset": dataset,
        "view_id": "choice",
        "stream": stream,
        "config": {"choice_style": "multiple"},
    }
