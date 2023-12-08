"""
To test this, create and activate a new prodigy env (using `prodigy_requirements.txt`) and run:
```
prodigy oa_classification taxonomy_data discovery_child_development/notebooks/labelling/prodigy/test_sample.jsonl -F discovery_child_development/notebooks/labelling/prodigy/taxonomy_classifier_recipe.py
```
or
```
prodigy oa_classification taxonomy_data inputs/data/labelling/taxonomy/training_validation_data.jsonl -F discovery_child_development/notebooks/labelling/prodigy/taxonomy_classifier_recipe.py
```

To export the data and have it saved locally, run:
```
prodigy db-out taxonomy_data > discovery_child_development/notebooks/labelling/prodigy/taxonomy_data.jsonl
```

If you have labelled your test examples and want to scrap those labels and start again (eg if you've switched to a different GPT model),
run:
```
prodigy drop taxonomy_data
```

"""

import prodigy
from prodigy.components.loaders import JSONL
from pathlib import Path
from typing import Iterator
import copy
import tiktoken

# import os
import dotenv

# from openai import OpenAI
# import json

# from discovery_child_development import PROJECT_DIR
from discovery_child_development.utils import taxonomy_labelling_utils as tlu

# import discovery_child_development.utils.openai_utils
from discovery_child_development.utils.openai_utils import client

dotenv.load_dotenv()

MODEL = "gpt-3.5-turbo-1106"
MODEL_INPUT_COST = 0.001  # based on https://openai.com/pricing
MODEL_OUTPUT_COST = 0.002
TEMPERATURE = 0.0
encoding = tiktoken.encoding_for_model(MODEL)

categories_flat = tlu.load_categories()


def make_tasks(stream: Iterator[dict], model=MODEL) -> Iterator[dict]:
    for eg in stream:
        task = copy.deepcopy(eg)
        text = eg["text"]

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
            {"id": category, "text": category}
            for category in list(categories_flat.keys())
        ]

        output_as_list = tlu.get_labels_from_gpt_response(response)

        task["tokens_input"] = response.usage.prompt_tokens
        task["tokens_output"] = response.usage.completion_tokens

        task["cost"] = (MODEL_INPUT_COST * (response.usage.prompt_tokens / 1000)) + (
            MODEL_OUTPUT_COST * (response.usage.completion_tokens / 1000)
        )

        task["options"] = options

        task["accept"] = output_as_list

        task["model_output"] = output_as_list

        yield task


@prodigy.recipe(
    "oa_classification",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .jsonl file", "positional", None, Path),
)
def custom_oa(dataset: str, source: str):
    stream = JSONL(source)

    stream = make_tasks(stream, model=MODEL)

    return {
        "dataset": dataset,
        "view_id": "choice",
        "stream": stream,
        "config": {"choice_style": "multiple"},
    }
