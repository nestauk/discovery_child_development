"""
To test this, cd into discovery_child_development/prodigy/ and run:
```
prodigy oa_classification taxonomy_data test_sample.jsonl -F taxonomy_classifier_recipe.py
```

To export the data and have it saved locally, run:
```
prodigy db-out taxonomy_data > taxonomy_data.jsonl
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

import os
import dotenv
from openai import OpenAI
import json

from utils import flatten_dictionary

dotenv.load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-3.5-turbo-1106"
TAXONOMY_PATH = "../prompts/taxonomy/taxonomy_categories.json"

with open(TAXONOMY_PATH) as json_file:
    categories = json.load(json_file)

categories_flat = flatten_dictionary(categories)
category_list = [
    f"{category}: {categories_flat[category]}" for category in categories_flat.keys()
]
categories_prompt = "\n".join(category_list)


def make_tasks(stream: Iterator[dict], model=MODEL) -> Iterator[dict]:
    for eg in stream:
        task = copy.deepcopy(eg)
        text = eg["text"]

        # Format the prompt with the text to be classified
        prompt = [
            {
                "role": "system",
                "content": "You are an expert Text Classification system. Your task is to accept Text as input and provide a category for the text based on the predefined labels.",
            },
            {
                "role": "user",
                "content": f"###Instructions###\nHere are the labels texts can be labelled with, and some indicative keywords associated with each category:\n -------------------------------------------------------- \n{categories_prompt}\n ----------------------------------- \n The task is non-exclusive, so you can provide more than one label. If the text cannot be classified into any of the provided labels, return `==NONE==`. \n Label the following text with one or more labels:\n ``` {text} ```\n",
            },
        ]

        function = {
            "name": "predict_category",
            "description": "Assign category labels to the given text",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": list(categories_flat.keys()),
                        "description": "Labels to assign to the given text",
                    }
                },
                "required": ["label"],
            },
        }

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=prompt,
            functions=[function],
            function_call={"name": "predict_category"},
        )

        output = response.choices[0].message.function_call.arguments

        # Process and format the output for Prodigy
        options = [
            {"id": category, "text": category}
            for category in list(categories_flat.keys())
        ]

        task["options"] = options

        task["accept"] = [
            category.strip() for category in json.loads(output)["label"].split(",")
        ]

        task["model_output"] = [
            category.strip() for category in json.loads(output)["label"].split(",")
        ]

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
