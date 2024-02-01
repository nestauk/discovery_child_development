"""
Recipe for evaluating taxonomy labels

```
python -m prodigy eval_labels test_data outputs/labels/evals_data/taxonomy_labels_eval.jsonl -F discovery_child_development/analysis/evaluate_labels/eval_taxonomy_recipe.py
```
or
```
python -m prodigy eval_labels taxonomy_data outputs/labels/evals_data/taxonomy_labels_eval.jsonl -F discovery_child_development/analysis/evaluate_labels/eval_taxonomy_recipe.py
```

To export the data and have it saved locally, run:
```
prodigy db-out taxonomy_data > outputs/labels/evals_data/taxonomy_labels_eval_annotated.jsonl
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
from prodigy.components.loaders import JSONL
import random
from typing import Iterator
from discovery_child_development.utils.utils import prepare_url, current_time


def make_tasks(
    stream: Iterator[dict],
) -> Iterator[dict]:
    """Determines what to do with each example in the stream.

    Args:
        stream (Iterator[dict]): Prodigy stream
        options (Iterator[str]): The options to be displayed to the user.

    Yields:
        Iterator[dict]: The tasks to be displayed to the user.
    """
    for eg in stream:
        # Deduplication: if the OpenAlex/patent ID has already been labelled, skip it
        task = copy.deepcopy(eg)
        task["label"] = task["prediction"]
        task["meta"] = {"url": prepare_url(eg["id"], eg["source"])}
        task["_datetime"] = current_time()
        yield task


@prodigy.recipe(
    "eval_labels",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .jsonl file", "positional", None, Path),
)
def eval_labels(dataset: str, source: str):
    """Recipe for evaluating taxonomy labels."""

    GLOBAL_CSS = (
        ".prodigy-content{font-size: 17px}"
        " .prodigy-option{width: 23%}"
        " .prodigy-options{justify-content: space-between}"
        " .prodigy-container{max-width: 1000px}"
    )

    stream = list(JSONL(source))
    # Randomise the order in which examples are presented:
    # otherwise you would see all the OpenAlex abstracts, then all the patents
    # random.shuffle(stream)

    # load in jsonl file and find unique prediction labels
    # options = pd.read_json(source, lines=True)["prediction"].unique().tolist()
    stream = make_tasks(stream)
    stream = list(stream)

    blocks = [
        {"view_id": "classification"},
        {
            "view_id": "text_input",
            "field_rows": 1,
            "field_label": "Leave any comments here, if needed:",
        },
    ]

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "config": {
            "instructions": "discovery_child_development/analysis/evaluate_labels/eval_taxonomy_instructions.html",
            "buttons": ["accept", "reject", "ignore"],
            "task_description": "Choose the best fitting category for this text",
            "choice_style": "single",
            "feed_overlap": False,
            "host": "0.0.0.0",
            "port": 8080,
            "instant_submit": True,
            "blocks": blocks,
            "global_css": GLOBAL_CSS,
        },
    }
