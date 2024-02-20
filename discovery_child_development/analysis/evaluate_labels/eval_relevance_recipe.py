"""
Recipe for evaluating relevance labels

```
python -m prodigy eval_labels test_data outputs/labels/evals_data/relevance_labels_eval.jsonl -F discovery_child_development/analysis/evaluate_labels/eval_relevance_recipe.py
```
or
```
python -m prodigy eval_labels relevance_data outputs/labels/evals_data/relevance_labels_eval.jsonl -F discovery_child_development/analysis/evaluate_labels/eval_relevance_recipe.py
```

To export the data and have it saved locally, run:
```
prodigy db-out relevance_data > outputs/labels/evals_data/relevance_labels_eval_annotated.jsonl
```

If you have labelled your test examples and want to scrap those labels and start again (eg if you've switched to a different GPT model),
run:
```
prodigy drop relevance_data
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
    options: Iterator[str],
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
        task["options"] = [{"id": option, "text": option} for option in options]
        task["accept"] = [eg["prediction"]]
        task["meta"] = {"url": prepare_url(eg["id"], eg["source"])}
        task["_datetime"] = current_time()
        yield task


@prodigy.recipe(
    "eval_labels",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .jsonl file", "positional", None, Path),
)
def eval_labels(dataset: str, source: str):
    """Recipe for evaluating relevance labels."""

    GLOBAL_CSS = (
        ".prodigy-content{font-size: 17px}"
        " .prodigy-option{width: 23%}"
        " .prodigy-options{justify-content: space-between}"
        " .prodigy-container{max-width: 1000px}"
    )

    stream = list(JSONL(source))
    # Randomise the order in which examples are presented:
    # otherwise you would see all the OpenAlex abstracts, then all the patents
    random.shuffle(stream)

    # load in jsonl file and find unique prediction labels
    # options = pd.read_json(source, lines=True)["prediction"].unique().tolist()
    options = ["Relevant", "Not-relevant", "Not-specified", "???"]
    stream = make_tasks(stream, options=options)
    stream = list(stream)

    blocks = [
        {"view_id": "choice"},
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
            "instructions": "discovery_child_development/analysis/evaluate_labels/eval_relevance_instructions.html",
            "buttons": ["accept", "ignore"],
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
