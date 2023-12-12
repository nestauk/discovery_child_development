"""
This script is used to test the relevancy labelling prompt and function.

Usage:
    python discovery_child_development/notebooks/labelling/run_relevancy_labelling_test.py
"""
from discovery_child_development import logging
from discovery_child_development.utils.utils import (
    load_jsonl,
    create_directory_if_not_exists,
    batch,
)

import json
import pandas as pd
import asyncio
import time

from discovery_child_development import PROJECT_DIR
from discovery_child_development.utils.openai_utils import (
    MessageTemplate,
    Classifier,
    FunctionTemplate,
)
from discovery_child_development.utils.labelling_utils import (
    create_category_description_string,
    create_examples_string,
)

from discovery_child_development.getters import openalex

PATH_TO_PROMPTS = (
    PROJECT_DIR / "discovery_child_development/notebooks/labelling/prompts/relevance"
)
PATH_TO_CATEGORIES = PATH_TO_PROMPTS / "categories.json"
PATH_TO_MESSAGE_PROMPT = PATH_TO_PROMPTS / "prompt.json"
PATH_TO_FUNCTION = PATH_TO_PROMPTS / "function.json"
PATH_TO_EXAMPLES = PATH_TO_PROMPTS / "examples.jsonl"
OUTPUT_FILEPATH = (
    PROJECT_DIR / "discovery_child_development/notebooks/labelling/data/relevance"
)
OUTPUT_FILENAME = "relevance_openalex"


async def main() -> None:
    """Fetch prompts and run the classifier"""

    # Fetch the data to be categorised
    # Load patent data
    texts_df = openalex.get_abstracts()
    # Remove texts that are already labelled
    try:
        labelled = load_jsonl(str(OUTPUT_FILEPATH / OUTPUT_FILENAME) + ".jsonl")
        labelled_ids = pd.DataFrame(labelled).id.to_list()
        texts_df = texts_df[~texts_df.id.isin(labelled_ids)]
    except FileNotFoundError:
        logging.info("No labelled data found")  # noqa: T001
    # Subsample for testing
    texts_df = texts_df.sample(250).reset_index(drop=True)

    # Fetch the categories
    categories = json.loads(PATH_TO_CATEGORIES.read_text())

    logging.info(f"Number of texts to classify: {len(texts_df)}")  # noqa: T001
    logging.info(f"Number of distinct categories: {len(categories)}")  # noqa: T001

    message = MessageTemplate.load(str(PATH_TO_MESSAGE_PROMPT))
    function = FunctionTemplate.load(str(PATH_TO_FUNCTION))
    examples = create_examples_string(load_jsonl(PATH_TO_EXAMPLES))

    model = "gpt-4-1106-preview"
    temperature = 0.0

    for i, batched_results in enumerate(batch(texts_df, 20)):
        logging.info(f"Batch {i} / {len(texts_df) // 20}")  # noqa: T001
        tasks = [
            Classifier.agenerate(
                model=model,
                temperature=temperature,
                messages=[message],
                message_kwargs={
                    "category_description": create_category_description_string(
                        categories, randomise=True
                    ),
                    "n_categories": len(categories),
                    "examples": examples,
                    "text": tup.text,
                    "id": tup.id,
                },
                functions=[function.to_prompt()],
                function_call={"name": "predict_relevance"},
                max_tokens=100,
                concurrency=5,
            )
            for tup in batched_results.itertuples()
        ]

        for future in asyncio.as_completed(tasks):
            result = await future  # Get the result (waits if not ready)
            await Classifier.write_line_to_file(
                result,
                OUTPUT_FILEPATH,
                OUTPUT_FILENAME,
            )  # Write to the file

        time.sleep(2)


if "__main__" == __name__:
    # create data/ directory if it doesn't exist
    create_directory_if_not_exists(OUTPUT_FILENAME)

    start = time.time()
    s = time.perf_counter()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    e = time.perf_counter()

    logging.info(
        f"Concurrent execution completed in: {e - s:0.2f} seconds"
    )  # noqa: T001
