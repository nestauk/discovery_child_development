"""
This script is used to test the relevancy labelling prompt and function.

Usage:
    python discovery_child_development/notebooks/labelling/run_relevancy_labelling_test.py
"""
from discovery_child_development.utils.openai_utils import openai
from discovery_child_development import logging
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
    batch,
    create_category_description_string,
    create_directory_if_not_exists,
)

from aiohttp import ClientSession

PATH_TO_PROMPTS = (
    PROJECT_DIR / "discovery_child_development/notebooks/labelling/prompts/relevance"
)
PATH_TO_CATEGORIES = PATH_TO_PROMPTS / "categories.json"
PATH_TO_MESSAGE_PROMPT = PATH_TO_PROMPTS / "prompt.json"
PATH_TO_FUNCTION = PATH_TO_PROMPTS / "function.json"
OUTPUT_FILENAME = (
    PROJECT_DIR / "discovery_child_development/notebooks/labelling/data/relevance"
)

TEST_TEXTS = [
    # relevant
    "A fun activity for babies aged 3-6 months to help development and language learning. Try blowing bubbles with your baby and see how they react. Talk to them about what they're seeing.",
    # non-relevant (child is too old)
    "A fun activity for 6 year old children to help development and language learning. Try blowing bubbles with your child and see how they react. Talk to them about what they're seeing.",
    # non-relevant (non human)
    "A fun activity for a piglet to help development and learning. Try blowing bubbles with your little one and see how they react. Talk to them.",
    # unclear (age not specified)
    "A fun activity for a child to help development and learning. Try blowing bubbles and see how they react. Talk to them.",
]


async def main() -> None:
    """Create prompts for path selection and infer paths."""
    openai.aiosession.set(ClientSession())

    # Fetch the data to be categorised (replace with real data)
    texts_df = pd.DataFrame(
        data={"text": TEST_TEXTS, "id": list(range(len(TEST_TEXTS)))}
    )

    # Fetch the categories
    categories = json.loads(PATH_TO_CATEGORIES.read_text())

    logging.info(f"Number of texts to classify: {len(texts_df)}")  # noqa: T001
    logging.info(f"Number of distinct categories: {len(categories)}")  # noqa: T001

    message = MessageTemplate.load(str(PATH_TO_MESSAGE_PROMPT))
    function = FunctionTemplate.load(str(PATH_TO_FUNCTION))
    model = "gpt-4"
    temperature = 0.6

    for i, batched_results in enumerate(batch(texts_df, 20)):
        print(f"Batch {i} / {len(texts_df) // 20}")  # noqa: T001
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
                result, OUTPUT_FILENAME
            )  # Write to the file

        time.sleep(2)

    await openai.aiosession.get().close()  # Close the http session at the end of the program


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

    print(f"Concurrent execution completed in: {e - s:0.2f} seconds")  # noqa: T001
