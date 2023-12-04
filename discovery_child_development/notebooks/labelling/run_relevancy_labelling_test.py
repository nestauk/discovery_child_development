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
    # 0: relevant
    "A fun activity for babies aged 3-6 months to help development and language learning. Try blowing bubbles with your baby and see how they react. Talk to them about what they're seeing.",
    # 1: non-relevant (child is too old)
    "A fun activity for 6 year old children to help development and language learning. Try blowing bubbles with your child and see how they react. Talk to them about what they're seeing.",
    # 2: non-relevant (non human)
    "A fun activity for a piglet to help development and learning. Try blowing bubbles with your little one and see how they react. Talk to them.",
    # 3: unclear (age not specified)
    "A fun activity for a child to help development and learning. Try blowing bubbles and see how they react. Talk to them.",
    # 4: not relevant (child is too old)
    "Developing Middle School Students' AI Literacy. In this experience report, we describe an AI summer workshop designed to prepare middle school students to become informed citizens and critical consumers of AI technology and to develop their foundational knowledge and skills to support future endeavors as AI-empowered workers. The workshop featured the 30-hour Developing AI Literacy or DAILy curriculum that is grounded in literature on child development, ethics education, and career development. The participants in the workshop were students between the ages of 10 and 14; 87% were from underrepresented groups in STEM and Computing.",
    # 5: unclear (age not specified)
    "Diverse methods are emerging as effective strategies for gathering high-frequency data on child development, integrating various data sources ranging from telephone interviews to noninvasive biomarker readings. This comprehensive approach offers a more holistic understanding of children's health and developmental progress.",
    # 6:relevant
    "Multimodal approaches have been shown to be a promising way to collect data on child development at high frequency, combining different data inputs (from phone surveys to signals from noninvasive biomarkers) to understand children’s health and development outcomes more integrally from multiple perspectives. We carried out a mixed study based on a transversal descriptive analysis and a longitudinal prospective analysis in Malawi. In each village, children were sampled to participate in weekly sessions in which data signals were collected through wearable devices (electrocardiography [ECG] hand pads and electroencephalography [EEG] headbands). During the implementation, 82 EEG/ECG data entry points were collected across four villages. The sampled children for EEG/ECG were 0 to 5 years old.",
]


async def main() -> None:
    """Fetch prompts and run the classifier"""

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
    examples = create_examples_string(load_jsonl(PATH_TO_PROMPTS / "examples.jsonl"))

    model = "gpt-4-1106-preview"
    temperature = 0.6

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
                OUTPUT_FILENAME,
                "test_data",
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
