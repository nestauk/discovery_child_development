"""
This script is used to test the relevancy labelling prompt and function.

Usage:
    python discovery_child_development/pipeline/labelling/relevance/relevance_labels.py --dataset openalex --num_samples 5 --model gpt-4-1106-preview
"""
import argparse
import json
import pandas as pd
import asyncio
import time
from pathlib import Path

from discovery_child_development import logging, PROJECT_DIR
from discovery_child_development.getters import openalex, patents
from discovery_child_development.utils.openai_utils import (
    MessageTemplate,
    Classifier,
    FunctionTemplate,
)
from discovery_child_development.utils.labelling_utils import (
    create_category_description_string,
    create_examples_string,
)
from discovery_child_development.utils.utils import (
    load_jsonl,
    create_directory_if_not_exists,
    batch,
    get_yaml_config,
)

# Get labelling config params
CONFIG = get_yaml_config(Path(__file__).resolve().parent / "config.yaml")
# Define paths to the prompt and function definitions
PATH_TO_PROMPTS = Path(__file__).resolve().parent / "prompts"
PATH_TO_CATEGORIES = PATH_TO_PROMPTS / "categories.json"
PATH_TO_MESSAGE_PROMPT = PATH_TO_PROMPTS / "prompt.json"
PATH_TO_FUNCTION = PATH_TO_PROMPTS / "function.json"
PATH_TO_EXAMPLES = PATH_TO_PROMPTS / "examples.jsonl"
# Define paths to the outputs
OUTPUT_FILEPATH = PROJECT_DIR / "outputs/labels/relevance"
OUTPUT_FILENAME = CONFIG["output_filename"]


async def main(
    dataset: str,
    num_samples: int,
    model: str,
    temperature: float,
) -> None:
    """Fetch prompts and run the classifier"""

    # Fetch the data to be categorised
    if dataset == "openalex":
        texts_df = openalex.get_abstracts()
    elif dataset == "patents":
        texts_df = patents.get_and_process_patents_from_s3()
    # Remove texts that are already labelled
    try:
        labelled = load_jsonl(str(OUTPUT_FILEPATH / OUTPUT_FILENAME) + ".jsonl")
        labelled_ids = pd.DataFrame(labelled).id.to_list()
        texts_df = texts_df[~texts_df.id.isin(labelled_ids)]
    except FileNotFoundError:
        logging.info("No labelled data found")  # noqa: T001
    # Subsample for testing
    texts_df = texts_df.sample(num_samples).reset_index(drop=True)
    texts_df["source"] = dataset
    texts_df["timestamp"] = time.time()

    # Fetch the categories
    categories = json.loads(PATH_TO_CATEGORIES.read_text())

    logging.info(f"Number of texts to classify: {len(texts_df)}")  # noqa: T001
    logging.info(f"Number of distinct categories: {len(categories)}")  # noqa: T001

    message = MessageTemplate.load(str(PATH_TO_MESSAGE_PROMPT))
    function = FunctionTemplate.load(str(PATH_TO_FUNCTION))
    examples = create_examples_string(load_jsonl(PATH_TO_EXAMPLES))

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
                    "source": tup.source,
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
            # Write to the file
            await Classifier.write_line_to_file(
                result,
                OUTPUT_FILEPATH,
                OUTPUT_FILENAME,
            )

        time.sleep(2)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the arguments for the script")

    # Add the arguments
    parser.add_argument("--dataset", type=str, help="The dataset")
    parser.add_argument("--num_samples", type=int, help="The number of samples")
    parser.add_argument("--model", type=str, help="The model")

    # Parse the arguments
    return parser.parse_args()


if "__main__" == __name__:
    # Define the arguments
    args = parse_arguments()
    try:
        dataset = args.dataset
    except ValueError as e:
        raise ValueError("Dataset must be one of 'patents' or 'openalex'")
    num_samples = args.num_samples if args.num_samples else CONFIG["num_samples"]
    model = args.model if args.model else CONFIG["model"]
    temperature = CONFIG["temperature"]

    # Create outputs directory if it doesn't exist
    create_directory_if_not_exists(OUTPUT_FILENAME)

    # Label the data
    start = time.time()
    s = time.perf_counter()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(dataset, num_samples, model, temperature))
    finally:
        loop.close()
    e = time.perf_counter()

    logging.info(
        f"Concurrent execution completed in: {e - s:0.2f} seconds"
    )  # noqa: T001
