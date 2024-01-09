"""
This script is used to test the relevancy labelling prompt and function.

Example usage:
    python discovery_child_development/pipeline/labelling/taxonomy_v2/taxonomy_labels.py --dataset openalex --num_samples 5 --model gpt-3.5-turbo-1106 --output_filename testing
"""
import argparse
import json
import pandas as pd
import asyncio
import time
from pathlib import Path

from nesta_ds_utils.loading_saving import S3
from discovery_child_development import logging, PROJECT_DIR, S3_BUCKET, get_yaml_config
from discovery_child_development.getters import get_dataset
from discovery_child_development.utils.openai_utils import (
    MessageTemplate,
    Classifier,
    FunctionTemplate,
)
from discovery_child_development.utils.labelling_utils import (
    create_category_description_string,
)
from discovery_child_development.utils.utils import (
    create_directory_if_not_exists,
    batch,
)
from discovery_child_development.utils import taxonomy_labelling_utils as tlu
from discovery_child_development.utils.jsonl_utils import load_jsonl

# Get labelling config params
CONFIG = get_yaml_config(Path(__file__).resolve().parent / "config.yaml")
# Define paths to the prompt and function definitions
PATH_TO_PROMPTS = Path(__file__).resolve().parent / "prompts"
PATH_TO_CATEGORIES = PATH_TO_PROMPTS / "categories.json"
PATH_TO_MESSAGE_PROMPT = PATH_TO_PROMPTS / "prompt.json"
PATH_TO_FUNCTION = PATH_TO_PROMPTS / "function.json"
# PATH_TO_EXAMPLES = PATH_TO_PROMPTS / "examples.jsonl"
# Define paths to the outputs
OUTPUT_FILEPATH = PROJECT_DIR / CONFIG["local_output_directory"]


async def main(
    dataset: str,
    num_samples: int,
    model: str,
    temperature: float,
    output_filename: str,
    s3_path: str,
) -> None:
    """Fetch data from S3, label it and upload the results to S3.

    Args:
        dataset (str): The dataset to be labelled
        num_samples (int): The number of samples to be labelled
        model (str): The model to be used
        temperature (float): The temperature to be used
        output_filename (str): The output filename
        s3_path (str): The S3 path to upload the results to
    """

    # Fetch the data to be categorised
    texts_df = get_dataset(dataset)
    # Remove datapoints that are already labelled
    try:
        # Update the outputs file with the S3 version
        S3.download_file(
            path_from=s3_path,
            bucket=S3_BUCKET,
            path_to=str(OUTPUT_FILEPATH / output_filename) + ".jsonl",
        )
        labelled = load_jsonl(str(OUTPUT_FILEPATH / output_filename) + ".jsonl")
        labelled_ids = pd.DataFrame(labelled).id.to_list()
        texts_df = texts_df[~texts_df.id.isin(labelled_ids)]
    except:
        logging.info("No labelled data found")  # noqa: T001
    # Subsample for testing
    if (num_samples == 0) or (num_samples > len(texts_df)):
        num_samples = len(texts_df)
    texts_df = texts_df.sample(num_samples).reset_index(drop=True)

    # Fetch the categories
    categories = tlu.load_categories(PATH_TO_CATEGORIES)

    logging.info(f"Number of texts to classify: {len(texts_df)}")  # noqa: T001
    logging.info(f"Number of distinct categories: {len(categories)}")  # noqa: T001

    # Load instruction messages
    messages = json.load(open(PATH_TO_MESSAGE_PROMPT, "r"))
    messages = [MessageTemplate.load(m) for m in messages]
    # Load function
    function = json.loads(open(PATH_TO_FUNCTION).read())
    function["parameters"]["properties"]["prediction"]["items"]["enum"] = list(
        categories.keys()
    )
    function = FunctionTemplate.load(function)

    for i, batched_results in enumerate(batch(texts_df, 20)):
        logging.info(f"Batch {i} / {len(texts_df) // 20}")  # noqa: T001
        tasks = [
            Classifier.agenerate(
                model=model,
                temperature=temperature,
                messages=messages,
                message_kwargs={
                    "category_description": create_category_description_string(
                        categories, randomise=True
                    ),
                    "n_categories": len(categories),
                    "text": tup.text,
                    "id": tup.id,
                    "source": tup.source,
                },
                functions=[function.to_prompt()],
                function_call={"name": "predict_labels"},
                max_tokens=500,
                concurrency=3,
            )
            for tup in batched_results.itertuples()
        ]
        time.sleep(10)

        for future in asyncio.as_completed(tasks):
            result = await future  # Get the result (waits if not ready)
            # Write to the file
            await Classifier.write_line_to_file(
                result,
                OUTPUT_FILEPATH,
                output_filename,
            )

        time.sleep(2)

    # Upload to S3
    S3.upload_file(
        path_from=str(OUTPUT_FILEPATH / output_filename) + ".jsonl",
        bucket=S3_BUCKET,
        path_to=s3_path,
    )


def parse_arguments():
    """Parse the arguments passed to the script."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the arguments for the script")
    # Add the arguments
    parser.add_argument("--dataset", type=str, help="The dataset")
    parser.add_argument("--model", type=str, help="The model")
    parser.add_argument("--num_samples", type=int, help="The number of samples")
    parser.add_argument("--output_filename", type=str, help="The output filename")
    # Parse the arguments
    return parser.parse_args()


if "__main__" == __name__:
    # Define the arguments
    args = parse_arguments()
    try:
        dataset = args.dataset
    except ValueError as e:
        raise ValueError(
            "Dataset must be one of 'taxonomy_labelling_data', 'patents', 'openalex'"
        )
    num_samples = args.num_samples if args.num_samples else CONFIG["num_samples"]
    model = args.model if args.model else CONFIG["model"]
    temperature = CONFIG["temperature"]
    output_filename = (
        args.output_filename if args.output_filename else CONFIG["output_filename"]
    )

    # Deal with local and s3 paths
    create_directory_if_not_exists(OUTPUT_FILEPATH)
    s3_path = CONFIG["s3_directory"] + output_filename + "_raw.jsonl"

    # Label the data
    start = time.time()
    s = time.perf_counter()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            main(
                dataset,
                num_samples,
                model,
                temperature,
                output_filename + "_raw",
                s3_path,
            )
        )
    finally:
        loop.close()
    e = time.perf_counter()

    logging.info(
        f"Concurrent execution completed in: {e - s:0.2f} seconds"
    )  # noqa: T001

    # Extra step: Clean up the labelled dataset
    cleaned_data = tlu.clean_labelled_data(
        labelled_data=pd.DataFrame(
            load_jsonl(str(OUTPUT_FILEPATH / output_filename) + "_raw.jsonl")
        ),
        categories=tlu.load_categories(PATH_TO_CATEGORIES),
    )
    S3.upload_obj(
        obj=cleaned_data,
        bucket=S3_BUCKET,
        path_to=CONFIG["s3_directory"] + output_filename + ".parquet",
    )
