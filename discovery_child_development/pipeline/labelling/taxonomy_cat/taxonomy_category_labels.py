"""
Generate relevance labels using OpenAI API and prompt engineering

Example usage:
python discovery_child_development/pipeline/labelling/taxonomy_cat/taxonomy_category_labels.py --topic ai --num_samples 5 --model gpt-3.5-turbo-1106 --output_filename testing_ai
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
    create_examples_string,
)
from discovery_child_development.utils.utils import (
    create_directory_if_not_exists,
    batch,
)

from discovery_child_development.utils.jsonl_utils import load_jsonl

# Get labelling config params
CONFIG = get_yaml_config(Path(__file__).resolve().parent / "config.yaml")
# Path to dataset to use
EVALS_DIR = PROJECT_DIR / "outputs/labels/evals_data"
DATA_DIR = PROJECT_DIR / "outputs/enrichments"
PATH_TO_DATASET = DATA_DIR / "openalex_patents_relevance_labels_only_relevant.csv"
# Define paths to the prompt and function definitions
PATH_TO_PROMPTS = Path(__file__).resolve().parent / "prompts"
PATH_TO_CATEGORIES = PATH_TO_PROMPTS / "categories.json"
PATH_TO_MESSAGE_PROMPT = PATH_TO_PROMPTS / "prompt.json"
PATH_TO_FUNCTION = PATH_TO_PROMPTS / "function.json"
PATH_TO_TOPICS = PATH_TO_PROMPTS / "topics.json"
# Define paths to the outputs
OUTPUT_FILEPATH = PROJECT_DIR / CONFIG["local_output_directory"]


async def main(
    topic: str,
    num_samples: int,
    model: str,
    temperature: float,
    output_filename: str,
    s3_path: str,
    only_evals: bool = False,
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
    # Parse the topic file
    topics = json.load(open(PATH_TO_TOPICS, "r"))
    topic_info = topics[topic]

    # Get evaluated samples
    evals_df = pd.read_csv(EVALS_DIR / "taxonomy_labels_eval_annotated.csv")
    texts_with_evals = evals_df.query("prediction == @topic_info['name']").assign(
        id=lambda df: df["id"].apply(lambda x: x.split("/")[-1])
    )
    logging.info(f"Number of evaluated texts: {len(texts_with_evals)}")  # noqa: T001
    # Combine both samples
    if only_evals:
        texts_df = texts_with_evals[["id", "text", "source"]]
    else:
        # Get all relevant data
        texts_df_full = pd.read_csv(PATH_TO_DATASET)
        # Rough training data curation:
        # Fetch the texts that contain the specified keywords
        texts_with_keywords = texts_df_full[
            texts_df_full.text.str.contains(
                "|".join(topic_info["keywords"]), case=False
            )
        ].id.to_list()
        logging.info(
            f"Number of texts with keywords: {len(texts_with_keywords)}"
        )  # noqa: T001
        # Add random sample of texts without keywords
        texts_without_keywords = (
            texts_df_full.query("id not in @texts_with_keywords")
            .sample(800)
            .id.to_list()
        )
        texts_df = texts_df_full[
            texts_df_full.id.isin(texts_with_keywords + texts_without_keywords)
        ][["id", "text", "source"]]
        texts_df = (
            pd.concat([texts_df, texts_with_evals])
            .drop_duplicates(subset=["id"])
            .reset_index(drop=True)
        )

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

    # Subsample
    if (num_samples == 0) or (num_samples > len(texts_df)):
        num_samples = len(texts_df)
    texts_df = texts_df.sample(num_samples).reset_index(drop=True)

    logging.info(f"Number of texts to classify: {len(texts_df)}")  # noqa: T001

    message = MessageTemplate.load(str(PATH_TO_MESSAGE_PROMPT))
    function = FunctionTemplate.load(str(PATH_TO_FUNCTION))
    examples = create_examples_string(topic_info["examples"])

    for i, batched_results in enumerate(batch(texts_df, 10)):
        logging.info(f"Batch {i} / {len(texts_df) // 10}")  # noqa: T001
        tasks = [
            Classifier.agenerate(
                model=model,
                temperature=temperature,
                messages=[message],
                message_kwargs={
                    "name": topic_info["name"],
                    "description": topic_info["description"],
                    "keywords": topic_info["keywords"],
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
    # Add the arguments, and define defaults
    parser.add_argument("--topic", type=str, help="The category", default="ai")
    parser.add_argument("--model", type=str, help="The model", default=CONFIG["model"])
    parser.add_argument(
        "--num_samples",
        type=int,
        help="The number of samples",
        default=CONFIG["num_samples"],
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        help="The output filename",
        default=CONFIG["output_filename"],
    )
    parser.add_argument(
        "--only_evals", type=bool, help="Only use evaluated data", default=False
    )
    # Parse the arguments
    return parser.parse_args()


if "__main__" == __name__:
    # Define the arguments
    args = parse_arguments()
    try:
        topic = args.topic
    except ValueError:
        raise ValueError("You must provide a valid topic")
    model = args.model
    temperature = CONFIG["temperature"]
    num_samples = args.num_samples
    output_filename = args.output_filename + "_" + topic
    only_evals = args.only_evals

    # Create outputs directory if it doesn't exist
    create_directory_if_not_exists(OUTPUT_FILEPATH)
    # Define s3 path
    s3_path = CONFIG["s3_directory"] + output_filename + ".jsonl"

    # Label the data
    start = time.time()
    s = time.perf_counter()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            main(
                topic,
                num_samples,
                model,
                temperature,
                output_filename,
                s3_path,
                only_evals,
            )
        )
    finally:
        loop.close()
    e = time.perf_counter()

    logging.info(
        f"Concurrent execution completed in: {e - s:0.2f} seconds"
    )  # noqa: T001
