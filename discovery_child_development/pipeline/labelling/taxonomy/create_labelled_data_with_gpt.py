from discovery_child_development import config, PROJECT_DIR, S3_BUCKET, logging
from discovery_child_development.getters import taxonomy
from discovery_child_development.utils import jsonl_utils as jsonl
from discovery_child_development.utils import taxonomy_labelling_utils as tlu
from discovery_child_development.utils.openai_utils import client

from nesta_ds_utils.loading_saving import S3 as nesta_s3

import argparse
import pandas as pd
import random
import tiktoken
import warnings

MODEL = "gpt-3.5-turbo-1106"  # "gpt-4"

MODEL_INPUT_COST, MODEL_OUTPUT_COST = tlu.get_model_cost(MODEL)
SEED = config["seed"]

random.seed(SEED)

encoding = tiktoken.encoding_for_model(MODEL)

S3_PATH = "data/labels/taxonomy_classifier/labelled_with_gpt/training_validation_data_patents_openalex_GPT_LABELLED.parquet"
S3_PATH_TEST = "data/labels/taxonomy_classifier/labelled_with_gpt/training_validation_data_patents_openalex_GPT_LABELLED_test.parquet"


def unique_list(series):
    return list(set(series))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--production",
        type=lambda x: (str(x).lower() == "true"),  # type=bool,
        default=False,
        help="Do you want to run the code in production? (default: False)",
    )

    args = parser.parse_args()

    data_to_be_labelled = pd.DataFrame(taxonomy.get_labelling_sample())

    n_duplicates = len(data_to_be_labelled) - len(data_to_be_labelled["id"].unique())
    if n_duplicates > 0:
        warnings.warn("There may be duplicate texts in the dataset")

    if args.production == False:
        logging.info("Running in test mode - only labelling 30 examples")
        data_to_be_labelled = data_to_be_labelled.head(30)
    else:
        logging.info("Running in production mode - labelling all examples")

    categories_flat = tlu.load_categories()

    function = tlu.format_function(categories_flat)

    labelled_data = []

    idx = 0
    for index, row in data_to_be_labelled.iterrows():
        print(idx)
        idx += 1
        prompt = tlu.build_prompt(row["text"], categories_flat)
        r = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=prompt,
            functions=[function],
            function_call={"name": "predict_category"},
        )
        llm_output = tlu.get_labels_from_gpt_response(r)
        output = {
            "id": row["id"],
            "text": row["text"],
            "source": row["source"],
            "labels": llm_output,
            "tokens_input": r.usage.prompt_tokens,
            "tokens_output": r.usage.completion_tokens,
            "cost": (MODEL_INPUT_COST * (r.usage.prompt_tokens / 1000))
            + (MODEL_OUTPUT_COST * (r.usage.completion_tokens / 1000)),
        }
        labelled_data.append(output)

    labelled_data = pd.DataFrame(labelled_data)

    labelled_data = labelled_data.explode("labels")

    category_names = list(categories_flat.keys())

    new_category_dict = tlu.make_keyword_dict(categories_flat)

    labelled_data["category"] = labelled_data["labels"].apply(
        lambda x: tlu.map_keywords_to_categories(x, new_category_dict, category_names)
    )

    labelled_data_cleaned = labelled_data[labelled_data["category"] != "no label"]

    labelled_data_cleaned = (
        labelled_data_cleaned.groupby(["id", "text", "source", "cost"])
        .agg({"labels": unique_list, "category": unique_list})
        .reset_index()
        .rename(columns={"labels": "labels_raw", "category": "labels"})
    )

    if args.production == True:
        logging.info(f"Uploading labelled data to S3: {S3_BUCKET}/{S3_PATH}")
        nesta_s3.upload_obj(labelled_data_cleaned, S3_BUCKET, S3_PATH)
    else:
        logging.info(f"Uploading labelled data to S3: {S3_BUCKET}/{S3_PATH_TEST}")
        nesta_s3.upload_obj(labelled_data_cleaned, S3_BUCKET, S3_PATH_TEST)
