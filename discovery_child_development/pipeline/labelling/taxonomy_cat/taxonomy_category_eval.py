from pathlib import Path
import argparse
import pandas as pd
import json

from nesta_ds_utils.loading_saving import S3
from discovery_child_development import logging, PROJECT_DIR, S3_BUCKET, get_yaml_config
from discovery_child_development.utils.jsonl_utils import load_jsonl

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# Get labelling config params
CONFIG = get_yaml_config(Path(__file__).resolve().parent / "config.yaml")

EVALS_DIR = PROJECT_DIR / "outputs/labels/evals_data"
LABELS_DIR = PROJECT_DIR / "outputs/labels/taxonomy_cat"

PATH_TO_PROMPTS = Path(__file__).resolve().parent / "prompts"
PATH_TO_TOPICS = PATH_TO_PROMPTS / "topics.json"


def parse_arguments():
    """Parse the arguments passed to the script."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the arguments for the script")
    # Add the arguments, and define defaults
    parser.add_argument("--topic", type=str, help="The category", default="ai")
    # Parse the arguments
    return parser.parse_args()


if "__main__" == __name__:
    topic = parse_arguments().topic
    output_filename = CONFIG["output_filename"] + "_" + topic
    s3_path = CONFIG["s3_directory"] + output_filename + ".jsonl"
    # Parse the topic file
    topics = json.load(open(PATH_TO_TOPICS, "r"))
    topic_info = topics[topic]

    # Load the eval data
    evals_df = (
        pd.read_csv(EVALS_DIR / "taxonomy_labels_eval_annotated.csv")
        .query("prediction == @topic_info['name']")
        .rename(columns={"answer": "human"})
        .assign(
            human=lambda x: x.human.map(
                {"reject": "Not-relevant", "accept": "Relevant"}
            )
        )
    )
    # Load the GPT-labelled data
    gpt_labels = (
        pd.DataFrame(load_jsonl(LABELS_DIR / f"taxonomy_cat_{topic}.jsonl"))
        .rename(columns={"prediction": "gpt"})
        .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
    )[["id", "gpt"]]
    #
    combined_df = (evals_df.merge(gpt_labels, on="id", how="left"))[
        ["id", "prediction", "source", "human", "gpt"]
    ]

    acc = accuracy_score(combined_df["human"], combined_df["gpt"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        combined_df["human"], combined_df["gpt"], average="weighted"
    )

    logging.info(f"Topic: {topic_info['name']}")  # noqa: T001
    logging.info(f"Accuracy: {round(acc,2)}")  # noqa: T001
    logging.info(f"Precision: {precision:.2f}")  # noqa: T001
    logging.info(f"Recall: {recall:.2f}")  # noqa: T001
    logging.info(f"F1: {f1:.2f}")  # noqa: T001

    # check if accuracy > 0.8
    if acc > 0.8:
        logging.info("Good! Accuracy is higher than 0.8!")  # noqa: T001

    # check how many samples are labelled by each category
    labels_by_cat = (
        gpt_labels.groupby("gpt")
        .agg(counts=("id", "count"))
        .sort_values("counts", ascending=False)
        .reset_index()
    )
    relevant = labels_by_cat.query("gpt == 'Relevant'")["counts"].values[0]
    not_relevant = labels_by_cat.query("gpt == 'Not-relevant'")["counts"].values[0]
    logging.info(f"Number of relevant labels: {relevant}")  # noqa: T001
    logging.info(f"Number of not relevant labels: {not_relevant}")  # noqa: T001

    # save the results in a text file (round everything to two decimals)
    with open(LABELS_DIR / f"taxonomy_cat_{topic}_log.txt", "w") as file:
        file.write(f"Topic: {topic_info['name']}\n")
        file.write("-------------\n")
        file.write(f"Total number of relevant labels: {relevant}\n")
        file.write(f"Total number of not relevant labels: {not_relevant}\n")
        file.write("-------------\n")
        file.write(f"Accuracy: {round(acc, 2)}\n")
        file.write(f"Precision: {round(precision, 2)}\n")
        file.write(f"Recall: {round(recall, 2)}\n")
        file.write(f"F1: {round(f1, 2)}\n")
        file.write("-------------\n")
        file.write(
            f"Good! Accuracy is higher than 0.8!"
            if acc > 0.8
            else "Danger! Accuracy is lower than 0.8"
        )
