"""
Generates a roughly balanced sample of OpenAlex and patents data for labelling.

The construction of the OpenAlex abstracts sample is:
* SAMPLE_SIZE per category from the taxonomy (with the taxonomy mapped to the abstracts via the concepts metadata) in order to include texts that should in theory represent all the areas of the taxonomy
* SAMPLE_SIZE * 10 of the abstracts that had no concepts metadata (these are "wildcards" in that they could be about anything!)

The construction of the patents sample is:
* SAMPLE_SIZE * number of categories from the taxonomy. The patents are chosen totally at random.

"""

import pandas as pd
import re

from discovery_child_development import PROJECT_DIR, config, S3_BUCKET, logging
from discovery_child_development.getters import openalex as oa
from discovery_child_development.getters import taxonomy, patents
from discovery_child_development.utils import taxonomy_labelling_utils as tlu
from discovery_child_development.utils import utils
from discovery_child_development.utils import keywords as kw
from discovery_child_development.utils import jsonl_utils as jsonl

SEED = config["seed"]
KEYWORDS = config["openalex_keywords"]
KEYWORDS = [term.replace("'", "") for term in KEYWORDS]
SAMPLE_SIZE = 10
# number of OpenAlex abstracts that do not have any concepts - these are "wildcards" in that they could be about anything!
NO_CONCEPT_SAMPLE_SIZE = SAMPLE_SIZE * 10

PATH_TO_PROMPTS = (
    PROJECT_DIR / "discovery_child_development/notebooks/labelling/prompts/taxonomy"
)
PATH_USER = PATH_TO_PROMPTS / "user.json"
PATH_SYSTEM = PATH_TO_PROMPTS / "system.json"
PATH_FUNCTION = PATH_TO_PROMPTS / "function.json"

DATA_LABELLING_PATH = PROJECT_DIR / "inputs/data/labelling/taxonomy/input"
OUTPUT_FILENAME = "training_validation_data_patents_openalex.jsonl"

S3_OUTPUT_PATH = f"data/labels/taxonomy_classifier/{OUTPUT_FILENAME}"


def clean_openalex_id(df, column_name="id"):
    """Cleans the OpenAlex ID to remove the prefix"""
    df[column_name] = df[column_name].str.extract(r"/(W\d+)$")
    return df


def sample_per_category(group, sample_size=SAMPLE_SIZE, seed=SEED):
    return group.sample(min(sample_size, len(group)), random_state=seed)


def filter_abstracts(df, keywords=KEYWORDS):
    """Filter the OpenAlex abstracts so that they have to contain one child-related word, AND a word from the taxonomy."""
    # Define the lists of terms
    child_terms = [
        "infant",
        "infancy",
        "child",
        "children",
        "prenatal",
        "neonatal",
        "pregnant",
        "pregnancy",
        "bab(?:y|ies)",
    ]

    # Create regex patterns
    first_pattern = "|".join(child_terms)
    second_pattern = "|".join(keywords)

    # Filter the DataFrame
    filtered_df = df[
        df["text"].str.contains(first_pattern, flags=re.IGNORECASE, na=False)
        & df["text"].str.contains(second_pattern, flags=re.IGNORECASE, na=False)
    ]

    return filtered_df


def prepare_patents(patent_sample_size, seed=SEED):
    keywords = patents.get_keywords_from_s3()

    # Load patent data
    data_raw_df = patents.get_patents_from_s3()

    data_df = (
        data_raw_df
        # Remove patents without text
        .dropna(subset=["title", "abstract"])
        # Combine title and abstract
        .assign(text=lambda df: df["title"] + ". " + df["abstract"])
        # Check which patents have keyword hits in the same sentence
        .assign(has_hits=lambda df: kw.check_keyword_hits(df.text, keywords))
    )

    patent_sample = data_df.sample(n=patent_sample_size, random_state=seed)

    patent_sample = patent_sample[["publication_number", "text"]].rename(
        columns={"publication_number": "id"}
    )
    patent_sample["label"] = ""
    patent_sample["source"] = "patents"

    return patent_sample


def prepare_openalex(
    sample_size=SAMPLE_SIZE, no_concept_sample_size=NO_CONCEPT_SAMPLE_SIZE, seed=SEED
):
    openalex_data = oa.get_abstracts()
    openalex_data = clean_openalex_id(openalex_data, "id")

    test_data, _ = oa.get_labelled_data(score_threshold=0.0, train=False)
    test_data = clean_openalex_id(test_data, "openalex_id")

    test_ids = test_data["openalex_id"].unique()

    openalex_data_subset = openalex_data[~openalex_data["id"].isin(test_ids)]

    openalex_data_filtered = filter_abstracts(openalex_data_subset)

    abstract_ids = openalex_data_filtered["id"].unique()

    openalex_concepts = oa.get_concepts_metadata()
    openalex_concepts = clean_openalex_id(openalex_concepts, "openalex_id")
    openalex_concepts = openalex_concepts[
        openalex_concepts["openalex_id"].isin(abstract_ids)
    ]

    taxonomy_data = taxonomy.get_taxonomy()

    taxonomy_concept_ids = taxonomy_data["concept_id"].unique()

    openalex_concepts_subset = openalex_concepts[
        openalex_concepts["concept_id"].isin(taxonomy_concept_ids)
        & openalex_concepts["score"]
        >= 0.6
    ].copy()

    # merge taxonomy
    openalex_concepts_subset = pd.merge(
        openalex_concepts_subset,
        taxonomy_data[["sub_category", "concept_id"]],
        how="left",
        on="concept_id",
    )

    openalex_data_merged = openalex_concepts_subset[
        [
            "openalex_id",
            "concept_id",
            "sub_category",
            "display_name",
            "level",
            "score",
        ]
    ].merge(
        openalex_data[["id", "text"]],
        left_on="openalex_id",
        right_on="id",
        how="outer",
    )

    openalex_data_no_concepts = openalex_data_merged[
        openalex_data_merged["concept_id"].isna()
    ].sample(n=no_concept_sample_size, random_state=seed)

    openalex_sample = openalex_data_merged.groupby(
        "sub_category", group_keys=False
    ).apply(sample_per_category, sample_size)
    openalex_sample = openalex_sample.rename(columns={"sub_category": "label"})

    openalex_sample = (
        openalex_sample[["id", "text", "label"]]
        .groupby(["id", "text"])["label"]
        .agg(lambda x: list(set(x)))
        .reset_index()
    )

    openalex_data_no_concepts = openalex_data_no_concepts.rename(
        columns={"sub_category": "label"}
    )
    openalex_data_no_concepts = (
        openalex_data_no_concepts[["id", "text", "label"]]
        .groupby(["id", "text"])["label"]
        .agg(lambda x: list(set(x)))
        .reset_index()
    )

    openalex_sample = pd.concat([openalex_sample, openalex_data_no_concepts])
    openalex_sample["source"] = "openalex"

    return openalex_sample


if __name__ == "__main__":
    logging.info("Preparing OpenAlex sample...")
    openalex_sample = prepare_openalex(
        sample_size=SAMPLE_SIZE,
        no_concept_sample_size=NO_CONCEPT_SAMPLE_SIZE,
        seed=SEED,
    )

    categories_flat = tlu.load_categories()
    PATENT_SAMPLE_SIZE = len(categories_flat.keys()) * SAMPLE_SIZE
    logging.info("Preparing Google patents sample...")
    patent_sample = prepare_patents(PATENT_SAMPLE_SIZE)

    data_sample = pd.concat([openalex_sample, patent_sample])

    logging.info(data_sample["source"].value_counts())

    data_for_labelling = (
        data_sample[["id", "text", "source"]]
        .to_json(orient="records", lines=True)
        .split("\n")
    )

    logging.info(f"Saving data sample to {DATA_LABELLING_PATH}/{OUTPUT_FILENAME}...")
    utils.create_directory_if_not_exists(DATA_LABELLING_PATH)
    with open(DATA_LABELLING_PATH / OUTPUT_FILENAME, "w") as f:
        for line in data_for_labelling:
            f.write(line + "\n")

    logging.info("Uploading prepared sample to S3...")
    jsonl.upload_file_to_s3(
        str(DATA_LABELLING_PATH / OUTPUT_FILENAME), S3_BUCKET, S3_OUTPUT_PATH
    )
