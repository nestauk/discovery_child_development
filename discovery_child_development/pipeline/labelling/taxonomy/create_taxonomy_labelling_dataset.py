"""
Generates a roughly balanced sample of OpenAlex and patents data for labelling.

The construction of the OpenAlex abstracts sample is:
* SAMPLE_SIZE per category from the taxonomy (with the taxonomy mapped to the abstracts via the concepts metadata) in order to include texts that should in theory represent all the areas of the taxonomy
* SAMPLE_SIZE * 10 of the abstracts that had no concepts metadata (these are "wildcards" in that they could be about anything!)

The construction of the patents sample is:
* SAMPLE_SIZE * number of categories from the taxonomy. The patents are chosen totally at random.

"""
import argparse
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
SAMPLE_SIZE = 100

PATH_TO_PROMPTS = (
    PROJECT_DIR / "discovery_child_development/notebooks/labelling/prompts/taxonomy"
)
PATH_USER = PATH_TO_PROMPTS / "user.json"
PATH_SYSTEM = PATH_TO_PROMPTS / "system.json"
PATH_FUNCTION = PATH_TO_PROMPTS / "function.json"

DATA_LABELLING_PATH = PROJECT_DIR / "inputs/data/labelling/taxonomy/input"
OUTPUT_FILENAME = "training_validation_data_patents_openalex.jsonl"

S3_OUTPUT_PATH = f"data/labels/taxonomy_classifier/{OUTPUT_FILENAME}"


def clean_openalex_id(df: pd.DataFrame, column_name: str = "id") -> pd.DataFrame:
    """
    Clean the OpenAlex ID to remove the prefix: you get just the last few characters eg 'W3154976785',
    not the full URL.
    """
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


def prepare_patents(patent_sample_size: int, seed: int = SEED) -> pd.DataFrame:
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
        columns={
            "publication_number": "id"
        }  # We are using publication_number as the unique identifier for patents
    )
    patent_sample[
        "label"
    ] = ""  # This data has no labels, but we need a 'label' column so that the dataframes can be concatenated
    patent_sample["source"] = "patents"

    return patent_sample


def prepare_openalex(
    sample_size: int = SAMPLE_SIZE,
    no_concept_sample_size: int = SAMPLE_SIZE * 10,
    seed: int = SEED,
    score_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    This function prepares a dataset of OpenAlex data to be labelled by GPT.

    It joins the OpenAlex abstracts with the taxonomy from Google Sheets (using the concepts metadata to make this join),
    meaning that <sample_size> abstracts from each taxonomy category go into the sample.

    In addition, some concept-free OpenAlex data is included: this comprises 10 * <sample_size> abstracts.

    The function performs the following steps:
    1. Retrieves OpenAlex abstracts and cleans their IDs.
    2. Filters out abstracts that are part of a test set to avoid data leakage.
    3. Further filters abstracts to include only those that contain both child-related and taxonomy-related terms.
    4. Retrieves and processes OpenAlex concepts metadata, to include only concepts that are in the taxonomy, and only concepts with a score over <score_threshold>
    5. Merges the taxonomy data with concepts metadata.
    6. Performs an outer join with the abstracts to include abstracts without concept metadata.
    7. Samples <sample_size> abstracts from each taxonomy category.
    7. Samples <no_concept_sample_size> abstracts that have no concept metadata.
    8. Concatenates these two datasets and adds a source label.

    Parameters:
    - sample_size (int): Size of the sample to be drawn from each category.
    - no_concept_sample_size (int): Size of the sample for abstracts with no associated concepts.
    - seed (int): Random seed for sample generation to ensure reproducibility.
    - score_threshold (float): Minimum score threshold for including concepts in the dataset.

    Returns:
    - pd.DataFrame: A DataFrame containing the prepared OpenAlex data, suitable for further analysis and modeling.
    """

    openalex_data = oa.get_abstracts()
    # For all the datasets, we want just the last set of characters from the ID, not the full URL.
    # No particular reason for this but it just seems a bit neater
    openalex_data = clean_openalex_id(openalex_data, "id")

    # Make sure none of the data for our labelling dataset comes from the test set.
    # With hindsight, maybe this isn't necessary because we will need to label our test set at some point too I think
    test_data, _ = oa.get_labelled_data(score_threshold=0.0, train=False)
    test_data = clean_openalex_id(test_data, "openalex_id")
    test_ids = test_data["openalex_id"].unique()
    openalex_data_subset = openalex_data[~openalex_data["id"].isin(test_ids)]

    # Filter the data to only abstracts that contain both a child-related word AND a taxonomy word
    openalex_data_filtered = filter_abstracts(openalex_data_subset)
    abstract_ids = openalex_data_filtered["id"].unique()

    # Find the concepts metadata for the abstracts that we have filtered (but NB not all abstracts will have concepts!). We only include:
    # * concepts that are in the taxonomy
    # * concepts with a score above the threshold
    taxonomy_data = taxonomy.get_taxonomy()
    taxonomy_concept_ids = taxonomy_data["concept_id"].unique()

    openalex_concepts = oa.get_concepts_metadata()
    openalex_concepts = clean_openalex_id(openalex_concepts, "openalex_id")
    openalex_concepts_subset = openalex_concepts[
        openalex_concepts["concept_id"].isin(taxonomy_concept_ids)
        & openalex_concepts["openalex_id"].isin(abstract_ids)
        & openalex_concepts["score"]
        >= score_threshold
    ].copy()

    # Merge the taxonomy and the concepts metadata...
    openalex_concepts_subset = pd.merge(
        openalex_concepts_subset,
        taxonomy_data[["sub_category", "concept_id"]],
        how="left",
        on="concept_id",
    )

    # ... and then merge the taxonomy/concepts metadata with the abstracts
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
        how="outer",  # It's an outer join because we want to keep the abstracts that don't have any concepts metadata
    )

    # Some OpenAlex abstracts have no concepts. We want to make sure these are represented in our sample too.
    openalex_data_no_concepts = openalex_data_merged[
        openalex_data_merged["concept_id"].isna()
    ].sample(n=no_concept_sample_size, random_state=seed)

    # Take a sample from each taxonomy category ("sub_category")
    openalex_sample = openalex_data_merged.groupby(
        "sub_category", group_keys=False
    ).apply(sample_per_category, sample_size)
    openalex_sample = openalex_sample.rename(columns={"sub_category": "label"})

    openalex_sample = (
        openalex_sample[["id", "text", "label"]]
        .groupby(["id", "text"])["label"]
        # Squish the unique labels together into one cell as a list
        .agg(lambda x: list(set(x)))
        .reset_index()
    )

    openalex_data_no_concepts = openalex_data_no_concepts.rename(
        columns={"sub_category": "label"}
    )
    openalex_data_no_concepts = (
        openalex_data_no_concepts[["id", "text", "label"]]
        .groupby(["id", "text"])["label"]
        # Squish the unique labels together into one cell as a list
        .agg(lambda x: list(set(x)))
        .reset_index()
    )

    openalex_sample = pd.concat([openalex_sample, openalex_data_no_concepts])
    openalex_sample[
        "source"
    ] = "openalex"  # in the overall dataset, this column identifies whether the text is OpenAlex or patents

    return openalex_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameters for creating a balanced dataset for the taxonomy classifier"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        dest="sample_size",
        help="How many datapoints do you want to sample per taxonomy category per data source (OpenAlex/patents)? The number of patents sampled will be sample_size*<number of taxonomy categories>",
        default=SAMPLE_SIZE,
    )

    args, unknown = parser.parse_known_args()

    logging.info("Preparing OpenAlex sample...")

    # number of OpenAlex abstracts that do not have any concepts - these are "wildcards" in that they could be about anything!
    NO_CONCEPT_SAMPLE_SIZE = args.sample_size * 10

    openalex_sample = prepare_openalex(
        sample_size=args.sample_size,
        no_concept_sample_size=NO_CONCEPT_SAMPLE_SIZE,
        seed=SEED,
    )

    categories_flat = tlu.load_categories()
    # The number of patents sampled will be sample_size * <number of taxonomy categories>
    PATENT_SAMPLE_SIZE = len(categories_flat.keys()) * args.sample_size
    logging.info("Preparing Google patents sample...")
    patent_sample = prepare_patents(PATENT_SAMPLE_SIZE)

    data_sample = pd.concat([openalex_sample, patent_sample])

    logging.info(data_sample["source"].value_counts())

    data_for_labelling = (
        data_sample[["id", "text", "source"]]
        .to_json(orient="records", lines=True)
        .split("\n")
    )

    data_for_labelling = [
        line for line in data_for_labelling if line != ""
    ]  # the code above leaves an empty string at the end of the list

    logging.info(f"Saving data sample to {DATA_LABELLING_PATH}/{OUTPUT_FILENAME}...")
    utils.create_directory_if_not_exists(DATA_LABELLING_PATH)
    with open(DATA_LABELLING_PATH / OUTPUT_FILENAME, "w") as f:
        last_index = len(data_for_labelling) - 1
        for index, line in enumerate(data_for_labelling):
            if index == last_index:
                f.write(line)  # Don't add a newline at the end of the last line
            else:
                f.write(line + "\n")  # Add newline after each line except the last

    logging.info("Uploading prepared sample to S3...")
    jsonl.upload_file_to_s3(
        str(DATA_LABELLING_PATH / OUTPUT_FILENAME), S3_BUCKET, S3_OUTPUT_PATH
    )
