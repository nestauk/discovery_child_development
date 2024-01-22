"""[IN PROGRESS]
Preprocess the text data for the binary classifier
--------------
To be used in the inference pipeline, the text data from various sources needs to be 
preprocessed. This script does the following:
* Splits the process into research sources for different preprocessing
* Openalex
    * Cleans the text data
    * Creates the text data for the binary classifier
* Crunchbase (TBD)
* Patents (TBD)
* Further sources (TBD)


Usage:

python discovery_child_development/pipeline/preprocessing/preprocessing_text_data.py

Optional arguments:
    --research_type : Determines whether to create the embeddings for the full dataset or a test sample (default: openalex)

"""

from discovery_child_development import (
    PROJECT_DIR,
    binary_config,
    config,
    S3_BUCKET,
    labelling_config,
    logging,
)
from discovery_child_development.getters.unlabelled_data import get_unprocessed_data
from discovery_child_development.utils.preprocess_openalex_utils import create_text_data
import argparse
from nesta_ds_utils.loading_saving import S3

S3_PATH = labelling_config["S3_PATH"]
FNAME = labelling_config["FNAME"]
extension = labelling_config["extension"]

if __name__ == "__main__":
    # Set up the command line arguments
    parser = argparse.ArgumentParser(
        description="Script to run with command line arguments."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="openalex",
        help="Which source do you want to preprocess? (default: openalex)",
    )

    # Parse the arguments
    args = parser.parse_args()
    logging.info(args)

    # Load the data
    data_for_cleaning = get_unprocessed_data(config=labelling_config)

    # Clean the data
    if args.source == "openalex":
        # Retain only works in English
        openalex_en = data_for_cleaning[
            data_for_cleaning["language"] == "en"
        ].reset_index(drop=True)

        openalex_en = openalex_en[
            (openalex_en["abstract_inverted_index"].notnull())
            & (openalex_en["title"].notnull())
        ].reset_index(drop=True)

        # Creating abstracts metadata
        openalex_en_abstracts = create_text_data(
            openalex_en[["id", "title", "abstract_inverted_index"]]
        )

        openalex_en_abstracts = openalex_en_abstracts[
            openalex_en_abstracts.text.notnull()
        ].reset_index(drop=True)

        openalex_en_abstracts = openalex_en_abstracts.drop_duplicates(
            subset=["id"], keep="first"
        ).reset_index(drop=True)

        data_for_binary_classifier = openalex_en_abstracts
    elif args.source == "crunchbase":
        # [ADD CLEANING FOR CRUNCHBASE]
        pass
    elif args.source == "patents":
        # [ADD CLEANING FOR PATENTS]
        pass
    else:
        logging.info("Please select a valid source: openalex, crunchbase, or patents")

    # Save the data
    if extension == "csv":
        S3.upload_obj(
            data_for_binary_classifier,
            S3_BUCKET,
            f"{S3_PATH}{FNAME}.{extension}",
        )
    elif extension == "json":
        data_for_binary_classifier = data_for_binary_classifier.to_dict()
        S3.upload_obj(
            data_for_binary_classifier,
            S3_BUCKET,
            f"{S3_PATH}{FNAME}.{extension}",
        )
    else:
        raise ValueError(
            f"The extension {extension} is not supported for cleaning. Please use one of the following: csv, json"
        )
