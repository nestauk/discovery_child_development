""" Appends locally labelled data to labelled data that is stored on S3.

Usage:
```
python discovery_child_development/pipeline/labelling/taxonomy/prodigy/export_data_to_s3.py
```
"""
import pandas as pd
import subprocess

from discovery_child_development import S3_BUCKET, PROJECT_DIR, logging
from discovery_child_development.utils import jsonl_utils as jsonl
from discovery_child_development.getters import taxonomy

LOCAL_PATH_LABELLED_DATA = PROJECT_DIR / "inputs/data/labelling/taxonomy/output"
PRODIGY_LABELLED_DATA_FILENAME = (
    "training_validation_data_patents_openalex_LABELLED_prodigy.jsonl"
)
LOCAL_PRODIGY_DATA = LOCAL_PATH_LABELLED_DATA / PRODIGY_LABELLED_DATA_FILENAME
DOWNLOADED_PRODIGY_DATA = (
    LOCAL_PATH_LABELLED_DATA
    / "training_validation_data_patents_openalex_LABELLED_prodigy_downloaded.jsonl"
)
prodigy_dataset = "taxonomy_data"
S3_PRODIGY_DATA_PATH = (
    f"data/labels/taxonomy_classifier/{PRODIGY_LABELLED_DATA_FILENAME}"
)

if __name__ == "__main__":
    # Export whatever data you have labelled locally from the prodigy database > local file
    command = f"prodigy db-out {prodigy_dataset} > {LOCAL_PRODIGY_DATA}"
    subprocess.run(command, shell=True, check=True)
    # ... and load in this data that you've just exported
    local_data = jsonl.load_jsonl(LOCAL_PRODIGY_DATA)

    # get prodigy labels from s3
    prodigy_data = pd.DataFrame(taxonomy.get_prodigy_labelled_data())

    # Compare IDs between data you have labelled locally, and data stored in s3.
    # We will append local data to s3 data, and we compare IDs so that we don't create duplicates.
    stored_prodigy_ids = set(prodigy_data["id"].unique())
    human_labels = pd.DataFrame(jsonl.load_jsonl(LOCAL_PRODIGY_DATA))
    local_ids = human_labels["id"].unique()
    novel_ids = set(local_ids) - set(stored_prodigy_ids)
    logging.info(
        f"Found {len(novel_ids)} local IDs that do not exist on S3. These will be uploaded to S3."
    )

    # Concatenate data stored on s3 and local labelled data
    output = (
        pd.concat([prodigy_data, human_labels[human_labels["id"].isin(novel_ids)]])
        .to_json(orient="records", lines=True)
        .split("\n")
    )
    output = [
        line for line in output if line != ""
    ]  # get rid of the empty string at the end of the list

    # overwrite local prodigy data with the combined data
    with open(LOCAL_PRODIGY_DATA, "w") as f:
        last_index = len(output) - 1
        for index, line in enumerate(output):
            if index == last_index:
                f.write(line)  # Don't add a newline at the end of the last line
            else:
                f.write(line + "\n")  # Add newline after each line except the last

    # Finally, upload data to s3
    jsonl.upload_file_to_s3(str(LOCAL_PRODIGY_DATA), S3_BUCKET, S3_PRODIGY_DATA_PATH)
