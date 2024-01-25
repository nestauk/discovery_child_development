"""
Usage:

python discovery_child_development/analysis/evaluate_labels/download_evals_data.py
"""

from discovery_child_development.utils.jsonl_utils import download_file_from_s3
from discovery_child_development import PROJECT_DIR, S3_BUCKET

OUTPUT_PATH = PROJECT_DIR / "outputs/labels/evals_data"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
S3_PATH = "data/labels/child_development/evals_data/"

if __name__ == "__main__":
    for filename in [
        "relevance_labels_eval.jsonl",
        "detection_management_labels_eval.jsonl",
    ]:
        download_file_from_s3(
            bucket_name=S3_BUCKET,
            s3_file_name=str(S3_PATH + filename),
            local_file=str(OUTPUT_PATH / filename),
        )
