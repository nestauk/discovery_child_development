import boto3
from io import BytesIO
import os
from dotenv import load_dotenv
from time import time
import pandas as pd
import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer
import s3fs

model = SentenceTransformer("all-MiniLM-L6-v2")

s3_client = boto3.client("s3")

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

CONCEPT_IDS = [
    "C109260823",  # child development
    "C2993937534",  # childhood development
    "C2777082460",  # early childhood
    "C2911196330",  # child rearing
    "C2993037610",  # child care
    "C2779415726",  # child protection
    "C2781192327",  # child behavior checklist
    "C15471489",  # child psychotherapy
    "C178229462",  # early childhood education
    # "C138496976",  # developmental psychology (level 1).
]

CONCEPT_IDS = "|".join(CONCEPT_IDS)

WORKS_PATH = "data/openAlex/"
WORKS_FILE = f"openalex_abstracts_{CONCEPT_IDS}_year-2023.csv"
VECTORS_PATH = "data/openAlex/vectors/"


def read_openalex_works(works_data_path: str, s3_client, s3_bucket: str):
    """Read in the concepts metadata for all abstracts in the current OpenAlex dataset.

    Args:
        s3_client (_type_): boto3 client
        s3_bucket (str): name of the bucket
        works_data_path (str): Complete filepath within the bucket e.g. "path/to/your/data.csv"

    Returns:
        pd.DataFrame: A dataframe with the concepts metadata
    """
    response = s3_client.get_object(Bucket=s3_bucket, Key=works_data_path)
    csv_data = response["Body"].read()
    openalex_text_df = pd.read_csv(BytesIO(csv_data), index_col=0)

    return openalex_text_df


if __name__ == "__main__":
    fs = s3fs.S3FileSystem(
        anon=False, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY
    )

    openalex_text_df = read_openalex_works(
        works_data_path=f"{WORKS_PATH}{WORKS_FILE}",
        s3_bucket=S3_BUCKET,
        s3_client=s3_client,
    )

    openalex_docs = openalex_text_df["text"].tolist()
    openalex_ids = openalex_text_df["id"].tolist()

    # embed the titles & abstracts (these have already been concatenated in the column 'text')
    t0 = time()
    sentence_vectors_384 = model.encode(openalex_docs, show_progress_bar=True)
    print(f"vectorization done in {time() - t0:.3f} s")

    vectors_as_list = [list(vec) for vec in sentence_vectors_384]

    vector_df = pd.DataFrame(
        {"openalex_id": openalex_ids, "miniLM_384_vector": vectors_as_list}
    )

    filename = f"sentence_vectors_384.parquet"
    out_path = f"s3://{S3_BUCKET}/{VECTORS_PATH}{filename}"

    with fs.open(out_path, "wb") as f:
        vector_df.to_parquet(f)
