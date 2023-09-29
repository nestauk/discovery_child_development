import boto3
from io import BytesIO
import os
from dotenv import load_dotenv
from time import time
import pandas as pd
import numpy as np
import pickle
import sentence_transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

s3_client = boto3.client("s3")

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]

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


def read_concepts_metadata(concepts_data_path: str, s3_client, s3_bucket: str):
    """Read in the concepts metadata for all abstracts in the current OpenAlex dataset.

    Args:
        s3_client (_type_): boto3 client
        s3_bucket (str): name of the bucket
        concepts_data_path (str): Complete filepath within the bucket e.g. "path/to/your/data.csv"

    Returns:
        pd.DataFrame: A dataframe with the concepts metadata
    """
    response = s3_client.get_object(Bucket=s3_bucket, Key=concepts_data_path)
    csv_data = response["Body"].read()
    openalex_text_df = pd.read_csv(BytesIO(csv_data), index_col=0)

    return openalex_text_df


def upload_to_s3(
    s3_client,
    s3_bucket,
    sentence_vectors_384,
    bucket_path="inputs/data/openAlex/vectors/",
):
    array_data = BytesIO()
    pickle.dump(sentence_vectors_384, array_data)
    array_data.seek(0)
    s3_client.upload_fileobj(
        array_data, s3_bucket, f"{bucket_path}sentence_vectors_384.pkl"
    )

    # download in order to check that uploading does not affect the vectors in any way
    my_array_data2 = BytesIO()
    s3_client.download_fileobj(
        s3_bucket, f"{bucket_path}sentence_vectors_384.pkl", my_array_data2
    )
    my_array_data2.seek(0)
    my_array2 = pickle.load(my_array_data2)

    # check that everything is correct
    if not np.allclose(sentence_vectors_384, my_array2):
        print("Warning: Arrays are not equal")


if __name__ == "__main__":
    openalex_text_df = read_concepts_metadata(
        concepts_data_path=f"inputs/data/openAlex/openalex_abstracts_{CONCEPT_IDS}_year-2023.csv",
        s3_bucket=S3_BUCKET,
        s3_client=s3_client,
    )

    openalex_docs = openalex_text_df["text"].tolist()

    # get rid of NaNs
    print(f"Number of docs before removing NaNs: {len(openalex_docs)}")
    openalex_docs = [doc for doc in openalex_docs if isinstance(doc, str)]
    print(f"Number of docs after removing NaNs: {len(openalex_docs)}")

    # embed the titles & abstracts (these have already been concatenated in the column 'text')
    t0 = time()
    sentence_vectors_384 = model.encode(openalex_docs, show_progress_bar=True)
    print(f"vectorization done in {time() - t0:.3f} s")

    upload_to_s3(
        s3_client,
        S3_BUCKET,
        sentence_vectors_384,
        bucket_path="inputs/data/openAlex/vectors/",
    )
