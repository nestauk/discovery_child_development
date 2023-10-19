"""
Embed the OpenAlex abstracts using the sentence-transformers library.
--------------

For the existing dataset of OpenAlex docs (already preprocessed with 00_preprocess_openalex.py)
* create embeddings using HuggingFace sentence transformers and the 'all-MiniLM-L6-v2' model
* save the document IDs and vectors to a parquet file

Usage:

python discovery_child_development/pipeline/embed_openalex_docs.py

"""

import os
from dotenv import load_dotenv
from time import time
import pandas as pd
import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer

from nesta_ds_utils.loading_saving import S3

model = SentenceTransformer("all-MiniLM-L6-v2")

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

WORKS_PATH = "data/openAlex/"
WORKS_FILE = f"openalex_abstracts_{CONCEPT_IDS}_year-2019-2020-2021-2022-2023.csv"
VECTORS_PATH = "data/openAlex/vectors/"
VECTORS_FILE = "sentence_vectors_384.parquet"

if __name__ == "__main__":
    openalex_text_df = S3.download_obj(
        S3_BUCKET,
        path_from=f"{WORKS_PATH}{WORKS_FILE}",
        download_as="dataframe",
        kwargs_reading={"index_col": 0},
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

    S3.upload_obj(vector_df, S3_BUCKET, f"{VECTORS_PATH}{VECTORS_FILE}")
