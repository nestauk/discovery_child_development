"""
Embed the OpenAlex abstracts using the sentence-transformers library.
--------------

For the existing dataset of OpenAlex docs (already preprocessed with 01_preprocess_openalex.py)
* create embeddings using HuggingFace sentence transformers and the 'all-MiniLM-L6-v2' model
* save the document IDs and vectors to a parquet file

Usage:

python discovery_child_development/pipeline/02_embed_openalex_docs.py

"""

import os
from dotenv import load_dotenv
from time import time
import pandas as pd
import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer

from nesta_ds_utils.loading_saving import S3

from discovery_child_development.getters import openalex
from discovery_child_development import PROJECT_DIR, logging

model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]

VECTORS_PATH = "data/openAlex/vectors/"
VECTORS_FILE = "sentence_vectors_384.parquet"

if __name__ == "__main__":
    openalex_text_df = openalex.get_abstracts()

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
    if len(openalex_docs) == len(vector_df):
        logging.info(f"Successfully embedded {len(openalex_docs)} docs")
    else:
        logging.warning("Embeddings were not created for all docs")

    S3.upload_obj(vector_df, S3_BUCKET, f"{VECTORS_PATH}{VECTORS_FILE}")
