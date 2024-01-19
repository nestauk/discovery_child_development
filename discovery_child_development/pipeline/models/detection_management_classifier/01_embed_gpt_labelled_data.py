"""
Embed the GPT labelled abstracts using the sentence-transformers library for the simple classifiers.
--------------

For the existing dataset of GPT labelled docs
* create embeddings using HuggingFace sentence transformers and the 'all-MiniLM-L6-v2' model
* save the document IDs and vectors to a parquet file

Usage:

python discovery_child_development/pipeline/models/detection_management_classifier/01_embed_gpt_labelled_data.py

"""

from time import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from nesta_ds_utils.loading_saving import S3
from discovery_child_development.getters.labels import get_detection_management_labels
from discovery_child_development import logging, S3_BUCKET

model = SentenceTransformer("all-MiniLM-L6-v2")

VECTORS_PATH = "data/labels/detection_management_classifier/vectors/"
VECTORS_FILE = "sentence_vectors_384_labelled.parquet"

if __name__ == "__main__":
    labelled_text_df = get_detection_management_labels()

    labelled_docs = labelled_text_df["text"].tolist()
    labelled_ids = labelled_text_df["id"].tolist()

    # embed the titles & abstracts (these have already been concatenated in the column 'text')
    t0 = time()
    sentence_vectors_384 = model.encode(labelled_docs, show_progress_bar=True)
    print(f"vectorization done in {time() - t0:.3f} s")

    vectors_as_list = [list(vec) for vec in sentence_vectors_384]

    vector_df = pd.DataFrame({"id": labelled_ids, "miniLM_384_vector": vectors_as_list})
    if len(labelled_docs) == len(vector_df):
        logging.info(f"Successfully embedded {len(labelled_docs)} docs")
    else:
        logging.warning("Embeddings were not created for all docs")

    S3.upload_obj(vector_df, S3_BUCKET, f"{VECTORS_PATH}{VECTORS_FILE}")
