"""
Embed the GPT labelled abstracts using the sentence-transformers library for the simple classifiers.
--------------

For the existing dataset of GPT labelled docs
* create embeddings using HuggingFace sentence transformers and the 'all-MiniLM-L6-v2' model
* save the document IDs and vectors to a parquet file

Usage:

python discovery_child_development/pipeline/models/taxonomy_classifier/02a_sentence_embeddings.py

"""
from nesta_ds_utils.loading_saving import S3 as nesta_s3
import pandas as pd
from sentence_transformers import SentenceTransformer
from time import time

from discovery_child_development import logging, S3_BUCKET, config, taxonomy_config
from discovery_child_development.getters import taxonomy_classifier

model = SentenceTransformer(taxonomy_config["sentence_embeddings_model"])

INPUT_DATA_PATH = taxonomy_config["s3_data_path"]
VECTORS_OUT_PATH = taxonomy_config["s3_vectors_path"]
VECTORS_FILENAME = taxonomy_config["s3_vectors_name"]


def embed_texts(
    split="train",
    s3_bucket=S3_BUCKET,
    vectors_out_path=VECTORS_OUT_PATH,
    vectors_filename=VECTORS_FILENAME,
):
    labelled_text_df = taxonomy_classifier.get_training_data(split=split)

    labelled_docs = labelled_text_df["text"].tolist()
    labelled_ids = labelled_text_df["id"].tolist()

    # embed the titles & abstracts (these have already been concatenated in the column 'text')
    logging.info(f"Creating sentence embeddings for the {split} split...")
    t0 = time()
    sentence_vectors_384 = model.encode(labelled_docs, show_progress_bar=True)
    print(f"vectorization done in {time() - t0:.3f} s")

    vectors_as_list = [list(vec) for vec in sentence_vectors_384]

    vector_df = pd.DataFrame({"id": labelled_ids, "miniLM_384_vector": vectors_as_list})
    if len(labelled_docs) == len(vector_df):
        logging.info(f"Successfully embedded {len(labelled_docs)} docs")
    else:
        logging.warning("Embeddings were not created for all docs")

    nesta_s3.upload_obj(
        vector_df,
        s3_bucket,
        f"{vectors_out_path}{vectors_filename.replace('SPLIT', split)}",
    )
    logging.info(
        f"Successfully uploaded '{vectors_filename.replace('SPLIT', split)}' to S3"
    )


if __name__ == "__main__":
    for split in ["train", "test", "val"]:
        embed_texts(split=split)
