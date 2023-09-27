# %%
import boto3
from io import BytesIO
import os
from dotenv import load_dotenv
from time import time
import pandas as pd
import numpy as np
import pickle

# %%
import sentence_transformers
from sentence_transformers import SentenceTransformer

# %%
model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
s3_client = boto3.client("s3")

# %%
load_dotenv()

# %%
S3_BUCKET = os.environ["S3_BUCKET"]

# %%

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

text_data_path = f"inputs/data/openAlex/openalex_abstracts_{CONCEPT_IDS}_year-2023.csv"
response = s3_client.get_object(Bucket=S3_BUCKET, Key=text_data_path)
csv_data = response["Body"].read()
openalex_text_df = pd.read_csv(BytesIO(csv_data), index_col=0)
openalex_text_df.head()

# %%
openalex_docs = openalex_text_df["text"].tolist()

# %%
# get rid of NaNs
print(f"Number of docs before removing NaNs: {len(openalex_docs)}")
openalex_docs = [doc for doc in openalex_docs if isinstance(doc, str)]
print(f"Number of docs after removing NaNs: {len(openalex_docs)}")

# %%
t0 = time()
sentence_vectors_384 = model.encode(openalex_docs, show_progress_bar=True)
print(f"vectorization done in {time() - t0:.3f} s")

# %%


# %%
# upload without using disk
array_data = BytesIO()
pickle.dump(sentence_vectors_384, array_data)
array_data.seek(0)
s3_client.upload_fileobj(
    array_data, S3_BUCKET, "inputs/data/openAlex/vectors/sentence_vectors_384.pkl"
)

# download without using disk
my_array_data2 = BytesIO()
s3_client.download_fileobj(
    S3_BUCKET, "inputs/data/openAlex/vectors/sentence_vectors_384.pkl", my_array_data2
)
my_array_data2.seek(0)
my_array2 = pickle.load(my_array_data2)

# check that everything is correct
np.allclose(sentence_vectors_384, my_array2)
