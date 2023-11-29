# %%
import altair as alt
from itertools import chain
import requests
from nesta_ds_utils.loading_saving import S3 as nesta_s3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import NoReturn, List, Any
from time import time
import sentence_transformers
from sentence_transformers import SentenceTransformer

from discovery_child_development import S3_BUCKET, config, logging
from discovery_child_development.utils import openalex_utils
from discovery_child_development.utils import cluster_analysis_utils as cau

API_ROOT = config["openalex_keywords_api_root"]
S3_PATH = "metaflow/openalex_keyword_search"
YEARS = config["openalex_years"]
KEYWORDS = config["openalex_keywords"]
SEED = config["seed"]

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

alt.data_transformers.disable_max_rows()

# %%
queries = openalex_utils.generate_keyword_queries(API_ROOT, KEYWORDS, YEARS)

queries

# %%
result = requests.get(queries[0])
# check that a single query runs ok
result

# %%
# find out how many hits we should get for all of the queries
total = 0

for query in queries:
    count = requests.get(query).json()["meta"]["count"]
    logging.info(f"Number of hits: {count}")
    total += count

total

# %% [markdown]
# The metaflow script `pipeline/openalex/openalex_keyword_search.py` runs all of the queries and stores the results on S3. Below, we load the results and do a little bit of EDA.

# %%
INPUT_FILES = [f"openalex_keywords_True_year-{year}.json" for year in YEARS]

# %%
openalex_df = openalex_utils.concat_json_files(INPUT_FILES, S3_BUCKET, S3_PATH)

len(openalex_df)

# %%
# Retain only works in English
openalex_en = openalex_df[openalex_df["language"] == "en"]
openalex_en = openalex_en[openalex_en["abstract_inverted_index"].notnull()]
openalex_en = openalex_en[openalex_en["title"].notnull()]

# %%
openalex_en_abstracts = openalex_utils.create_text_data(
    openalex_en[["id", "title", "abstract_inverted_index"]]
)

openalex_en_abstracts.head()

# %%
openalex_docs = openalex_en_abstracts["text"].tolist()

# %%
# check out a random example
openalex_docs[10]

# %%
t0 = time()
sentence_vectors_384 = model.encode(openalex_docs, show_progress_bar=True)
print(f"vectorization done in {time() - t0:.3f} s")

# %%
umap_params = {
    "n_components": 50,  # apparently hdbscan does not work very well with more than 50 components
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

# %%
# reduce dimensionality of the embeddings
sentence_vectors_50 = cau.umap_reducer(
    sentence_vectors_384, umap_params, random_umap_state=SEED
)

# %%
kmeans_labels = cau.kmeans_clustering(
    sentence_vectors_50, kmeans_params={"init": "k-means++", "n_clusters": 20}
)

# %%
# Reduce original vectors to 2D for plotting
openalex_texts_2d = cau.reduce_to_2D(sentence_vectors_384, random_state=SEED)

# %%
kmeans_labels

# %%
cluster_df = openalex_en_abstracts.assign(
    cluster=kmeans_labels,
    x=openalex_texts_2d[:, 0],
    y=openalex_texts_2d[:, 1],
)

# %%
fig_hdbscan = (
    alt.Chart(cluster_df)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("cluster:N", legend=alt.Legend(title="cluster")),
        tooltip=["title", "cluster"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig_hdbscan

# %%
# CLUSTER_SUMMARY_MESSAGE = "Here are the most central texts of a cluster. \
# Summarise what texts in this cluster are about in 2 sentences. \
# \n\n##Abstracts\n\n {} \n\n##Description (2 short sentences)"

# cluster_descriptions = cau.describe_clusters_with_gpt(
#     cluster_df=cluster_df,
#     embeddings=sentence_vectors_384,
#     n_central=10,
#     gpt_message=CLUSTER_SUMMARY_MESSAGE,
# )

# cluster_names_dict = cau.generate_cluster_names_with_gpt(
#     cluster_descriptions=cluster_descriptions,
# )

# %%
