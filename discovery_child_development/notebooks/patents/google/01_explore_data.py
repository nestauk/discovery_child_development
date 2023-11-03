# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: discovery_child_development
#     language: python
#     name: python3
# ---

# # Explore patents data
#
# - Filter out patents where combinations of keywords don't appear in the same sentence
# - Inspect themes by clustering the data
# - Characterise the clusters by summarising the characteristic patents
# - Visualise the patent data

# +
from discovery_child_development.getters import patents
from discovery_child_development.utils import keywords as kw
import discovery_child_development.utils.cluster_analysis_utils as cau
import pandas as pd
import numpy as np

import importlib

importlib.reload(patents)
importlib.reload(kw)
importlib.reload(cau)

import altair as alt

alt.data_transformers.disable_max_rows()
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

SEED = 42

UMAP_PARAMS = {
    "n_components": 50,
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

# +
# Load keywords
keywords = patents.get_keywords_from_s3()

# Load patent data
data_raw_df = patents.get_patents_from_s3()

data_df = (
    data_raw_df
    # Combine title and abstract
    .assign(text=lambda df: df["title"] + ". " + df["abstract"])
    # Remove patents without text
    .dropna(subset=["text"])
    # Check which patents have keyword hits in the same sentence
    .assign(has_hits=lambda df: kw.check_keyword_hits(df.text, keywords))
)
# -

# Check proportion of patents that have the keyword hits in the same sentence
len(data_df.query("has_hits == True")) / len(data_df)

# Create embeddings for the unique concepts
embeddings = model.encode(data_df["text"].tolist(), show_progress_bar=True)

embeddings.shape

# +
# Reduce dimensionality of the embeddings
embeddings_50 = cau.umap_reducer(embeddings, UMAP_PARAMS, random_umap_state=SEED)

# Run with an arbitrary number of clusters
kmeans_labels = cau.kmeans_clustering(
    embeddings_50,
    kmeans_params={"init": "k-means++", "n_clusters": 20, "random_state": SEED},
)

# Reduce original vectors to 2D for plotting
embeddings_2d = cau.reduce_to_2D(embeddings, random_state=SEED)

# -

# Add 2D vectors into the dataframe for plotting
clusters_df = (
    data_df.copy()
    .reset_index(drop=True)
    .assign(
        cluster=kmeans_labels,
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
    )
)

# +
CLUSTER_SUMMARY_MESSAGE = "Here are the most central patents of a patent cluster. \
Describe what kind of innovations is this cluster capturing, in 2 sentences. \
\n\n##Abstracts\n\n {} \n\n##Description (2 short sentences)"

cluster_descriptions = cau.describe_clusters_with_gpt(
    cluster_df=cluster_df,
    embeddings=embeddings,
    n_central=10,
    gpt_message=CLUSTER_SUMMARY_MESSAGE,
)

cluster_names_dict = cau.generate_cluster_names_with_gpt(
    cluster_descriptions=cluster_descriptions,
)
# -

cluster_summaries = pd.DataFrame(
    data={
        "cluster": cluster_names_dict.keys(),
        "cluster_name": cluster_names_dict.values(),
        "cluster_description": cluster_descriptions,
    }
)

pd.set_option("display.max_colwidth", None)
cluster_summaries

# +
clusters_df_final = clusters_df.copy().merge(
    cluster_summaries, left_on="cluster", right_on="cluster", how="left"
)

fig = (
    alt.Chart(clusters_df_final)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("cluster_name:N", legend=alt.Legend(title="cluster name")),
        tooltip=["cluster", "cluster_name", "cluster_description", "title", "abstract"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig
# -

fig.save("patent_cluster_plot.html")
