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
# - Clusters
# - Cluster keywords
# - Visualisation

# +
from nesta_ds_utils.loading_saving import S3
import os
import numpy as np
import pandas as pd
import copy

from ast import literal_eval
import discovery_child_development.utils.cluster_analysis_utils as cau
import altair as alt
from sentence_transformers import SentenceTransformer
from typing import List
import openai

model = SentenceTransformer("all-MiniLM-L6-v2")

PATENTS_PATH = "data/patents/"

openai.api_key = os.environ["OPENAI_API_KEY"]

# +
data_df = S3.download_obj(
    os.environ["S3_BUCKET"],
    f"{PATENTS_PATH}GooglePatents_test_data.csv",
    download_as="dataframe",
)

data_df = data_df.assign(text=lambda df: df["title"] + " " + df["abstract"]).dropna(
    subset=["text"]
)


# +
def convert_google_embedding_to_vector(string: str) -> np.ndarray:
    """Convert a string of space-separated numbers to a numpy array"""
    # Remove brackets
    array = string.split("[")[1].split("]")[0].strip()
    # Split by spaces
    array = "[" + ",".join([x for x in array.split(" ") if x != ""]) + "]"
    return np.array(literal_eval(array))


def get_google_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Get the Google embeddings for a dataframe"""
    return np.vstack(
        np.array(df.embedding_v1.apply(convert_google_embedding_to_vector))
    )


# -

# Convert embeddings (not used)
embeddings = get_google_embeddings(data_df)

# Create embeddings for the unique concepts
embeddings = model.encode(data_df["text"].tolist(), show_progress_bar=True)

# +
SEED = 42

umap_params = {
    "n_components": 50,
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

# Reduce dimensionality of the embeddings
embeddings_50 = cau.umap_reducer(embeddings, umap_params, random_umap_state=SEED)

# Run with an arbitrary number of clusters
kmeans_labels = cau.kmeans_clustering(
    embeddings_50,
    kmeans_params={"init": "k-means++", "n_clusters": 15, "random_state": SEED},
)

# Reduce original vectors to 2D for plotting
embeddings_2d = cau.reduce_to_2D(embeddings, random_state=SEED)


# +
# Add 2D vectors into the dataframe for the sake of plotting
embeddings_df = (
    data_df.copy()
    .reset_index(drop=True)
    .assign(
        cluster=kmeans_labels,
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
    )
)

cluster_keywords = cau.cluster_keywords(
    documents=embeddings_df["text"], cluster_labels=embeddings_df["cluster"], n=10
)
for i in range(len(cluster_keywords)):
    print(f"{i} : {cluster_keywords[i]}")


embeddings_df["cluster_keywords"] = embeddings_df["cluster"].apply(
    lambda x: cluster_keywords[x]
)

fig = (
    alt.Chart(embeddings_df)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("cluster:N", legend=alt.Legend(title="cluster")),
        tooltip=["title", "abstract", "cluster", "cluster_keywords"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig


# +
# Get the most central data points for each cluster


# get centroids
def get_centroids(data_df: pd.DataFrame, embeddings: np.ndarray) -> List[np.ndarray]:
    """Get the centroids of each cluster"""
    centroids = []
    for i in range(len(data_df["cluster"].unique())):
        cluster = data_df[data_df["cluster"] == i]
        centroid = np.mean(embeddings[cluster.index], axis=0)
        centroids.append(centroid)
    return centroids


def get_n_most_similar(
    embeddings: np.ndarray, centroid: np.ndarray, n: int = 10
) -> np.ndarray:
    """Get the n most similar data points to a centroid"""
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return np.argsort(distances)[:n]


centroids = get_centroids(embeddings_df, embeddings)
most_central = []
for i in range(len(centroids)):
    most_central.append(get_n_most_similar(embeddings, centroids[i], n=10))
# -

base_message = "Here are 10 most central patents of a patent cluster. Describe what kind of innovations is this cluster capturing, in 2 sentences.\n\n##Abstracts\n\n {} \n\n##Description (2 short sentences)"


chatgpt_outputs = []
for i in range(len(centroids)):
    abstracts = data_df.iloc[most_central[i]].text.to_list()
    messages = [
        {
            "role": "user",
            "content": copy.deepcopy(base_message).format("\n".join(abstracts)),
        }
    ]
    chatgpt_output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.6,
        max_tokens=1000,
    ).to_dict()
    chatgpt_outputs.append(chatgpt_output["choices"][0]["message"]["content"])

cluster_descriptions = chatgpt_outputs.copy()
cluster_descriptions = [f"{i}: {x}" for i, x in enumerate(cluster_descriptions)]

base_summarising_message = (
    "Summarise these cluster descriptions in 2-3 words.\n\n##Descriptions\n\n {}"
)

messages = [
    {
        "role": "user",
        "content": copy.deepcopy(base_summarising_message).format(
            "\n".join(cluster_descriptions)
        ),
    }
]
chatgpt_output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.6,
    max_tokens=1000,
).to_dict()

cluster_names = chatgpt_output["choices"][0]["message"]["content"].split("\n")

pd.set_option("display.max_colwidth", None)
df = pd.DataFrame(cluster_descriptions, columns=["description"])
df["cluster_name"] = cluster_names
df

df = df.reset_index().rename(columns={"index": "cluster_id"})

# +
embeddings_df_ = embeddings_df.copy().merge(
    df, left_on="cluster", right_on="cluster_id", how="left"
)

fig = (
    alt.Chart(embeddings_df_)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("cluster:N", legend=alt.Legend(title="cluster")),
        tooltip=["cluster", "cluster_name", "description", "title", "abstract"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig
# -

fig.save("cluster_plot.html")
