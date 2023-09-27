# This notebook uses data created by `discovery_child_development/pipeline/openalex_metaflow.py`.
#
# The steps in this notebook are:
# * create TFIDF vectors, cluster and visualise these (in order to see if simple vectors work OK)
# * Cluster and visualise sentence embeddings for the OpenAlex texts
# * Find the top terms per cluster

# +
import boto3
from io import BytesIO
import os
from dotenv import find_dotenv, load_dotenv
from time import time
import pandas as pd
import numpy as np
import pickle

import umap.umap_ as umap
import hdbscan
import altair as alt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# -

# We store the concepts we're using as a variable so that they can be pasted into file names (the metaflow process stores the concepts that were used to get the data in the file name, and it seems wise to preserve this information at different stages of the process).
#
# At some point it may help to store these in a config file, in case we want to work with different sets of concepts at the same time for comparison.

CONCEPTS = "C109260823|C2993937534|C2777082460|C2911196330|C2993037610|C2779415726|C2781192327|C15471489|C178229462"

# +
load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]

s3_client = boto3.client("s3")
# -

# Read in the text data
text_data_path = f"inputs/data/openAlex/openalex_abstracts_{CONCEPTS}_year-2023.csv"
response = s3_client.get_object(Bucket=S3_BUCKET, Key=text_data_path)
csv_data = response["Body"].read()
openalex_text_df = pd.read_csv(BytesIO(csv_data), index_col=0)
openalex_text_df.head()

# Put the text in a format suitable for embedding, clustering etc
openalex_docs = openalex_text_df["text"].tolist()

# get rid of NaNs
print(f"Number of docs before removing NaNs: {len(openalex_docs)}")
openalex_docs = [doc for doc in openalex_docs if isinstance(doc, str)]
print(f"Number of docs after removing NaNs: {len(openalex_docs)}")

# apply the same filtering on the dataframe
print(f"Number of docs in dataframe before removing NaNs: {len(openalex_text_df)}")
openalex_text_df = openalex_text_df[
    openalex_text_df["text"].apply(lambda x: isinstance(x, str))
]
print(f"Number of docs in dataframe AFTER removing NaNs: {len(openalex_text_df)}")

# The embeddings were computed separately because this part can be time consuming. We can simply load them from the pickle file:

# +
# Load embeddings
buffer = BytesIO()
s3_client.download_fileobj(
    S3_BUCKET, "inputs/data/openAlex/vectors/sentence_vectors_384.pkl", buffer
)
buffer.seek(0)
openalex_vectors = pickle.load(buffer)

# Normalize the vectors
openalex_vectors_normalized = normalize(openalex_vectors)
# -

openalex_vectors_normalized.shape

# # TFIDF
#
# We first try simple TFIDF vectors before using sentence embeddings.

tfidf_vectorizer = TfidfVectorizer(
    input="content", max_df=0.5, min_df=1, stop_words="english"
)

t0 = time()
tfidf_vectors = tfidf_vectorizer.fit_transform(openalex_docs)
print(f"vectorization done in {time() - t0:.3f} s")
print(f"n_samples: {tfidf_vectors.shape[0]}, n_features: {tfidf_vectors.shape[1]}")


def apply_umap_and_cluster(vectors, seed=None):
    if seed is not None:
        np.random.seed(seed)

    umap_vectors = umap.UMAP(
        n_neighbors=15, n_components=50, random_state=seed
    ).fit_transform(vectors)

    # use hdbscan to cluster, and assing all points to a cluster
    clusters = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    ).fit(umap_vectors)

    # use umap to reduce to 2-d
    umap_2d = umap.UMAP(
        n_neighbors=15, n_components=2, random_state=seed
    ).fit_transform(vectors)

    return clusters.labels_, umap_2d


tfidf_vectors_labels, tfidf_vectors_2d = apply_umap_and_cluster(tfidf_vectors, seed=42)

openalex_clusters = (
    openalex_text_df.assign(cluster_tfidf=tfidf_vectors_labels)
    .assign(x=tfidf_vectors_2d[:, 0])
    .assign(y=tfidf_vectors_2d[:, 1])
)

# Around ~30-40% of observations are not assigned to a cluster.

openalex_clusters["cluster_tfidf"].value_counts(normalize=True)


def plot_clusters(df, cluster_label, x_col, y_col, plot_noise=False):
    plot_df = df.copy()

    if plot_noise:
        # Set an opacity column based on the cluster_label
        plot_df["cluster_name"] = plot_df[cluster_label]
        plot_df["opacity"] = plot_df[cluster_label].apply(
            lambda x: 0.3 if x == -1 else 1.0
        )

        plot_df["color"] = plot_df[cluster_label].apply(
            lambda x: "noise" if x == -1 else x
        )
    else:
        plot_df["cluster_name"] = plot_df[cluster_label]
        plot_df = plot_df[plot_df[cluster_label] != -1]
        plot_df["opacity"] = 1.0
        plot_df["color"] = plot_df[cluster_label]

    fig = (
        alt.Chart(plot_df)
        .mark_circle()
        .encode(
            x=x_col,
            y=y_col,
            color=alt.Color("color:N", legend=alt.Legend(title=cluster_label)),
            opacity="opacity:Q",
            tooltip=["title", cluster_label],
        )
        .properties(width=800, height=600)
        .interactive()
    )

    return fig


fig1 = plot_clusters(openalex_clusters, "cluster_tfidf", "x", "y", plot_noise=True)
fig1

# # Embeddings

openalex_docs[0:10]

vectors_labels, vectors_2d = apply_umap_and_cluster(
    openalex_vectors_normalized, seed=42
)

# +
openalex_clusters = (
    openalex_clusters.assign(cluster_minilm=vectors_labels)
    .assign(x_minilm=vectors_2d[:, 0])
    .assign(y_minilm=vectors_2d[:, 1])
)

openalex_clusters.head()
# -

# About 30% of observations are in the noise cluster, which seems like an improvement!

openalex_clusters["cluster_minilm"].value_counts(normalize=True)

fig2 = plot_clusters(
    openalex_clusters, "cluster_minilm", "x_minilm", "y_minilm", plot_noise=True
)
fig2

# +
# Group the data by 'cluster_minilm' and aggregate the text for each cluster
grouped_text = (
    openalex_clusters.groupby("cluster_minilm")
    .agg({"text": " ".join, "title": "size"})
    .reset_index()
    .rename(columns={"title": "text_count"})
)
# openalex_clusters.groupby('cluster_minilm')['text'].apply(' '.join).reset_index()

grouped_text.head()

# +
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_df=0.5, stop_words="english")

# Fit and transform the aggregated text for each cluster
tfidf_matrix = vectorizer.fit_transform(grouped_text["text"])

# Extract the top terms for each cluster
top_terms = {}
feature_names = vectorizer.get_feature_names_out()
for i, row in enumerate(tfidf_matrix.toarray()):
    top_terms[grouped_text.iloc[i]["cluster_minilm"]] = [
        feature_names[index] for index in row.argsort()[-10:][::-1]
    ]

top_terms_df = pd.DataFrame(top_terms).T.reset_index()
top_terms_df.columns = ["cluster_minilm"] + [f"term_{i+1}" for i in range(10)]
# -

# Observations:
# * "digital" is a key term in the noise cluster, so we may want to explore the noise a bit more
# * 31 is about autism and ADHD
# * 30 mentions "screening" - may be worth looking into
# * 37 is promising: all about digital technologies
# * 38 seems to be about exposure to devices
# * 24 looks to be all about robots!

top_terms_df.merge(grouped_text, on="cluster_minilm").sort_values(
    "text_count", ascending=False
).head(50)

# # Write artefacts to s3

# +
from io import StringIO

csv_buffer = StringIO()
openalex_clusters.to_csv(csv_buffer)
s3_resource = boto3.resource("s3")
s3_resource.Object(
    S3_BUCKET,
    f"inputs/data/openAlex/openalex_texts_and_clusters_{CONCEPTS}_year-2023.csv",
).put(Body=csv_buffer.getvalue())
# -

csv_buffer = StringIO()
top_terms_df.to_csv(csv_buffer)
s3_resource = boto3.resource("s3")
s3_resource.Object(
    S3_BUCKET,
    f"inputs/data/openAlex/openalex_top_terms_per_cluster_{CONCEPTS}_year-2023.csv",
).put(Body=csv_buffer.getvalue())
