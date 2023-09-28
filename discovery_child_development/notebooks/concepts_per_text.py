# %%
import pandas as pd
import boto3
from io import BytesIO
import os
from dotenv import load_dotenv
from time import time
import umap.umap_ as umap
import hdbscan
import altair as alt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
load_dotenv()
S3_BUCKET = os.environ["S3_BUCKET"]
s3_client = boto3.client("s3")

# %%
CONCEPTS = "C109260823|C2993937534|C2777082460|C2911196330|C2993037610|C2779415726|C2781192327|C15471489|C178229462"

# %%
data_path = f"inputs/data/openAlex/concepts/concepts_metadata_{CONCEPTS}_year-2023.csv"
response = s3_client.get_object(Bucket=S3_BUCKET, Key=data_path)
csv_data = response["Body"].read()
concepts_df = pd.read_csv(BytesIO(csv_data), index_col=0)
concepts_df.head()

# %%
concepts_cleaned_df = concepts_df[concepts_df["score"] > 0]

# %% [markdown]
# # Using score and concept names

# %%
# Pivot the DataFrame
pivot_df = concepts_cleaned_df.pivot(
    index="openalex_id", columns="display_name", values="score"
)

# Fill NaN values with 0
pivot_df = pivot_df.fillna(0)

pivot_df.head()

# %%
# Convert the DataFrame to a numpy array
data_array = pivot_df.values

data_array.shape

# %%
# reduce dimensionality of the vectors
score_vectors_50 = umap.UMAP(
    n_neighbors=15, n_components=50, random_state=42
).fit_transform(data_array)

# %%
score_clusters = hdbscan.HDBSCAN(
    min_cluster_size=15,  # min_samples=15,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
).fit(score_vectors_50)

# %%
score_vectors_2d = umap.UMAP(
    n_neighbors=15, n_components=2, random_state=42
).fit_transform(data_array)

# %%
score_df = concepts_cleaned_df[["openalex_id", "title"]].drop_duplicates()

score_df = score_df.assign(
    cluster=score_clusters.labels_, x=score_vectors_2d[:, 0], y=score_vectors_2d[:, 1]
)

score_df["opacity"] = score_df["cluster"].apply(lambda x: 0.3 if x == -1 else 1.0)
score_df["color"] = score_df["cluster"].apply(lambda x: "noise" if x == -1 else x)

score_df.head()

# %%
fig = (
    alt.Chart(score_df)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("color:N", legend=alt.Legend(title="cluster")),
        opacity="opacity:Q",
        tooltip=["title", "cluster"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig

# %%

# %%

# %% [markdown]
# # Embed the combination of concepts attached to each abstract

# %%
# For each openalex_id, concatenate all the concept_id values
concepts_per_text = (
    concepts_cleaned_df.groupby(["openalex_id", "title"])["display_name"]
    .apply(lambda x: ",".join(x))
    .reset_index()
)

# Rename columns for clarity
concepts_per_text.columns = ["openalex_id", "title", "concepts"]

concepts_per_text.head()

# %%
# prepare to embed the concepts
concepts_per_text_list = concepts_per_text["concepts"].tolist()

# %%
import sentence_transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

t0 = time()
concepts_per_text_384 = model.encode(concepts_per_text_list, show_progress_bar=True)
print(f"vectorization done in {time() - t0:.3f} s")

# %%
concepts_per_text_50 = umap.UMAP(
    n_neighbors=15, n_components=25, random_state=42
).fit_transform(concepts_per_text_384)

# %%
concept_per_text_clusters = hdbscan.HDBSCAN(
    min_cluster_size=15,  # min_samples=15,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
).fit(concepts_per_text_50)

# %%
concepts_per_text_2d = umap.UMAP(
    n_neighbors=15, n_components=2, random_state=42
).fit_transform(concepts_per_text_384)

# %%
concepts_per_text = concepts_per_text.assign(
    cluster=concept_per_text_clusters.labels_,
    x=concepts_per_text_2d[:, 0],
    y=concepts_per_text_2d[:, 1],
)

concepts_per_text["opacity"] = concepts_per_text["cluster"].apply(
    lambda x: 0.3 if x == -1 else 1.0
)
concepts_per_text["color"] = concepts_per_text["cluster"].apply(
    lambda x: "noise" if x == -1 else x
)

# %% [markdown]
# From the figure:
# * Cluster 45 is about education and technology
# * 44 also has a few papers about devices / gadgets

# %%
fig2 = (
    alt.Chart(concepts_per_text)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("color:N", legend=alt.Legend(title="cluster")),
        opacity="opacity:Q",
        tooltip=["title", "cluster"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig2

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
t0 = time()
tfidf_vectors = tfidf_vectorizer.fit_transform(concepts_per_text_list)
print(f"vectorization done in {time() - t0:.3f} s")
print(f"n_samples: {tfidf_vectors.shape[0]}, n_features: {tfidf_vectors.shape[1]}")

# %%
concepts_per_text_50 = umap.UMAP(
    n_neighbors=15, n_components=50, random_state=42
).fit_transform(tfidf_vectors)

# %%
concept_per_text_clusters = hdbscan.HDBSCAN(
    min_cluster_size=15,  # min_samples=15,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
).fit(concepts_per_text_50)

# %%
concepts_per_text_2d = umap.UMAP(
    n_neighbors=15, n_components=2, random_state=42
).fit_transform(tfidf_vectors)

# %%
concepts_per_text.head()

# %%

# %%
concepts_per_text.head()

# %%
fig = (
    alt.Chart(concepts_per_text)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("color:N", legend=alt.Legend(title="cluster")),
        opacity="opacity:Q",
        tooltip=["title", "cluster"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
concepts_per_text = concepts_per_text.assign(
    cluster=concept_per_text_clusters.labels_,
    x=concepts_per_text_2d[:, 0],
    y=concepts_per_text_2d[:, 1],
)

concepts_per_text["opacity"] = concepts_per_text["cluster"].apply(
    lambda x: 0.3 if x == -1 else 1.0
)
concepts_per_text["color"] = concepts_per_text["cluster"].apply(
    lambda x: "noise" if x == -1 else x
)

# %%
concepts_per_text.head()

# %%
fig = (
    alt.Chart(concepts_per_text)
    .mark_circle()
    .encode(
        x="x",
        y="y",
        color=alt.Color("color:N", legend=alt.Legend(title="cluster")),
        opacity="opacity:Q",
        tooltip=["concepts", "cluster"],
    )
    .properties(width=800, height=600)
    .interactive()
)

fig

# %%
